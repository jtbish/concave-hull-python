import logging
from collections import namedtuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

MIN_K = 3
MIN_DIST = 0
MIN_ANG_INCL = 0
MAX_ANG_EXCL = (2 * np.pi)
# re-add first point to candidates after 3 steps (i.e. after 3 edges of polygon
# created) to avoid just creating triangle back to first point
FIRST_POINT_RE_ADD_STEP = 3
RAY_Y_EPSILON = 1e-5

Point2D = namedtuple("Point2D", ["x", "y"])

LineSegment2D = namedtuple("LineSegment2D", ["p1", "p2"])


class ConcaveHullException(Exception):
    pass


def find_concave_hull(points, k=MIN_K):
    """Finds the concave hull for given points, starting at the given k value.

    The points argument is assumed to be an iterable where each element of the
    iterable is itself a 2D iterable representing an (x, y) point in 2D space.
    This means the function can accept varying types of points inputs: e.g.
    a list of 2-tuples or a numpy ndarray with 2 columns, etc.

    Assuming the number of *unique* elements in points is n,
    the function automatically determines the smallest value of k that produces
    a valid concave hull, incrementing k by 1 and re-trying if an invalid hull
    is produced.
    Since k represents the number of nearest neighbours to consider when adding
    edges to the hull, this automatic incrementing stops when k = (n - 1), as
    if there are n unique points there are a maximum of (n - 1) neighbours.

    If no valid hull is found after trying all possible values of k, then an
    exception is raised. This probably indicates the data is somehow not
    appropriate for the algorithm.

    Returns a 2-tuple, containing:

        1. A list of the found concave hull vertices as Point2D objects,
        which can be connected in order to form the hull edges.
        The first and last elements of this list are the same, as the hull
        forms a polygon around the given points.

        2. The value of k used to successfully generate this concave hull.
    """

    k = int(k)
    if k < MIN_K:
        raise Exception(f"k must be at least {MIN_K}")

    points = _convert_points(points)
    logging.info(f"Given points list contains {len(points)} points...")
    points = _remove_dup_points(points)
    logging.info(f"of which {len(points)} are unique")

    if len(points) < 3:
        raise Exception("A minimum of 3 dissimilar points is required")
    if len(points) == 3:
        logging.warning("Only 3 points, so concave hull is points themselves")
        return points

    logging.info("Computing distance matrix...")
    dist_mat = _compute_dist_mat(points)
    logging.info("Computing sorted neighbour map...")
    sn_map = _compute_sorted_neighbour_map(points, dist_mat)

    first_point = _find_min_y_point(points)
    max_k = (len(points) - 1)
    k = min(k, max_k)

    hull = None
    found_hull = False
    while (not found_hull and k <= max_k):
        try:
            hull = _find_concave_hull(points, sn_map, first_point, k)
        except ConcaveHullException:
            logging.info("Error encountered, incrementing value of k")
            k += 1
        else:
            found_hull = True

    if hull is None:
        raise Exception(
            "Unable to find hull after trying all possible values of k")
    else:
        logging.info(f"Success! Hull found with k = {k}")
        return (hull, k)


def _convert_points(points):
    return [Point2D(x, y) for (x, y) in points]


def _remove_dup_points(points):
    uniques = set()
    res = []
    for p in points:
        if p not in uniques:
            uniques.add(p)
            res.append(p)
    return res


def _compute_dist_mat(points):
    """Computes symmetric distance matrix of size (n * n) for set of points of
    size n."""
    n = len(points)
    dist_mat = np.full(shape=(n, n), fill_value=np.nan)

    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                # copy transpose element if it is not NaN
                transpose_elem = dist_mat[j][i]
                if not np.isnan(transpose_elem):
                    dist_mat[i][j] = transpose_elem
                else:
                    point_diff_arr = (np.asarray(points[i]) -
                                      np.asarray(points[j]))
                    dist_mat[i][j] = np.linalg.norm(point_diff_arr)
            else:
                dist_mat[i][j] = MIN_DIST

    any_nans = np.isnan(dist_mat).any()
    assert not any_nans
    return dist_mat


def _compute_sorted_neighbour_map(points, dist_mat):
    """Computes the 'sorted neighbour map' of the given set of points using the
    distance matrix.
    This map has each of the n points as a key, with each value being the
    (n - 1) neighbours of that point sorted in ascending order of distance.
    This makes k-NN queries trivial later on."""
    n = len(points)
    sn_map = {}

    for (this_idx, this_point) in enumerate(points):
        dist_vals_arr = dist_mat[this_idx, :]
        dist_vals_w_idxs = list(enumerate((dist_vals_arr)))

        # sort neighbour idxs by dist ascending
        sorted_dist_vals_w_idxs = sorted(dist_vals_w_idxs,
                                         key=lambda tup: tup[1],
                                         reverse=False)

        # make sure first entry is this_idx with MIN_DIST, then remove it
        assert sorted_dist_vals_w_idxs[0][0] == this_idx
        assert sorted_dist_vals_w_idxs[0][1] == MIN_DIST
        del sorted_dist_vals_w_idxs[0]

        # add the actual neighbour points in order
        sn_map[this_point] = []
        for (other_idx, _) in sorted_dist_vals_w_idxs:
            assert other_idx != this_idx
            other_point = points[other_idx]
            sn_map[this_point].append(other_point)
        assert len(sn_map[this_point]) == (n - 1)

    assert len(sn_map) == n
    return sn_map


def _find_min_y_point(points):
    return min(points, key=lambda p: p.y)


def _find_concave_hull(all_points, sn_map, first_point, k):
    logging.info(f"Trying to find concave hull with k = {k}")

    hull = [first_point]  # vertices of polygon formed by hull
    line_segments = []  # edges of polygon formed by hull
    curr_point = first_point
    # candidates are points that could possibly be added as hull vertices
    candidate_points = set(all_points)
    candidate_points.remove(first_point)

    step = 0
    revisited_first_point = False

    while not revisited_first_point:

        if step == FIRST_POINT_RE_ADD_STEP:
            candidate_points.add(first_point)

        k_nearest_cands = _find_k_nearest_candidates(curr_point, sn_map,
                                                     candidate_points, k)
        last_lsegm = _get_last_lsegm(line_segments)
        cand_angles = _calc_cand_angles(curr_point, last_lsegm,
                                        k_nearest_cands)
        cands_with_angles = list(zip(k_nearest_cands, cand_angles))
        # sort candidates by their left-hand turn angles in ascending order
        # (i.e. assuming minimising left-hand turn angle),
        # which is equivalent to maximising right-hand turn angles as in
        # the paper.
        sorted_cands_with_angles = sorted(cands_with_angles,
                                          key=lambda tup: tup[1],
                                          reverse=False)

        next_point = _find_next_point(curr_point, line_segments,
                                      sorted_cands_with_angles)
        next_lsegm = LineSegment2D(p1=curr_point, p2=next_point)
        line_segments.append(next_lsegm)
        candidate_points.remove(next_point)
        hull.append(next_point)

        curr_point = next_point
        revisited_first_point = (curr_point == first_point)
        step += 1

    assert step == len(line_segments)
    _check_hull_validity(hull, all_points, candidate_points, line_segments,
                         first_point)

    return hull


def _find_k_nearest_candidates(curr_point, sn_map, candidate_points, k):
    sorted_neighbours = sn_map[curr_point]
    candidate_sorted_neighbours = [
        p for p in sorted_neighbours if p in candidate_points
    ]
    assert len(candidate_sorted_neighbours) > 0
    # slicing op. implicitly takes into account len of iterable so can't take
    # more than len(iterable) things
    k_nearest_cands = candidate_sorted_neighbours[0:k]
    return k_nearest_cands


def _get_last_lsegm(line_segments):
    try:
        last_lsegm = line_segments[-1]
    except IndexError:
        last_lsegm = None
    return last_lsegm


def _calc_cand_angles(curr_point, last_lsegm, k_nearest_cands):
    # the current point is the pole / origin around which angles need to be
    # measured
    p0 = curr_point
    rot_ang = _calc_rot_angle(p0, last_lsegm)

    cand_angles = [
        _calc_cand_angle(p0, rot_ang, cand) for cand in k_nearest_cands
    ]
    return cand_angles


def _calc_rot_angle(p0, last_lsegm):
    if last_lsegm is not None:

        # end point of last lsegm must be current point
        assert last_lsegm.p2 == p0
        p1 = last_lsegm.p1

        p1_trans = _translate_point(p0, p1)
        # counter-clockwise angle of p1_trans from pos. x axis
        p1_trans_ang = _my_arctan2(p1_trans.y, p1_trans.x)
        # remaining counter-clockwise angle required to get p1 back to lie on
        # pos. x axis
        rot_ang = (2 * np.pi) - p1_trans_ang
        return rot_ang

    else:
        # no rotation required since no last lsegm and therefore already
        # implicitly using pos. x axis to measure candidate angles from
        return 0


def _calc_cand_angle(p0, rot_ang, cand):
    # apply translation then rotation to cand before measuring angle
    cand_trans = _translate_point(p0=p0, p1=cand)
    cand_trans_rot = _rotate_point(cand_trans, theta=rot_ang)
    return _my_arctan2(cand_trans_rot.y, cand_trans_rot.x)


def _translate_point(p0, p1):
    """Translate p1 considering p0 as the origin."""
    (x0, y0) = p0
    (x1, y1) = p1
    return Point2D(x=(x1 - x0), y=(y1 - y0))


def _rotate_point(p, theta):
    if theta == 0:
        return p
    else:
        (x, y) = p
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_dash = x * cos_theta - y * sin_theta
        y_dash = y * cos_theta + x * sin_theta
        return Point2D(x=x_dash, y=y_dash)


def _my_arctan2(y, x):
    """arctan2 function that returns range of [0, 2*pi) instead of [-pi, pi].
    This allows reflex angles to be indentified on increasing (rather than
    decreasing) scale.
    Note order of args!"""
    ang = np.arctan2(y, x)
    if ang < 0:
        ang += (2 * np.pi)
    assert MIN_ANG_INCL <= ang < MAX_ANG_EXCL
    return ang


def _find_next_point(curr_point, line_segments, sorted_cands_with_angles):
    """Finds first point in candidates that is valid, i.e. produces a next line
    segment that does not intersect with any existing line segment."""
    for (cand_next_point, _) in sorted_cands_with_angles:
        if _is_valid_point(curr_point, line_segments, cand_next_point):
            return cand_next_point

    msg = "No valid next point found"
    logging.error(msg)
    raise ConcaveHullException(msg)


def _is_valid_point(curr_point, line_segments, cand_next_point):
    """Checks whether adding candidate point to polygon results in next line
    segment intersecting any existing line segment."""
    next_lsegm = LineSegment2D(p1=curr_point, p2=cand_next_point)
    # Iter backwards over line segments, as more likely to intersect with more
    # recent segments so more likely to exit the function earlier this way.
    # Start at second most recent segment since impossible to intersect with
    # most recent one (excluding the necessary shared endpoint!).
    m = len(line_segments)
    for idx in range((m - 2), -1, -1):
        lsegm_to_check = line_segments[idx]
        # Handle edge case where lsegm_to_check is the first lsegm and the
        # next_lsegm trying to add is connecting to it (i.e. completing the
        # hull). In this case, they will share an overlapping point (and so
        # technically intersect), but obviously need to allow this.
        if (idx == 0 and (lsegm_to_check.p1 == next_lsegm.p2)):
            continue
        else:
            if _do_lsegms_intersect(lsegm_a=lsegm_to_check,
                                    lsegm_b=next_lsegm):
                return False
    return True


def _check_hull_validity(hull, all_points, candidate_points, line_segments,
                         first_point):
    _check_hull_linkage(hull, first_point, line_segments)
    # remove duplicate end point for purposes of remaining checks
    hull = hull[0:-1]

    # hull + candidate_points should now be equal to all points,
    # since candidate points should now contain all the points that didn't get
    # included in the hull
    non_hull_points = list(candidate_points)
    _check_hull_points(all_points, hull, non_hull_points)

    # now, check that all non-hull points are enclosed by hull polygon
    # via ray casting algorithm.
    # "inf" x val for the rays only needs to be larger than largest x val in
    # the hull
    hull_x_max = max([p.x for p in hull])
    x_inf = (hull_x_max * 2)

    for p in non_hull_points:
        if not _is_point_in_polygon(p, line_segments, x_inf):
            msg = "Point not in polygon formed by hull"
            logging.error(msg)
            raise ConcaveHullException(msg)


def _check_hull_linkage(hull, first_point, line_segments):
    # check first point inclusion in hull points
    assert hull[0] == first_point
    assert hull[-1] == first_point

    # check that all line segments share end and starting points
    m = len(line_segments)
    for this_idx in range(0, m - 1):
        next_idx = this_idx + 1
        this_lsegm = line_segments[this_idx]
        next_lsegm = line_segments[next_idx]
        assert this_lsegm.p2 == next_lsegm.p1
    assert line_segments[m - 1].p2 == line_segments[0].p1


def _check_hull_points(all_points, hull, non_hull_points):
    assert sorted(all_points) == sorted(hull + non_hull_points)
    # also be pedantic and check empty intersection between hull and non-hull
    # points
    intersection = set(hull).intersection(set(non_hull_points))
    assert len(intersection) == 0


def _is_point_in_polygon(p, line_segments, x_inf):
    """Checks if point p is in polygon defined by line_segments using ray
    casting. "Inf" x val for rays given by x_inf.
    Handles edge cases of:
        1. Point is located exactly on an edge of the polygon, in which case we
        choose to say that it is *inside* the polygon.
        2. Point y val is equal to that of any of the polygon vertex y vals,
        where if this is not handled intersections would be double-counted,
        causing points that are actually inside the polygon to be incorrectly
        deemed as outside the polygon."""
    num_intersections = 0

    for lsegm in line_segments:
        (pi, pj) = lsegm
        pk = p

        if (pk.y == pi.y or pk.y == pj.y):
            # handle edge case where point is vertically inline with one of
            # the vertices of the current polygon edge, meaining ray
            # intersection will be double-counted because edge endpoints are
            # shared. fix by slightly perturbing point y val in positive
            # direction so it could only possibly be counted once for the
            # "upper" edge (if it actually is in the polygon - if outside the
            # count will now give 0 since it has effectively been moved
            # slightly upwards outside edge boundary).
            pk = Point2D(x=pk.x, y=(pk.y + RAY_Y_EPSILON))

        inf_point = Point2D(x=x_inf, y=pk.y)
        horiz_lsegm = LineSegment2D(p1=pk, p2=inf_point)

        if _do_lsegms_intersect(lsegm_a=lsegm, lsegm_b=horiz_lsegm):
            num_intersections += 1

            # Given that there is an intersection (that cannot be due to
            # vertex intersection because this is ruled out by check above),
            # check if p actually lies on the segment defined by
            # (pi, pj). If it does, should be considered as inside the polygon
            # and don't need to do any more checks.
            is_p_colinear_with_lsegm = (_direction(pi, pj, pk) == 0)
            if is_p_colinear_with_lsegm:
                is_p_on_lsegm = _is_on_segment(pi, pj, pk)
                if is_p_on_lsegm:
                    return True

    # if made it here, point only in polygon if odd num intersections with
    # polygon edges
    odd_num_intersections = (num_intersections % 2 == 1)
    return odd_num_intersections


def _do_lsegms_intersect(lsegm_a, lsegm_b):
    """SEGMENTS-INTERSECT algorithm (and associated sub-algs),
    from Sec. 33.1 of CLRS textbook 3rd. ed."""
    # setup points as per alg.
    p1 = lsegm_a.p1
    p2 = lsegm_a.p2
    p3 = lsegm_b.p1
    p4 = lsegm_b.p2

    d1 = _direction(p3, p4, p1)
    d2 = _direction(p3, p4, p2)
    d3 = _direction(p1, p2, p3)
    d4 = _direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or
                                                         (d3 < 0 and d4 > 0)):
        return True
    elif d1 == 0 and _is_on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and _is_on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and _is_on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and _is_on_segment(p1, p2, p4):
        return True
    else:
        return False


def _direction(pi, pj, pk):
    # (pk - pi)
    p1 = _translate_point(p0=pi, p1=pk)
    # (pj - pi)
    p2 = _translate_point(p0=pi, p1=pj)

    return _cross_product(p1, p2)


def _cross_product(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return (x1 * y2) - (x2 * y1)


def _is_on_segment(pi, pj, pk):
    """Checks whether pk is on segment produced by (pi, pj).
    Assumes pk is colinear with segment (pi, pj),
    i.e. _direction(pi, pj, pk) == 0"""
    (xi, yi) = pi
    (xj, yj) = pj
    (xk, yk) = pk
    if (min(xi, xj) <= xk <= max(xi, xj)) and (min(yi, yj) <= yk <= max(
            yi, yj)):
        return True
    else:
        return False
