import matplotlib.pyplot as plt
import numpy as np

from concave_hull import MIN_K, find_concave_hull

np.random.seed(420)

points = np.random.uniform(size=(250, 2))

print("Example 1")
# Example 1, automatically determine smallest valid k value starting at minimum
# possible value
(hull, k) = find_concave_hull(points, k=MIN_K)

plt.figure()
plt.scatter(points[:, 0], points[:, 1], s=20)
plt.plot([p.x for p in hull], [p.y for p in hull], "r")
plt.title(f"Concave hull with k = {k}")
plt.savefig("hull_eg1.png")
plt.show()

print("\nExample 2")
# Example 2, use maximum possible k value to recover convex hull
(hull, k) = find_concave_hull(points, k=(len(points) - 1))

plt.figure()
plt.scatter(points[:, 0], points[:, 1], s=20)
plt.plot([p.x for p in hull], [p.y for p in hull], "r")
plt.title(f"Concave hull with k = {k}")
plt.savefig("hull_eg2.png")
plt.show()
