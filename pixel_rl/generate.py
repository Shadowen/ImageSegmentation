import numpy as np

from scipy.spatial import ConvexHull
from matplotlib.path import Path

n = 100
polygon_points = 5
image_size = 32
border_size = 2

poly_verts = np.zeros([n, polygon_points, 2])
ground_truth_array = np.empty([n, image_size, image_size])
image_array = np.empty([n, image_size, image_size])

for i in range(n):
    # Ground truth polygon
    points = np.random.randint(border_size, image_size - border_size, [polygon_points, 2])  # 30 random points in 2-D
    hull = ConvexHull(points)
    poly_verts = [(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices]

    # import matplotlib.pyplot as plt
    #
    # plt.plot(points[:, 0], points[:, 1], 'ro')
    # for v in poly_verts:
    #     plt.plot(v[0], v[1], 'k-')

    # Ground truth pixel mask
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    x, y = x.flatten(), y.flatten()
    path = Path(poly_verts)
    ground_truth = path.contains_points(np.vstack((x, y)).T).reshape([image_size, image_size])
    ground_truth_array[i] = ground_truth

    # Image
    image = ground_truth
    image_array[i] = image

np.savez('dataset_simple.npz', [image_array, poly_verts, ground_truth_array])
