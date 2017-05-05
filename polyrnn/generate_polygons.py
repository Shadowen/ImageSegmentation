import os
import shutil

import scipy.ndimage
import scipy.spatial
from skimage.io import imsave

from polyrnn.util import *


def create_valid_polygon(image_size, shape_complexity, min_area, min_distance, max_distance, reduction_tolerance):
    while True:
        try:
            # Create the polygon
            points = np.random.randint(0, image_size, [shape_complexity, 2])
            hull = scipy.spatial.ConvexHull(points)
            poly_verts = np.array([(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices])
            poly_verts = reduce_polygon(poly_verts, eps=reduction_tolerance)

            image = np.stack([create_shape_mask(poly_verts, image_size) for _ in range(3)], axis=2)
        except:
            continue

        area = np.count_nonzero(image)
        distances = [np.linalg.norm(b - a) for a, b in iterate_in_ntuples(poly_verts, n=2)]
        smallest_distance = min(distances)
        largest_distance = max(distances)
        if area > min_area and smallest_distance > min_distance and largest_distance < max_distance:
            break
    return poly_verts / image_size, image


def get_angle(a, b, c):
    """Gets the angle between points a, b, and c in degrees."""
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def reduce_polygon(verts, eps):
    """Deletes any vertices with interior angles closer than eps to 180 degrees."""
    keep_reducing = True
    while keep_reducing:
        keep_reducing = False
        for i, (a, b, c) in enumerate(iterate_in_ntuples(verts, n=3, offset=-1)):
            angle = get_angle(a, b, c)
            if abs(180 - angle) < eps:
                verts = np.delete(verts, i, 0)
                keep_reducing = True
    return verts


if __name__ == '__main__':
    # Test
    lst = [1, 2, 3, 4, 5, 6, 7]
    assert (len(list(iterate_in_ntuples(lst, n=3, offset=-1))) == len(lst))
    assert (abs(get_angle(np.array([0, 0]), np.array([1, 1]), np.array([2, 2])) - 180) < 0.01)

    # Execute
    save_dir = '/home/wesley/data/tiny-polygons'
    num_images = 100
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print('Generating images...')
    for i in range(num_images):
        print('\rImage {}/{}...'.format(i + 1, num_images), end='')
        vertices, image = create_valid_polygon(image_size=224, shape_complexity=5, min_distance=10, max_distance=100,
                                               min_area=10,
                                               reduction_tolerance=30)
        imsave('{}/{}.jpg'.format(save_dir, i), image)
        np.save('{}/{}.npy'.format(save_dir, i), vertices)
    print('Done!')
