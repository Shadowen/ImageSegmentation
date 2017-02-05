import numpy as np
import skimage.draw
import scipy.ndimage
import scipy.spatial
import matplotlib.pyplot as plt


def create_valid_polygon(image_size, shape_complexity, min_area):
    while True:
        try:
            # Create the polygon
            points = np.random.randint(0, image_size, [shape_complexity, 2])
            hull = scipy.spatial.ConvexHull(points)
            poly_verts = [(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices]

            # Ground truth pixel mask
            ground_truth = create_shape_mask(poly_verts, image_size)
            area = np.count_nonzero(ground_truth)
        except:
            area = 0
        if area > min_area:
            break
    return poly_verts, ground_truth


def create_training_sample(image_size, poly_verts, ground_truth, start_idx=None, num_points=None):
    total_num_verts = len(poly_verts)
    if start_idx is None:
        start_idx = np.random.randint(total_num_verts + 1)
    if num_points is None:
        num_points = np.random.randint(1, total_num_verts + 1)
    elif num_points == 0:
        raise 'Cannot have 0 points to start...'

    image = create_image(ground_truth)
    poly_verts = poly_verts[start_idx:] + poly_verts[:start_idx]
    # Probably should optimize this with a numpy array and clever math. Use np.roll

    player_mask = create_history_mask(poly_verts, num_points, image_size)
    cursor_mask = create_point_mask(poly_verts[num_points - 1], image_size)
    state = np.stack([image, player_mask, cursor_mask], axis=2)

    next_point = np.array(poly_verts[num_points % total_num_verts])

    return state, next_point


def create_shape_mask(vertices, image_size):
    mask = np.zeros([image_size, image_size])
    rr, cc = skimage.draw.polygon(*zip(*vertices))
    mask[rr, cc] = 1
    return mask


def create_history_mask(vertices, num_points, image_size):
    player_mask = np.zeros([image_size, image_size])
    for i in range(1, num_points):
        rr, cc = skimage.draw.line(*vertices[i - 1], *vertices[i])
        player_mask[rr, cc] = 1
    return player_mask


def create_point_mask(point, image_size):
    mask = np.zeros([image_size, image_size])
    mask[tuple(point)] = 1
    return mask


def create_image(ground_truth):
    """ Apply distortion to the ground truth to generate the image the algorithm will see. """
    image = np.copy(ground_truth)

    # Salt and pepper noise
    amount = 0.03
    num_salt = np.ceil(amount * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords] = 1

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords] = 0

    # Blur
    image = scipy.ndimage.filters.gaussian_filter(image, np.random.random() * 2)

    return image


if __name__ == '__main__':
    many_samples = [create_valid_polygon(image_size=32, shape_complexity=5, min_area=86) for _ in range(10000)]

    # from supervised_vertices.analyze import display_sample
    # for start in range(len(many_samples[0][0])):
    #     for num in range(1, len(many_samples[0][0]) + 1):
    #         print('start={}\tnum={}'.format(start, num))
    #         x, truth = create_training_sample(32, *many_samples[0], start_idx=start, num_points=num)
    #         display_sample(x, truth)
    # plt.show(block=True)

    np.save('dataset_polygons', many_samples)
