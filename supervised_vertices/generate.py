import numpy as np
import itertools
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import skimage.draw

import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib.path import Path


def create_polygon(image_size, shape_complexity=3):
    # Create the polygon
    polygon_points = shape_complexity
    border_size = 2
    # Ground truth polygon
    points = np.random.randint(border_size, image_size - border_size, [polygon_points, 2])
    hull = scipy.spatial.ConvexHull(points)
    poly_verts = [(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices]

    return poly_verts


def create_valid_polygon(image_size, shape_complexity, min_area):
    while True:
        try:
            poly_verts = create_polygon(image_size, shape_complexity=shape_complexity)
            # Ground truth pixel mask
            ground_truth = create_shape_mask(poly_verts, image_size)
            area = np.count_nonzero(ground_truth)
            image = ground_truth
        except:
            area = 0
        if area > min_area:
            break
    return poly_verts, ground_truth


def create_training_sample(image_size, poly_verts, ground_truth):
    # Create training examples out of it
    image = ground_truth

    total_num_verts = len(poly_verts)
    start_idx = np.random.randint(total_num_verts)
    poly_verts = poly_verts[start_idx:] + poly_verts[
                                          :start_idx]  # Probably should optimize this with a numpy array and clever
    # math    # Use np.roll

    num_points = np.random.randint(1, total_num_verts)

    player_mask = create_player_mask(poly_verts, num_points, image_size)
    cursor_mask = create_point_mask(poly_verts[num_points - 1], image_size)
    state = np.stack([image, player_mask, cursor_mask], axis=2)

    next_point = poly_verts[num_points % total_num_verts]
    # next_point_mask = create_point_mask(next_point, image_size)

    return state, next_point[0] * image_size + next_point[1]


def create_shape_mask(vertices, image_size):
    mask = np.zeros([image_size, image_size])
    rr, cc = skimage.draw.polygon(*zip(*vertices))
    mask[rr, cc] = 1
    return mask


def create_player_mask(vertices, num_points, image_size):
    player_mask = np.zeros([image_size, image_size])
    for i in range(num_points):
        rr, cc = skimage.draw.line(*vertices[i - 1], *vertices[i])
        player_mask[rr, cc] = 1
    return player_mask


def create_point_mask(point, image_size):
    mask = np.zeros([image_size, image_size])
    mask[point] = 1
    return mask


def create_sample(image_size, shape_complexity=5, allow_inverted=False):
    samples = []
    # Generate a valid polygon
    poly_verts, ground_truth = create_valid_polygon(image_size, shape_complexity, min_area=image_size * 3)
    image = ground_truth

    # Create training examples out of it
    total_num_verts = len(poly_verts)
    for start_idx in range(total_num_verts):
        poly_verts = poly_verts[1:] + [
            poly_verts[0]]  # Probably should optimize this with a numpy array and clever math
        for num_points in range(1, total_num_verts):
            player_mask = create_player_mask(poly_verts, num_points, image_size)
            cursor_mask = create_point_mask(poly_verts[num_points - 1], image_size)
            state = np.stack([image, player_mask, cursor_mask])
            next_point = poly_verts[num_points % total_num_verts]
            next_point_mask = create_point_mask(next_point, image_size)

            samples.append((state, next_point_mask))

    if not allow_inverted:
        return samples

    # The other orientation
    poly_verts = list(reversed(poly_verts))
    for start_idx in range(total_num_verts):
        poly_verts = poly_verts[1:] + [
            poly_verts[0]]  # Probably should optimize this with a numpy array and clever math
        for num_points in range(1, total_num_verts):
            player_mask = create_player_mask(poly_verts, num_points, image_size)
            cursor_mask = create_point_mask(poly_verts[0], image_size)
            state = np.stack([image, player_mask, cursor_mask])
            next_point = poly_verts[num_points % total_num_verts]
            next_point_mask = create_point_mask(next_point, image_size)

            samples.append((state, next_point_mask))

    return samples


if __name__ == '__main__':
    # samples = [np.concatenate([x[0], x[1][np.newaxis]]) for x in create_sample(image_size=32, shape_complexity=3)]
    # fig, ax = plt.subplots(nrows=len(samples), ncols=samples[0].shape[0])
    # [ax[0][e].set_title(['Image', 'History', 'Cursor', 'Next'][e]) for e in range(samples[0].shape[0])]
    # for i in range(len(samples)):
    #     for e in range(samples[i].shape[0]):
    #         ax[i][e].axis('off')
    #         ax[i][e].imshow(samples[i][e], cmap='gray', interpolation='nearest')
    # plt.show()
    # print(samples[1][0].nbytes)
    # # plt.savefig('sample_single')

    # many_samples = list(itertools.chain.from_iterable(
    #     create_sample(image_size=32, shape_complexity=5) for _ in range(1000)))  # Out of disk space?
    # # print(sum(x[0].nbytes + x[1].nbytes for x in many_samples))
    # np.save('dataset_large', many_samples)

    many_samples = [create_valid_polygon(image_size=32, shape_complexity=5, min_area=86) for _ in range(10000)]
    np.save('dataset_polygons', many_samples)
