import sys

sys.path.append('.')
import numpy as np
import skimage.draw
import scipy.ndimage
import scipy.spatial
import skimage.measure


def create_valid_polygon(image_size, shape_complexity, min_area, reduction_tolerance):
    while True:
        try:
            # Create the polygon
            points = np.random.randint(0, image_size, [shape_complexity, 2])
            hull = scipy.spatial.ConvexHull(points)
            poly_verts = np.array([(points[simplex, 0], points[simplex, 1]) for simplex in hull.vertices])
            # TODO: Hacky, should probably just change to Nx2 numpy array later
            poly_verts = list(skimage.measure.approximate_polygon(poly_verts, tolerance=reduction_tolerance))

            # Ground truth pixel mask
            ground_truth = _create_shape_mask(poly_verts, image_size)
            area = np.count_nonzero(ground_truth)
        except:
            area = 0
        if area > min_area:
            break
    return poly_verts, ground_truth


def _create_shape_mask(vertices, image_size):
    mask = np.zeros([image_size, image_size])
    rr, cc = skimage.draw.polygon(*zip(*vertices))
    mask[rr, cc] = 1
    return mask


if __name__ == '__main__':
    from supervised_vertices import analyze
    import matplotlib.pyplot as plt

    many_samples = [
        create_valid_polygon(image_size=227, shape_complexity=5, min_area=200, reduction_tolerance=5) for _
        in range(5000)]

    # Preview regular shapes
    # for i in range(len(many_samples)):
    #     plt.figure()
    #     plt.imshow(many_samples[i][1])
    #     plt.show(block=True)

    # # Preview RNN sequences
    # for i in range(len(many_samples)):
    #     inputs, outputs = create_sample_sequence(32, *many_samples[i])
    #     analyze.display_samples(x=np.rollaxis(inputs, 3, 0), ground_truths=np.rollaxis(outputs, 1, 0))

    # for i in range(len(many_samples)):
    #     x, truth = create_training_sample(32, *many_samples[i], start_idx=0, num_points=len(many_samples[i][0]))
    #     display_sample(x, truth)
    #     plt.show(block=True)

    # for i in range(len(many_samples)):
    #     xs = []
    #     truths = []
    #     for start in range(len(many_samples[i][0])):
    #         for num in range(1, len(many_samples[i][0]) + 1):
    #             print('start={}\tnum={}'.format(start, num))
    #             x, truth = create_training_sample(32, *many_samples[i], start_idx=start, num_points=num)
    #             xs.append(x)
    #             truths.append(truth)
    #     display_samples(xs, truths)
    #     plt.show(block=True)

    np.save('polygons_5-sided', many_samples)
