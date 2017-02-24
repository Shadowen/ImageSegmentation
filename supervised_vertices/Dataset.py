import skimage.draw
import skimage.measure
import numpy as np


def get_train_and_valid_datasets(filename):
    data = np.load(filename)
    print('{} polygons loaded.'.format(data.shape[0]))
    valid_size = data.shape[0] // 10
    training_data = data[valid_size:]
    validation_data = data[:valid_size]
    del data  # Make sure we don't contaminate the training set
    print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))

    return Dataset(training_data), Dataset(validation_data)


class Dataset():
    def __init__(self, data):
        self.data = data

    def get_batch(self, batch_size=50, max_timesteps=10):
        """
        :param batch_size:
        :param max_timesteps:
        :return: tuple(batch_d, batch_x, batch_t)
        where
            batch_d is a NumPy array of shape [batch_size]
            batch_x is a NumPy array of shape [batch_size, max_timesteps, 32, 32, 3]
            batch_t is a NumPy array of shape [batch_size, max_timesteps, 32, 32]
        """
        batch_indices = np.random.choice(self.data.shape[0], batch_size, replace=False)

        batch_d = np.zeros([batch_size])
        batch_x = np.zeros([batch_size, max_timesteps, 32, 32, 3])
        batch_t = np.zeros([batch_size, max_timesteps, 32, 32])
        for idx, (vertices, truth) in enumerate(self.data[batch_indices]):
            d, x, t = self.create_sample_sequence(32, vertices, truth)
            batch_d[idx] = d
            batch_x[idx, :d, ::] = x
            batch_t[idx, :d, ::] = t
        return batch_d, batch_x, batch_t

    def create_sample_sequence(self, image_size, poly_verts, ground_truth):
        total_num_verts = len(poly_verts)

        image = _create_image(ground_truth)
        start_idx = np.random.randint(total_num_verts + 1)
        # Probably should optimize this with a numpy array and clever math. Use np.roll
        poly_verts = poly_verts[start_idx:] + poly_verts[:start_idx]

        inputs = np.empty([total_num_verts, image_size, image_size, 3])
        outputs = np.empty([total_num_verts, image_size, image_size], dtype=np.uint16)
        for idx in range(total_num_verts):
            history_mask = _create_history_mask(poly_verts, idx + 1, image_size)
            cursor_mask = _create_point_mask(poly_verts[idx], image_size)

            state = np.stack([image, history_mask, cursor_mask], axis=2)
            next_point = np.array(poly_verts[(idx + 1) % total_num_verts])

            inputs[idx, :, :] = state
            outputs[idx, :, :] = _create_point_mask(next_point, image_size)

        return total_num_verts, inputs, outputs


def _create_image(ground_truth):
    """ Apply distortion to the ground truth to generate the image the algorithm will see. """
    image = np.copy(ground_truth)

    # # Salt and pepper noise
    # amount = 0.03
    # num_salt = np.ceil(amount * image.size)
    # coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    # image[coords] = 1
    #
    # coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    # image[coords] = 0
    #
    # # Blur
    # image = scipy.ndimage.filters.gaussian_filter(image, np.random.random() * 2)

    return image


def _create_history_mask(vertices, num_points, image_size, use_lines=True):
    if use_lines:
        player_mask = np.zeros([image_size, image_size])
        for i in range(1, num_points):
            rr, cc = skimage.draw.line(*vertices[i - 1], *vertices[i])
            player_mask[rr, cc] = 1
        return player_mask
    else:  # TODO remove this?
        player_mask = np.zeros([image_size, image_size])
        for i in range(1, num_points):
            player_mask[tuple(vertices[i])] = 1
        return player_mask


def _create_point_mask(point, image_size):
    mask = np.zeros([image_size, image_size])
    mask[tuple(point)] = 1
    return mask
