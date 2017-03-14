import skimage.draw
import skimage.measure
import numpy as np


def get_train_and_valid_datasets(filename, local=True):
    """
    :param filename:
    :return:
    """
    if not local:
        import json
        import os
        import matplotlib.image as mpimage
        from supervised_vertices.generate import _create_shape_mask
        # Load from the CS cluster
        original_directory = os.getcwd()
        os.chdir(filename)

        datasets = []
        for usage in ['train', 'val']:
            data = []
            images = []

            for number in filter(lambda s: s.endswith('.info'), os.listdir('{}/{}/info'.format(filename, usage))):
                # Read the file
                with open('{}/{}/info/{}'.format(filename, usage, number), 'r') as f:
                    contents = json.load(f)
                segmentation, patch_path = contents['segmentation'], contents['patch_path']
                image = mpimage.imread(patch_path)
                # Convert to correct format
                image_size = image.shape[0]
                # TODO(wheung) figure out why rounding is necessary for segmentation->poly_vert conversion
                poly_verts = np.array(np.roll(np.array(segmentation[0]), 1, axis=1) * image_size).astype(np.int32)
                ground_truth = _create_shape_mask(poly_verts, image_size)
                # Store it
                data.append((poly_verts, ground_truth))
                images.append(image)
            datasets.append(Dataset(np.array(data), image_size=image_size, images=np.array(images)))

        print('{} polygons loaded from {}'.format((sum(map(len, datasets))), filename))
        print('{} for training. {} for validation.'.format(len(datasets[0]), len(datasets[1])))
        os.chdir(original_directory)
        return datasets
    else:
        data = np.load(filename)
        print('{} polygons loaded from {}.'.format(data.shape[0], filename))
        valid_size = data.shape[0] // 10
        training_data = data[valid_size:]
        validation_data = data[:valid_size]
        del data  # Make sure we don't contaminate the training set
        print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))

        # TODO(wheung) image size may vary, but not important for now
        return Dataset(training_data, image_size=32), Dataset(validation_data, image_size=32)


class Dataset():
    def __init__(self, data, image_size, images=None):
        self.data = data
        self.images = images
        self.image_size = image_size

    def get_batch_for_cnn(self, batch_size=50):
        """
        :param batch_size:
        :param max_timesteps:
        :return: tuple(batch_x, batch_t)
        where
            batch_x is a NumPy array of shape [batch_size, 32, 32, 3]
            batch_t is a NumPy array of shape [batch_size, 32, 32]
        """
        batch_indices = np.random.choice(self.data.shape[0], batch_size, replace=False)

        batch_x = np.zeros([batch_size, self.image_size, self.image_size, 3])
        batch_t = np.zeros([batch_size, self.image_size, self.image_size])
        if self.images is not None:
            batch_images = self.images[batch_indices]
            for idx, ((vertices, truth), image) in enumerate(zip(self.data[batch_indices], batch_images)):
                x, t = self._create_sample(self.image_size, vertices, truth, image)
                batch_x[idx] = x
                batch_t[idx] = t
        else:
            raise NotImplementedError()

        return batch_x, batch_t

    def _create_sample(self, image_size, poly_verts, ground_truth, image=None):
        total_num_verts = len(poly_verts)

        image = image if image is not None else _create_image(ground_truth)
        start_idx = np.random.randint(total_num_verts + 1)
        num_verts = np.random.randint(total_num_verts)
        poly_verts = np.roll(poly_verts, start_idx, axis=0)

        history_mask = _create_history_mask(poly_verts, num_verts + 1, image_size)
        cursor_mask = _create_point_mask(poly_verts[num_verts], image_size)

        state = np.stack([image, history_mask, cursor_mask], axis=2)
        next_point = np.array(poly_verts[(num_verts + 1) % total_num_verts])

        return state, _create_point_mask(next_point, image_size)

    def get_batch_for_rnn(self, batch_size=50, max_timesteps=5):
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

        batch_d = np.zeros([batch_size], dtype=np.int32)
        batch_x = np.zeros([batch_size, max_timesteps, self.image_size, self.image_size, 3])
        batch_t = np.zeros([batch_size, max_timesteps, self.image_size, self.image_size])
        for idx, (vertices, truth) in enumerate(self.data[batch_indices]):
            d, x, t = self._create_sample_sequence(self.image_size, vertices, truth)
            batch_d[idx] = d
            batch_x[idx, :d, ::] = x
            batch_t[idx, :d, ::] = t
        return batch_d, batch_x, batch_t

    def _create_sample_sequence(self, image_size, poly_verts, ground_truth, image=None):
        total_num_verts = len(poly_verts)

        image = image if image is not None else _create_image(ground_truth)
        start_idx = np.random.randint(total_num_verts + 1)
        poly_verts = np.roll(poly_verts, start_idx, axis=0)

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

    def create_input_for_test(self):
        batch_indices = np.random.choice(self.data.shape[0], batch_size, replace=False)
        image_size = 32

        for idx, (vertices, ground_truth) in enumerate(self.data[batch_indices]):
            total_num_verts = len(poly_verts)

            image = _create_image(ground_truth)
            start_idx = np.random.randint(total_num_verts + 1)
            # Probably should optimize this with a numpy array and clever math. Use np.roll
            poly_verts = poly_verts[start_idx:] + poly_verts[:start_idx]

            history_mask = _create_history_mask(poly_verts, 1, image_size)
            cursor_mask = _create_point_mask(poly_verts[0], image_size)

            state = np.stack([image, history_mask, cursor_mask], axis=2)

            yield state

    def __len__(self):
        return self.data.shape[0]


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
            rr, cc = skimage.draw.line(vertices[i - 1][0], vertices[i - 1][1], vertices[i][0], vertices[i][1])
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


def _create_shape_mask(vertices, image_size):
    mask = np.zeros([image_size, image_size])
    rr, cc = skimage.draw.polygon(*zip(*vertices))
    mask[rr, cc] = 1
    return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
    training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data/')

    # # CNN
    # x, t = training_set.get_batch_for_cnn()
    # batch_size, _, _, _ = x.shape
    # for b in range(batch_size):
    #     fig, ax = plt.subplots(nrows=2, ncols=3)
    #
    #     for c in range(x[b].shape[-1]):
    #         ax[0][c].axis('off')
    #         ax[0][c].imshow(x[b, :, :, c], cmap='gray', interpolation='nearest')
    #
    #     ax[1][0].axis('off')
    #     ax[1][0].imshow(t[b], cmap='gray', interpolation='nearest')
    #     plt.show(block=True)
    #
    # # RNN
    # d, x, t = training_set.get_batch_for_rnn()
    # batch_size, max_timesteps, _, _, _ = x.shape
    # for b in range(batch_size):
    #     fig, ax = plt.subplots(nrows=4, ncols=max_timesteps, figsize=(10, 2 * max(len(x), 2)))
    #     [ax[0][t].set_title(d[b]) for t in range(max_timesteps)]
    #     # [ax[0][t].yaxis.ylabel(str(t)) for t in range(max_timesteps)]
    #
    #     for c in range(x[b].shape[0]):
    #         for i in range(x[b].shape[-1]):
    #             ax[i][c].axis('off')
    #             ax[i][c].imshow(x[b, c, :, :, i], cmap='gray', interpolation='nearest')
    #
    #         i += 1
    #         ax[i][c].axis('off')
    #         ax[i][c].imshow(t[b, c, :, :], cmap='gray', interpolation='nearest')
    #     plt.show(block=True)
