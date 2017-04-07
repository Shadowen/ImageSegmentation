import skimage.draw
import skimage.measure
import numpy as np
from scipy.misc import imread, imresize


def get_train_and_valid_datasets(filename, image_size, input_channels, prediction_size, is_local=True):
    """
    :param filename:
    :param image_size:
    :param prediction_size:
    :param is_local: Optional. True for cluster-style loading.
    :return: tuple(train_data, valid_data)
    where
        train_data is a Dataset consisting of the training set
        valid_data is a Dataset consisting of the validation set
    """
    if not is_local:
        import json
        import os
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
                image = imread(patch_path)
                # Convert to correct format
                image = imresize(image, [image_size, image_size])
                # TODO(wheung) figure out why rounding is necessary for segmentation->poly_vert conversion
                poly_verts = np.floor(np.array(np.roll(np.array(segmentation[0]), 1, axis=1) * prediction_size))
                poly_verts = skimage.measure.approximate_polygon(poly_verts, tolerance=1).astype(
                    np.int8)  # "Perfect" approximation?
                ground_truth = _create_shape_mask(poly_verts, prediction_size)
                # Store it
                data.append((poly_verts, ground_truth))
                images.append(image)
            datasets.append(Dataset(np.array(data),
                                    image_size=image_size,
                                    input_channels=input_channels,
                                    prediction_size=prediction_size,
                                    images=np.array(images)))

        print('{} polygons loaded from {}'.format((sum(map(len, datasets))), filename))
        print('{} for training. {} for validation.'.format(len(datasets[0]), len(datasets[1])))
        os.chdir(original_directory)
        return tuple(datasets)
    else:
        data = np.load(filename)
        # poly_verts, ground_truths = [np.array([np.array(d[idx]) for idx in range(d.shape[0])]) for d in
        #                              [data[:, 0], data[:, 1]]]
        # poly_verts = (poly_verts * (prediction_size / image_size)).tolist()
        # ground_truths = ground_truths.tolist()
        # for idx in range(len(ground_truths)):
        #     ground_truths[idx] = np.array(imresize(ground_truths[idx], [image_size, image_size]), dtype=np.dtype('O'))
        # ground_truths = np.array(ground_truths, dtype=np.dtype('O'))
        # data = np.array([(np.floor(p).astype(np.int8), g) for p, g in zip(poly_verts, ground_truths)])
        # del poly_verts, ground_truths
        print('{} polygons loaded from {}.'.format(data.shape[0], filename))
        valid_size = data.shape[0] // 10
        training_data = data[valid_size:]
        validation_data = data[:valid_size]
        del data  # Make sure we don't contaminate the training set
        print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))

        return Dataset(training_data,
                       image_size=image_size,
                       input_channels=input_channels,
                       prediction_size=prediction_size), \
               Dataset(validation_data,
                       image_size=image_size,
                       input_channels=input_channels,
                       prediction_size=prediction_size)


class Dataset():
    def __init__(self, data, image_size, input_channels, prediction_size, images=None):
        self._data = data
        self._images = images
        self._input_channels = input_channels
        self._image_size = image_size
        self._prediction_size = prediction_size

    def _get_image(self, idx):
        """
        :param idx: The index of the item in the data array
        :return: An NumPy array of shape [self._image_size, self._image_size].
            This is the requested image for the given index. If no images have been loaded, generates an image.
        """
        return self._images[idx] if self._images is not None else np.expand_dims(_create_image(self._data[idx, 1]),
                                                                                 axis=2)

    def get_batch_for_cnn(self, batch_size=50):
        """
        :param batch_size:
        :param max_timesteps:
        :return: tuple(batch_x, batch_t)
        where
            batch_x is a NumPy array of shape [batch_size, 32, 32, 5]
            batch_t is a NumPy array of shape [batch_size, 32, 32]
        """
        batch_indices = np.random.choice(self._data.shape[0], batch_size, replace=False)

        batch_x = np.zeros([batch_size, self._image_size, self._image_size, 3])
        batch_t = np.zeros([batch_size, self._image_size, self._image_size])
        if self._images is not None:
            batch_images = self._images[batch_indices]
            for idx, ((vertices, truth), image) in enumerate(zip(self._data[batch_indices], batch_images)):
                x, t = self._create_sample(self._image_size, vertices, truth, image)
                batch_x[idx] = x
                batch_t[idx] = t
        else:
            for idx, (vertices, truth) in enumerate(self._data[batch_indices]):
                x, t = self._create_sample(self._image_size, vertices, truth)
                batch_x[idx] = x
                batch_t[idx] = t

        return batch_x, batch_t

    def _create_sample(self, image_size, poly_verts, ground_truth, image=None):
        total_num_verts = len(poly_verts)

        image = image if image is not None else np.expand_dims(_create_image(ground_truth),
                                                               axis=2)  # TODO use self._get_image
        start_idx = np.random.randint(total_num_verts + 1)
        num_verts = np.random.randint(total_num_verts)
        poly_verts = np.roll(poly_verts, start_idx, axis=0)

        history_mask = _create_history_mask(poly_verts, num_verts + 1, image_size)
        cursor_mask = _create_point_mask(poly_verts[num_verts], image_size)

        state = np.concatenate([image, np.expand_dims(history_mask, axis=2), np.expand_dims(cursor_mask, axis=2)],
                               axis=2)
        next_point = np.array(poly_verts[(num_verts + 1) % total_num_verts])

        return state, _create_point_mask(next_point, image_size)

    def get_batch_for_rnn(self, batch_size=50, max_timesteps=5):
        """
        :param batch_size:
        :param max_timesteps:
        :return: tuple(batch_d, batch_x, batch_t, poly_verts)
        where
            batch_d is a NumPy array of shape [batch_size]
            batch_x is a NumPy array of shape [batch_size, max_timesteps, 32, 32, 3]
            batch_t is a NumPy array of shape [batch_size, max_timesteps, 32, 32]
            poly_verts is a Python List of length `batch_size` containing the vertices used to generate the polygon
        """
        batch_indices = np.random.choice(self._data.shape[0], batch_size, replace=False)

        batch_d = np.zeros([batch_size], dtype=np.int32)
        batch_x = np.zeros([batch_size, max_timesteps, self._image_size, self._image_size, 3])
        batch_t = np.zeros([batch_size, max_timesteps, self._image_size, self._image_size])
        poly_verts = []
        for idx, (vertices, truth) in enumerate(self._data[batch_indices]):
            d, x, t = self._create_sample_sequence(vertices, image=self._get_image(idx))
            batch_d[idx] = d
            batch_x[idx, :d, ::] = x
            batch_t[idx, :d, ::] = t
            poly_verts.append(vertices)
        return batch_d, batch_x, batch_t, poly_verts

    def get_sample_for_rnn(self):
        """
        :param batch_size:
        :return: tuple(d, x, t, poly_verts)
        where
            d is a NumPy array of shape []
            x is a NumPy array of shape [max_timesteps, image_size, image_size, 3]
            t is a NumPy array of shape [max_timesteps, prediction_size, prediction_size]
            poly_verts is a Python List of length `batch_size` containing the vertices used to generate the polygon
        """
        idx = np.random.choice(self._data.shape[0], 1, replace=False)[0]

        vertices, truth = self._data[idx]
        d, x, t = self._create_sample_sequence(vertices, image=self._get_image(idx))
        return d, x, t, vertices

    def _create_sample_sequence(self, poly_verts, image=None):
        """
        :param image_size:
        :param poly_verts:
        :param ground_truth:
        :param image: Optional.
        :return:
        """
        total_num_verts = len(poly_verts)

        start_idx = np.random.randint(total_num_verts + 1)
        poly_verts = np.roll(poly_verts, start_idx, axis=0)

        inputs = np.empty([total_num_verts, self._image_size, self._image_size, self._input_channels])
        targets = np.empty([total_num_verts, self._prediction_size, self._prediction_size], dtype=np.uint16)
        for idx in range(total_num_verts):
            history_mask = np.expand_dims(_create_history_mask(poly_verts, idx + 1, self._image_size), axis=2)
            cursor_mask = np.expand_dims(_create_point_mask(poly_verts[idx], self._image_size), axis=2)

            state = np.concatenate([image, history_mask, cursor_mask], axis=2)
            next_point = np.array(poly_verts[(idx + 1) % total_num_verts])

            inputs[idx, :, :] = state
            targets[idx, :, :] = _create_point_mask(next_point, self._prediction_size)

        return total_num_verts, inputs, targets

    def raw_sample(self, batch_size):
        """Sample a minibatch from the dataset.
        :return: zip(batch_images, batch_verts, batch_t)
        where
            batch_verts is a Python list of polygon vertices (as NumPy arrays).
            batch_t is the ground truth image as a NumPy array.
        """
        batch_indices = np.random.choice(self._data.shape[0], batch_size, replace=False)
        batch_verts, batch_t = zip(*self._data[batch_indices])
        batch_images = self._images[batch_indices] if self._images is not None else np.expand_dims(batch_t,
                                                                                                   axis=3)  # TODO generate images
        return zip(batch_images, batch_verts, batch_t)

    def __len__(self):
        return self._data.shape[0]

    @property
    def image_size(self):
        return self._image_size

    def show_samples(self, rows=5, cols=5):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(rows, cols)
        plt.suptitle('Shape samples')
        for i in range(rows):
            for e in range(cols):
                ax[i][e].imshow(self._data[i * cols + e][1], cmap='gray', interpolation='nearest')
        plt.show(block=True)


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


def _create_history_mask(vertices, num_points, image_size):
    history_mask = np.zeros([image_size, image_size])
    for i in range(1, num_points):
        rr, cc = skimage.draw.line(vertices[i - 1][0], vertices[i - 1][1], vertices[i][0], vertices[i][1])
        history_mask[rr, cc] = 1
    history_mask[vertices[0][0], vertices[0][1]] = 1
    return history_mask


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
