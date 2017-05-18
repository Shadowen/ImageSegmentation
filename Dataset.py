import json
import os

from scipy.misc import imread, imresize

from util import *


def get_train_and_valid_datasets(filename, max_timesteps, image_size, prediction_size, history_length, is_local=True,
                                 load_max_images=None, validation_set_percentage=0.1):
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
                image = imresize(image, [image_size, image_size], interp='nearest')
                poly_verts = np.floor(np.array(np.roll(np.array(segmentation[0]), 1, axis=1) * prediction_size)).astype(
                    np.int8)
                ground_truth = create_shape_mask(poly_verts, prediction_size)
                # Store it
                data.append((poly_verts, ground_truth))
                images.append(image)
            datasets.append(Dataset(np.array(data),
                                    max_timesteps=max_timesteps,
                                    image_size=image_size,
                                    prediction_size=prediction_size,
                                    images=np.array(images),
                                    history_length=history_length))

        print('{} polygons loaded from {}'.format((sum(map(len, datasets))), filename))
        print('{} for training. {} for validation.'.format(len(datasets[0]), len(datasets[1])))
        os.chdir(original_directory)
        return tuple(datasets)
    else:
        print('Loading data from numpy archives...')
        numbers = [s[:-4] for s in os.listdir('{}'.format(filename)) if s.endswith('.npy')]
        total_num_images = len(numbers) if load_max_images is None else min(len(numbers), load_max_images)
        all_images = np.empty([total_num_images, image_size, image_size, 3], dtype=np.float64)
        all_vertices = np.empty([total_num_images], dtype=np.object)
        for idx, number in enumerate(numbers):
            if idx >= total_num_images:
                break
            print('\r{} of {} images.'.format(idx + 1, total_num_images), end='')
            # Read the file
            vertices = np.load('{}/{}.npy'.format(filename, number))
            image = imread('{}/{}.jpg'.format(filename, number)) / 255
            # Convert to correct format
            if image.shape[0] != image_size:
                image = imresize(image, [image_size, image_size])
            all_images[idx, ::] = image
            all_vertices[idx] = np.floor(vertices * prediction_size).astype(int)

        validation_set_size = np.floor(total_num_images * validation_set_percentage).astype(int)
        train_images = all_images[validation_set_size:]
        valid_images = all_images[:validation_set_size]
        train_vertices = all_vertices[validation_set_size:]
        valid_vertices = all_vertices[:validation_set_size]

        print('\n{} polygons loaded from {}'.format(len(train_images) + len(valid_images), filename))
        print('{} for training. {} for validation.'.format(len(train_images), len(valid_images)))
        return Dataset(images=train_images,
                       vertices=train_vertices,
                       max_timesteps=max_timesteps,
                       prediction_size=prediction_size,
                       history_length=history_length), \
               Dataset(images=valid_images,
                       vertices=valid_vertices,
                       max_timesteps=max_timesteps,
                       prediction_size=prediction_size,
                       history_length=history_length)


class Dataset():
    def __init__(self, images, vertices, max_timesteps, prediction_size, history_length):
        self._images = images
        self._vertices = vertices
        self._max_timesteps = max_timesteps
        self._image_size = self._images.shape[1]
        self._prediction_size = prediction_size
        self._history_length = history_length

    def get_batch_for_rnn(self, batch_size=8, start_idx=None):
        """
        :param batch_size:
        :param max_timesteps:
        :return: tuple(batch_d, batch_images, batch_t, batch_vertices)
        where
            batch_d is a NumPy array of shape [batch_size]
            batch_images is a NumPy array of shape [batch_size, self._max_timesteps, self._image_size, self._image_size, 3]
            batch_h is a NumPy array of shape [batch_size, self._max_timesteps, self._prediction_size, self._prediction_size, self._history_length]
            batch_t is a NumPy array of shape [batch_size, self._max_timesteps, 2]
            batch_vertices is a NumPy array of shape [batch_size, ..., 2] containing the vertices used to generate the polygon
        """
        batch_indices = np.random.choice(self._images.shape[0], batch_size, replace=False)

        batch_d = np.zeros([batch_size], dtype=np.int32)
        batch_images = self._images[batch_indices]
        batch_h = np.zeros(
            [batch_size, self._max_timesteps, self._prediction_size, self._prediction_size, self._history_length])
        batch_t = np.zeros([batch_size, self._max_timesteps, 2], dtype=np.int32)
        batch_vertices = self._vertices[batch_indices]
        for i, vertices in enumerate(batch_vertices):
            duration, history, truth, poly_verts = self._create_sample_sequence(vertices, start_idx)
            batch_d[i] = duration
            batch_h[i, ::] = history
            batch_t[i, ::] = truth
            batch_vertices[i] = poly_verts
        return batch_d, batch_images, batch_h, batch_t, batch_vertices

    def _create_sample_sequence(self, poly_verts, start_idx=None):
        """
        :param poly_verts:
        :return:
        """
        total_num_verts = len(poly_verts)

        start_idx = np.random.randint(total_num_verts) if start_idx is None else start_idx
        poly_verts = np.roll(poly_verts, start_idx, axis=0)

        histories = np.zeros([self._max_timesteps, self._prediction_size, self._prediction_size, self._history_length],
                             dtype=np.uint16)
        targets = np.zeros([self._max_timesteps, 2], dtype=np.int32)
        for idx in range(min(total_num_verts - self._history_length + 1, self._max_timesteps)):
            histories[idx, :, :, :] = create_history(poly_verts, idx, self._history_length, self._prediction_size)
            next_point = np.array(poly_verts[(idx + 1) % total_num_verts])
            targets[idx, :] = next_point

        return total_num_verts - self._history_length + 1, histories, targets, poly_verts

    def raw_sample(self, batch_size):
        """Sample a minibatch of images and vertices only from the dataset.
        :return: (batch_images, batch_verts)
        where
            batch_images is a NumPy array.
            batch_verts is a Python list of polygon vertices (as NumPy arrays).
        """
        batch_indices = np.random.choice(self._images.shape[0], batch_size, replace=False)
        batch_images = self._images[batch_indices]
        batch_verts = np.array([np.roll(p, np.random.randint(len(p)), axis=0) for p in self._vertices[batch_indices]])
        return batch_images, batch_verts

    def __len__(self):
        assert self._images.shape[0] == self._vertices.shape[0]
        return self._images.shape[0]

    @property
    def image_size(self):
        return self._image_size


if __name__ == '__main__':
    training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data/polygons_dataset_2',
                                                                max_timesteps=5,
                                                                image_size=224,
                                                                prediction_size=28,
                                                                history_length=2,
                                                                is_local=True)
    training_set.get_batch_for_rnn(batch_size=1)
