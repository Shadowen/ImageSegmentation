import numpy as np

from polyrnn.Dataset import get_train_and_valid_datasets, create_shape_mask

image_size = 28
prediction_size = 28
max_timesteps = 5
history_length = 1

print('Loading dataset from numpy archive...')
training_set, validation_set = get_train_and_valid_datasets('/data/polygons_dataset',
                                                            max_timesteps=5,
                                                            image_size=image_size,
                                                            prediction_size=prediction_size,
                                                            history_length=history_length,
                                                            is_local=True)
print('Done!')

print('Training set')
batch_images = training_set._images
batch_verts = training_set._vertices
for idx, (image, vertices) in enumerate(zip(batch_images, batch_verts)):
    true_mask = create_shape_mask(vertices, prediction_size)
    if np.count_nonzero(true_mask) == 0:
        print(idx)

print('Validation set')
batch_images = validation_set._images
batch_verts = validation_set._vertices
for idx, (image, vertices) in enumerate(zip(batch_images, batch_verts)):
    true_mask = create_shape_mask(vertices, prediction_size)
    if np.count_nonzero(true_mask) == 0:
        print(idx)
