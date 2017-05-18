from collections import defaultdict

import numpy as np

from Dataset import get_train_and_valid_datasets

image_size = 28
prediction_size = 28
max_timesteps = 10
history_length = 2

print('Loading dataset from numpy archive...')
training_set, validation_set = get_train_and_valid_datasets('/home/wesley/docker_data/polygons_dataset_3',
                                                            max_timesteps=5,
                                                            image_size=image_size,
                                                            prediction_size=prediction_size,
                                                            history_length=history_length,
                                                            is_local=True,
                                                            validation_set_percentage=0)
print('Done!')

print('Training set')
batch_images = training_set._images
batch_verts = training_set._vertices
vertex_counts = defaultdict(lambda: 0)
max = 0
min = 1000

for idx, (image, vertices) in enumerate(zip(batch_images, batch_verts)):
    l = len(vertices)
    vertex_counts[l] += 1

print(vertex_counts)
print('Max = {}'.format(np.max([int(x) for x in vertex_counts.keys()])))
print('Min = {}'.format(np.min([int(x) for x in vertex_counts.keys()])))

# polygons_dataset_3
"""
3: 1390
4: 30798
5: 49247
6: 16912
7: 1610
8: 43
"""
