import matplotlib.pyplot as plt

from Dataset import get_train_and_valid_datasets
from util import *

image_size = 28
prediction_size = 28
max_timesteps = 10
history_length = 2

train_data, valid_data = get_train_and_valid_datasets('/home/wesley/data/polygons_dataset_2',
                                                      max_timesteps=max_timesteps,
                                                      image_size=image_size, prediction_size=prediction_size,
                                                      history_length=history_length, is_local=True,
                                                      load_max_images=1)

batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1)

fig, ax = plt.subplots(nrows=11, ncols=3)
ax[0, 0].imshow(batch_images[0])
for t in range(batch_h.shape[1]):
    for e in range(2):
        ax[t + 1, e].imshow(batch_h[0, t, :, :, e])
    ax[t + 1, 2].imshow(create_point_mask(batch_t[0, t], prediction_size))

plt.show(block=True)
