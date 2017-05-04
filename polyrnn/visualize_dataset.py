import matplotlib.lines
import matplotlib.pyplot as plt
from scipy.misc import imresize

from polyrnn.Dataset import get_train_and_valid_datasets, _create_point_mask
from polyrnn.util import *

image_size = 224
prediction_size = 28

max_timesteps = 5
history_length = 2

print('Loading dataset from numpy archive...')
training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data/tiny-polygons',
                                                            max_timesteps=5,
                                                            image_size=image_size,
                                                            prediction_size=prediction_size,
                                                            history_length=history_length,
                                                            is_local=True)
print('Done!')

batch = training_set.get_batch_for_rnn(batch_size=len(training_set))
for i, (d, image, histories, targets, vertices) in enumerate(zip(*batch)):
    print('Shape {} (duration={})...'.format(i, d), end='')
    fig, ax = plt.subplots()
    plt.imshow(imresize(image, prediction_size / image_size, interp='nearest'))
    # Ground truth
    for e, v in enumerate(vertices):
        ax.add_artist(plt.Circle(v, 0.5, color='lightgreen'))
        # plt.text(v[0], v[1], e, color='forestgreen')
    for a, b in iterate_in_ntuples(vertices, n=2):
        ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='forestgreen'))
    # Targets
    z = np.zeros([prediction_size, prediction_size])
    for e in range(d):
        target_mask = _create_point_mask(targets[e], prediction_size)
        plt.imshow(np.stack([z, target_mask, z, target_mask], axis=2))
        plt.text(targets[e][0], targets[e][1], targets[e], color='green')

    plt.show(block=True)
    print('Done!')
