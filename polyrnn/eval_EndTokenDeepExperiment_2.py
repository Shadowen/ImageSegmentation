import numpy as np
import tensorflow as tf

from polyrnn.EndTokenDataset import get_train_and_valid_datasets
from polyrnn.EndTokenDeepExperiment_2 import ExperimentModel

if __name__ == '__main__':
    image_size = 28
    prediction_size = 28
    max_timesteps = 10
    history_length = 2

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.Session() as sess:
        model_dir = '/data/EndTokenDeepExperiment_2/'

        train_data, valid_data = get_train_and_valid_datasets('/data/polygons_dataset_3',
                                                              max_timesteps=max_timesteps,
                                                              image_size=image_size,
                                                              prediction_size=prediction_size,
                                                              history_length=history_length, is_local=True,
                                                              load_max_images=100000, validation_set_percentage=0.1)

        model = ExperimentModel(sess, max_timesteps, image_size, prediction_size, history_length, model_dir)
        sess.run(tf.global_variables_initializer())
        model.maybe_restore()

        inc = tf.assign_add(global_step, 1, name='increment_global_step')
        sess.run(inc)
        sess.run(inc)

        np.random.seed(1)
        batch_images, batch_vertices = train_data.raw_sample(batch_size=100)
        model.validate_iou(batch_images, batch_vertices, summary_prefix='train')

        batch_images, batch_vertices = valid_data.raw_sample(batch_size=100)
        model.validate_iou(batch_images, batch_vertices, summary_prefix='valid')
