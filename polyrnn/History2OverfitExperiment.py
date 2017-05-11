import os
import shutil

import tensorflow as tf

from polyrnn import Model
from polyrnn.Dataset import get_train_and_valid_datasets
from polyrnn.convLSTM import ConvLSTMCell
from polyrnn.util import lazyproperty


class ExperimentModel(Model.Model):
    def _build_graph(self):
        # conv1 [batch, x, y, c]
        conv1 = tf.layers.conv2d(inputs=self.image_pl / 255, filters=16, kernel_size=[5, 5], padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv2 [batch, x, y, c]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                 name='conv2')
        # conv3 [batch, x, y, c]
        conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 name='conv3')
        # tiled [batch, timestep, c]
        tiled = tf.tile(tf.expand_dims(conv3, axis=1), multiples=[1, self.max_timesteps, 1, 1, 1])
        concat = tf.concat([tiled, self.history_pl], axis=4)

        with tf.variable_scope('rnn'):
            # Make placeholder time major for RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_input = tf.transpose(concat, (1, 0, 2, 3, 4))

            rnn_cell = lambda: ConvLSTMCell(shape=[28, 28], filters=16, kernel=[3, 3])
            rnn_layers = 2
            multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(rnn_layers)])
            self._rnn_zero_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self._rnn_initial_state_pl = tuple(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(self.rnn_zero_state[i].c,
                                            shape=[None] + multi_rnn_cell.state_size[i].c.as_list()),
                tf.placeholder_with_default(self.rnn_zero_state[i].h,
                                            shape=[None] + multi_rnn_cell.state_size[i].h.as_list())) for i in
                                               range(rnn_layers))
            rnn_output, self._rnn_final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=rnn_input,
                                                                  sequence_length=self._duration_pl,
                                                                  initial_state=self._rnn_initial_state_pl,
                                                                  time_major=True)
            # Make placeholder batch major again after RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2, 3, 4))

        self._prediction_logits = tf.reshape(tf.layers.dense(inputs=tf.reshape(rnn_output, shape=[-1, 28 * 28 * 16]),
                                                             units=prediction_size ** 2, name='prediction'),
                                             shape=[-1, self.max_timesteps, self.prediction_size ** 2])

    @lazyproperty
    def rnn_initial_state_pl(self):
        return self._rnn_initial_state_pl

    @lazyproperty
    def rnn_zero_state(self):
        return self._rnn_zero_state

    @lazyproperty
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazyproperty
    def prediction_logits(self):
        return self._prediction_logits


if __name__ == '__main__':
    image_size = 28
    prediction_size = 28
    max_timesteps = 10
    history_length = 2

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_data, valid_data = get_train_and_valid_datasets('/data/polygons_dataset_2',
                                                          max_timesteps=max_timesteps,
                                                          image_size=image_size, prediction_size=prediction_size,
                                                          history_length=history_length, is_local=True,
                                                          load_max_images=1)

    with tf.Session() as sess:
        model_dir = '/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        model = ExperimentModel(sess, max_timesteps, image_size, prediction_size, history_length, model_dir)
        sess.run(tf.global_variables_initializer())
        # model.maybe_restore()

        total_steps = 2000
        for step_num in range(total_steps):
            batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1)
            model.train(batch_images, batch_d, batch_h, batch_t)

            if step_num > 1000 and step_num % 10 == 0:
                print('Step {}'.format(step_num))
                batch_images, batch_vertices = train_data.raw_sample(batch_size=1)
                model.validate_iou(batch_images, batch_vertices, summary_prefix='train')
                model.save()
