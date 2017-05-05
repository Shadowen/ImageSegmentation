import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from polyrnn import Model
from polyrnn.Dataset import get_train_and_valid_datasets
from polyrnn.convLSTM import ConvLSTMCell
from polyrnn.util import lazyproperty


class ExperimentModel(Model.Model):
    def _build_graph(self):
        # conv1 [batch, x, y, c]
        conv1 = tf.layers.conv2d(inputs=self.image_pl / 255, filters=32, kernel_size=[1, 1], padding='same',
                                 activation=tf.nn.relu)
        # tiled_conv1 [batch, timestep, c]
        tiled_conv1 = tf.tile(tf.expand_dims(conv1, axis=1), multiples=[1, self.max_timesteps, 1, 1, 1])

        concat = tf.concat([tiled_conv1, self.history_pl], axis=4)

        with tf.variable_scope('rnn'):
            # Make placeholder time major for RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_input = tf.transpose(concat, (1, 0, 2, 3, 4))

            rnn_cell = lambda: ConvLSTMCell(shape=[28, 28], filters=16, kernel=[3, 3])
            rnn_layers = 1
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
        self.predictions_dense = tf.cast(tf.argmax(self._prediction_logits, axis=2), dtype=tf.int32)

        self.accuracy_op = tf.count_nonzero(
            tf.logical_and(tf.equal(self.targets_dense, self.predictions_dense),
                           tf.sequence_mask(self._duration_pl, max_timesteps)), dtype=tf.int32) / tf.reduce_sum(
            self._duration_pl)

    @lazyproperty
    def rnn_initial_state_pl(self):
        return self._rnn_initial_state_pl

    @lazyproperty
    def rnn_zero_state(self):
        return self._rnn_zero_state

    @lazyproperty
    def duration_pl(self):
        return self._duration_pl

    @lazyproperty
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazyproperty
    def prediction_logits(self):
        return self._prediction_logits

    @lazyproperty
    def prediction_max(self):
        return self._prediction_max

    def _create_summaries(self):
        training_summaries, validation_summaries = super(ExperimentModel, self)._create_summaries()

        accuracy_summary_op = tf.summary.scalar('Accuracy', self.accuracy_op)
        training_summaries.append(accuracy_summary_op)

        validation_accuracy_summary_op = tf.summary.scalar('Validation_Accuracy', self.accuracy_op)
        validation_summaries.append(validation_accuracy_summary_op)
        return training_summaries, validation_summaries

    def train(self, images, durations, histories, targets, additional_feed_args={}):
        step, summaries, loss, acc, _ = self.sess.run(
            [self.global_step, self._training_summary_ops, self.loss, self.accuracy_op, self.train_op],
            feed_dict={self.image_pl: images, self.duration_pl: durations,
                       self.history_pl: histories, self.targets_pl: targets,
                       **additional_feed_args})
        self._summary_writer.add_summary(summaries, global_step=step)
        print('Step {}: Loss={}\tAcc={}'.format(step, loss, acc))


if __name__ == '__main__':
    image_size = 28
    prediction_size = 28
    max_timesteps = 5
    history_length = 1

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_data, valid_data = get_train_and_valid_datasets('/home/wesley/data/tiny-polygons',
                                                          max_timesteps=max_timesteps,
                                                          image_size=image_size, prediction_size=prediction_size,
                                                          history_length=history_length, is_local=True,
                                                          load_max_images=1, validation_set_percentage=0)

    with tf.Session() as sess:
        model_dir = '/home/wesley/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        model = ExperimentModel(sess, max_timesteps, image_size, prediction_size, history_length, model_dir)
        sess.run(tf.global_variables_initializer())
        model.maybe_restore()

        total_steps = 1000
        for step_num in range(total_steps):
            batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1)
            model.train(batch_images, batch_d, batch_h, batch_t)

            # if step_num % 100 == 0 and len(valid_data) > 0:
            #     batch_d, batch_images, batch_h, batch_t, batch_vertices = valid_data.get_batch_for_rnn(batch_size=1)
            #     model.validate(batch_images, batch_d, batch_h, batch_t)
        model.save()