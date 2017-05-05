import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from polyrnn import Model
from polyrnn.Dataset import get_train_and_valid_datasets
from polyrnn.util import lazyproperty


class ExperimentModel(Model.Model):
    def _build_graph(self):
        image_flat = tf.reshape(self.image_pl, shape=[-1, self.image_size * self.image_size * 3])
        tiled_image = tf.tile(tf.expand_dims(image_flat / 255, axis=1),
                              multiples=[1, self.max_timesteps, 1])

        history_flat = tf.reshape(self.history_pl, shape=[-1, self.max_timesteps, 28 * 28])
        concat = tf.concat([tiled_image, history_flat], axis=2)

        with tf.variable_scope('rnn'):
            rnn_cell = tf.contrib.rnn.BasicRNNCell(prediction_size ** 2)
            self._rnn_zero_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self._rnn_initial_state_pl = tf.placeholder_with_default(self.rnn_zero_state,
                                                                     shape=[None, rnn_cell.state_size])
            rnn_output, self._rnn_final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=concat,
                                                                  sequence_length=self._duration_pl,
                                                                  initial_state=self._rnn_initial_state_pl)
        self._prediction_logits = tf.layers.dense(inputs=rnn_output, units=prediction_size ** 2, name='prediction')
        self.predictions_dense = tf.cast(tf.argmax(self._prediction_logits, axis=2), dtype=tf.int32)

        masked_correct = tf.logical_and(tf.equal(self.targets_dense, self.predictions_dense),
                                        tf.sequence_mask(self._duration_pl, max_timesteps))
        self.accuracy_op = tf.count_nonzero(masked_correct, dtype=tf.int32) / tf.reduce_sum(
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
        step, summaries, loss, acc, t, p, _ = self.sess.run(
            [self.global_step, self._training_summary_ops, self.loss, self.accuracy_op, self.targets_dense,
             self.predictions_dense,
             self.train_op],
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
        # model.maybe_restore()

        total_steps = 1200
        for step_num in range(total_steps):
            batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1,
                                                                                                   start_idx=None)
            model.train(batch_images, batch_d, batch_h, batch_t)
        # model.save()

        batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1,
                                                                                               start_idx=None)
        print(batch_vertices)
        p = model.predict_logits_one_step(batch_images, batch_h)
        plt.figure()
        plt.title('Image')
        plt.imshow(batch_images[0])
        plt.figure()
        plt.title('History')
        plt.imshow(batch_h[0, 0, :, :, 0])
        plt.figure()
        plt.title('Truth')
        t = np.zeros([28, 28])
        t[batch_t[0, 0, 1], batch_t[0, 0, 0]] = 1
        plt.imshow(t)
        plt.figure()
        plt.title('Prediction')
        plt.imshow(np.rollaxis(np.reshape(p[0][0], [28, 28]), axis=1), cmap='Greys')  # May want to softmax this...
        plt.show(block=True)
