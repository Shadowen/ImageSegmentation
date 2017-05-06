import os

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

    def train(self, images, durations, histories, targets, additional_feed_args={}):
        step, summaries, loss, acc, t, p, _ = self.sess.run(
            [self.global_step, self._training_summary_ops, self.loss, self.accuracy_op, self.targets_dense,
             self.predictions_dense,
             self.train_op],
            feed_dict={self.image_pl: images, self.duration_pl: durations,
                       self.history_pl: histories, self.targets_pl: targets,
                       **additional_feed_args})
        self.summary_writer.add_summary(summaries, global_step=step)
        print('Step {}: Loss={}\tAcc={}'.format(step, loss, acc))


if __name__ == '__main__':
    image_size = 28
    prediction_size = 28
    max_timesteps = 5
    history_length = 1

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_data, valid_data = get_train_and_valid_datasets('/home/wesley/data/polygons_dataset',
                                                          max_timesteps=max_timesteps,
                                                          image_size=image_size, prediction_size=prediction_size,
                                                          history_length=history_length, is_local=True,
                                                          load_max_images=4, validation_set_percentage=0.5)

    with tf.Session() as sess:
        model_dir = '/home/wesley/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        model = ExperimentModel(sess, max_timesteps, image_size, prediction_size, history_length, model_dir)
        sess.run(tf.global_variables_initializer())
        model.maybe_restore()

        total_steps = 500
        for step_num in range(total_steps):
            batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=2)
            model.train(batch_images, batch_d, batch_h, batch_t)

            if step_num % 100 == 0:
                batch_images, batch_vertices = train_data.raw_sample(batch_size=2)
                model.validate_iou(batch_images, batch_vertices, summary_prefix='train')
                model.save()
