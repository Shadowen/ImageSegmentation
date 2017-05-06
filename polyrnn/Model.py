import os
from abc import abstractmethod, abstractproperty

import numpy as np
import tensorflow as tf

from polyrnn.util import lazyproperty, create_shape_mask


class Model():
    def __init__(self, sess, max_timesteps, image_size, prediction_size, history_length, filepath):
        self.sess = sess
        self.max_timesteps = max_timesteps
        self.image_size = image_size
        self.prediction_size = prediction_size
        self.history_length = history_length
        self._filepath = filepath

        self.global_step = tf.train.get_global_step()

        self.scope_name = 'Estimator'
        with tf.variable_scope(self.scope_name):
            self.image_pl = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3],
                                           name='image')  # [batch, x, y, c]
            self.batch_size = tf.shape(self.image_pl)[0]
            self._duration_pl = tf.placeholder(tf.int32, shape=[None], name='duration')  # [batch]
            self.history_pl = tf.placeholder(tf.float32, shape=[None, max_timesteps, prediction_size, prediction_size,
                                                                history_length], name='history')

            with tf.variable_scope('target'):
                self.targets_pl = tf.placeholder(tf.int32, shape=[None, max_timesteps, 2], name='target')
                targets_x, targets_y = [tf.squeeze(c, axis=2) for c in tf.split(self.targets_pl, 2, axis=2)]
                self.targets_dense = targets_x * prediction_size + targets_y

            self._build_graph()
            num_samples = tf.reduce_sum(self._duration_pl)
            self.sequence_mask = tf.sequence_mask(self._duration_pl, max_timesteps)
            self._prediction_argmax = tf.argmax(self.prediction_logits, axis=2)
            self.predictions_dense = tf.cast(self._prediction_argmax, dtype=tf.int32)
            # Accuracy
            masked_correct = tf.logical_and(tf.equal(self.targets_dense, self.predictions_dense), self.sequence_mask)
            self._accuracy_op = tf.count_nonzero(masked_correct, dtype=tf.int32) / num_samples
            # L2 Error
            individual_errors = tf.norm(
                tf.cast(tf.cast(self.targets_pl, dtype=tf.int64) - self.prediction_max, dtype=tf.float64),
                axis=2) * tf.cast(self.sequence_mask, dtype=tf.float64)
            self._avg_error_op = tf.reduce_sum(individual_errors) / tf.cast(num_samples, dtype=tf.float64)
            self._max_error_op = tf.reduce_max(individual_errors)
            # Make sure everything exists
            for p in [self.rnn_initial_state_pl, self.rnn_zero_state, self.duration_pl, self.rnn_final_state,
                      self.prediction_logits, self.trainable_variables, self.loss, self.train_op, self.summary_writer]:
                if p is None:
                    raise NotImplementedError()

        self._training_summary_ops = tf.summary.merge(self._create_summaries('train'))
        self._validation_summary_ops = tf.summary.merge(self._create_summaries('valid'))

        self._iou_pl = tf.placeholder(tf.float32, shape=[], name='iou')
        self._iou_summary_op = tf.summary.scalar('iou', self._iou_pl)

    @abstractmethod
    def _build_graph(self):
        pass

    @abstractproperty
    def rnn_initial_state_pl(self):
        pass

    @abstractproperty
    def rnn_zero_state(self):
        pass

    @property
    def duration_pl(self):
        return self._duration_pl

    @abstractproperty
    def rnn_final_state(self):
        pass

    @abstractproperty
    def prediction_logits(self):
        pass

    @lazyproperty
    def prediction_max(self):
        return tf.stack(
            [self._prediction_argmax // self.prediction_size, self._prediction_argmax % self.prediction_size], axis=2)

    @lazyproperty
    def loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets_dense,
                                                                             logits=self.prediction_logits))

    @property
    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

    @lazyproperty
    def train_op(self):
        grads = tf.gradients(self.loss, self.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        return optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables),
                                         global_step=self.global_step)

    @lazyproperty
    def summary_writer(self):
        return tf.summary.FileWriter(logdir=self._filepath, graph=self.sess.graph)

    def _create_summaries(self, prefix):
        """
        :param prefix: Either 'train' or 'valid'
        To add summaries in a subclass, override this and call super, appending your custom summaries to the list returned
        :return: list [summaries]
        """

        loss_summary_op = tf.summary.scalar(prefix + '/loss', self.loss)
        accuracy_summary_op = tf.summary.scalar(prefix + '/accuracy', self._accuracy_op)
        avg_error_summary_op = tf.summary.scalar(prefix + '/avg_error', self._avg_error_op)
        max_error_summary_op = tf.summary.scalar(prefix + '/max_error', self._max_error_op)

        return [loss_summary_op, accuracy_summary_op, avg_error_summary_op, max_error_summary_op]

    @lazyproperty
    def saver(self):
        saver = tf.train.Saver(var_list=self.trainable_variables + [self.global_step], max_to_keep=2,
                               keep_checkpoint_every_n_hours=12)
        return saver

    def maybe_restore(self):
        if not os.path.exists(self._filepath):
            os.makedirs(self._filepath)
        if os.path.exists(self._filepath + 'checkpoint'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self._filepath))
            global_step = self.sess.run(self.global_step)
            print('Restored model from {} at step {}'.format(self._filepath, global_step))

    def save(self):
        self.saver.save(self.sess, self._filepath + 'model', global_step=self.global_step)

    def train(self, images, durations, histories, targets, additional_feed_args={}):
        step, summaries, _ = self.sess.run([self.global_step, self._training_summary_ops, self.train_op],
                                           feed_dict={self.image_pl: images, self.duration_pl: durations,
                                                      self.history_pl: histories, self.targets_pl: targets,
                                                      **additional_feed_args})
        self.summary_writer.add_summary(summaries, global_step=step)

    def validate(self, images, durations, histories, targets, additional_feed_args={}):
        step, summaries = self.sess.run([self.global_step, self._validation_summary_ops],
                                        feed_dict={self.image_pl: images, self.duration_pl: durations,
                                                   self.history_pl: histories, self.targets_pl: targets,
                                                   **additional_feed_args})
        self.summary_writer.add_summary(summaries, global_step=step)

    def predict_one_step(self, images, histories, rnn_state, additional_feed_args={}):
        return self.sess.run([self.prediction_max, self.rnn_final_state],
                             feed_dict={self.image_pl: images, self.duration_pl: np.ones([images.shape[0]]),
                                        self.history_pl: histories, self.rnn_initial_state_pl: rnn_state,
                                        **additional_feed_args})

    def validate_iou(self, images, true_vertices, summary_prefix='valid', additional_feed_args={}):
        batch_size = images.shape[0]
        predicted_vertices = np.empty([batch_size, self.max_timesteps, 2])

        histories = np.zeros([batch_size, self.max_timesteps, 28, 28, self.history_length])
        step, rnn_state = self.sess.run([self.global_step, self.rnn_zero_state], feed_dict={self.image_pl: images})
        for i in range(self.max_timesteps):
            out, rnn_state = self.predict_one_step(images, histories, rnn_state, **additional_feed_args)
            predicted_vertices[:, i, :] = out[:, 0, :]

            predicted_mask = np.zeros([batch_size, self.prediction_size, self.prediction_size])
            predicted_mask[:, out[:, 0, 1], out[:, 0, 0]] = 1

            histories = np.roll(histories, axis=1, shift=1)
            histories[:, 0, :, :, 0] = predicted_mask

        ious = []
        for i in range(predicted_vertices.shape[0]):
            predicted_mask = create_shape_mask(predicted_vertices[i], self.prediction_size)
            true_mask = create_shape_mask(true_vertices[i], self.prediction_size)
            intersection = np.count_nonzero(np.logical_and(predicted_mask, true_mask))
            union = np.count_nonzero(np.logical_or(predicted_mask, true_mask))
            ious.append(intersection / union)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=summary_prefix + '/iou' if summary_prefix else'iou', simple_value=np.mean(ious))])
        self.summary_writer.add_summary(summary, global_step=step)
