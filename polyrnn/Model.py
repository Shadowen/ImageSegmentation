import os
from abc import abstractmethod, abstractproperty

import numpy as np
import tensorflow as tf

from polyrnn.util import lazyproperty


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
            # Make sure everything exists
            for p in [self.rnn_initial_state_pl, self.rnn_zero_state, self.duration_pl, self.rnn_final_state,
                      self.prediction_logits, self.trainable_variables, self.loss, self.train_op]:
                if p is None:
                    raise NotImplementedError()

            training_summaries, validation_summaries = self._create_summaries()
            self._training_summary_ops = tf.summary.merge(training_summaries)
            self._validation_summary_ops = tf.summary.merge(validation_summaries)

    @abstractmethod
    def _build_graph(self):
        pass

    @abstractproperty
    def rnn_initial_state_pl(self):
        pass

    @abstractproperty
    def rnn_zero_state(self):
        pass

    @abstractproperty
    def duration_pl(self):
        pass

    @abstractproperty
    def rnn_final_state(self):
        pass

    @abstractproperty
    def prediction_logits(self):
        pass

    @abstractproperty
    def prediction_max(self):
        pass

    @lazyproperty
    def loss(self):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets_dense, logits=self.prediction_logits))

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

    def _create_summaries(self):
        """
        :return: [training summaries], [validation summaries]
        """
        self._summary_writer = tf.summary.FileWriter(logdir=self._filepath, graph=self.sess.graph)

        self._loss_summary_op = tf.summary.scalar('Loss', self.loss)

        self._validation_loss_summary_op = tf.summary.scalar('Validation_Loss', self.loss)

        return [self._loss_summary_op], [self._validation_loss_summary_op]

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
        self._summary_writer.add_summary(summaries, global_step=step)

    def validate(self, images, durations, histories, targets, additional_feed_args={}):
        step, summaries = self.sess.run([self.global_step, self._validation_summary_ops],
                                        feed_dict={self.image_pl: images, self.duration_pl: durations,
                                                   self.history_pl: histories, self.targets_pl: targets,
                                                   **additional_feed_args})
        self._summary_writer.add_summary(summaries, global_step=step)

    def predict_logits_one_step(self, images, histories, additional_feed_args={}):
        return self.sess.run(self.prediction_logits,
                             feed_dict={self.image_pl: images, self.duration_pl: np.ones([images.shape[0]]),
                                        self.history_pl: histories, **additional_feed_args})

    def predict_one_step(self, images, durations, histories, additional_feed_args={}):
        self.sess.run(self.prediction_max,
                      feed_dict={self.image_pl: images, self.duration_pl: durations,
                                 self.history_pl: histories, **additional_feed_args})
