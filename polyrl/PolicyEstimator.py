from abc import abstractmethod

import tensorflow as tf

from util import *


class PolicyEstimator():
    def __init__(self, state_size, action_size, tf_session, summary_writer):
        self.state_size = state_size
        self.action_size = action_size
        self.tf_session = tf_session
        self.summary_writer = summary_writer

        self.global_step = tf.train.get_global_step()
        assert self.global_step is not None

        with tf.variable_scope('policy_estimator'):
            # Build inference graph
            self.state_pl = tf.placeholder(shape=[None] + state_size, dtype=tf.float32, name='state')
            self.action_logits = self._build_graph(self.state_pl)
            self.softmax = tf.nn.softmax(self.action_logits)
            # Loss and training
            self.td_target_pl = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_taken_pl = tf.placeholder(shape=[None], dtype=tf.int32)
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_taken_pl,
                                                                                         logits=self.action_logits) * self.td_target_pl)
            # Ensure everything else is initialized
            for p in [self.train_op]:
                if p is None:
                    raise NotImplementedError()
            assert self.action_logits.get_shape().as_list() == [None] + self.action_size
            # Summaries
            self._create_summaries()

    @abstractmethod
    def _build_graph(self, state_pl):
        """

        :param state_pl:
        :return: action_logits
        """
        pass

    @lazyproperty
    def train_op(self):
        """
        Define the train_op here. The gradients, clipping, and optimizer should be defined here as well.
        Because this is a @lazyproperty, this will only be called once to initialize. The result will automatically be
        cached in the property.
        :return:
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.gradients_ops, self.trainable_variables = zip(*self.optimizer.compute_gradients(self.loss_op))
        self.clipped_gradients_ops, self.gradient_global_norms_op = tf.clip_by_global_norm(self.gradients_ops,
                                                                                           clip_norm=1.0)
        self._train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients_ops, self.trainable_variables),
                                                        global_step=tf.contrib.framework.get_global_step())

        self.gradient_holders = []
        for idx, var in enumerate(self.trainable_variables):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)
        self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.trainable_variables),
                                                           global_step=tf.contrib.framework.get_global_step())

        self.gradBuffer = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]
        return self._train_op

    def _create_summaries(self):
        self.action_probs_summary = tf.summary.histogram('actions_probs', self.softmax)

        self.max_gradient_norm_op = tf.reduce_max([tf.reduce_max(g) for g in self.gradients_ops])
        self.max_gradient_norm_summary_op = tf.summary.scalar('max_gradient_norm', self.max_gradient_norm_op)
        self.global_norm_summary_op = tf.summary.scalar('gradient_global_norm', self.gradient_global_norms_op)
        self.loss_summary_op = tf.summary.scalar('loss', self.loss_op)

        self.training_summaries_op = tf.summary.merge(
            [self.max_gradient_norm_summary_op, self.global_norm_summary_op, self.loss_summary_op])

    def predict(self, state):
        """Picks an action given a state input."""
        a_dist = self.tf_session.run(self.softmax, feed_dict={self.state_pl: [state]})[0]
        action = np.random.choice(np.arange(len(a_dist)), p=a_dist)
        return action

    def update(self, state, action, td_target, apply_grads):
        """
        Updates the policy estimator according to the policy gradient given state, action, and td-target tuples.
        Only actually applies the gradient back to the estimator if apply_grads=True.
        """
        # Accumulate gradients
        grads = self.tf_session.run(self.gradients_ops, feed_dict={self.td_target_pl: td_target, self.action_taken_pl: action,
                                                                   self.state_pl: state})
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        if apply_grads:
            global_step, _ = self.tf_session.run(
                [self.global_step, self.update_batch],
                feed_dict=dict(zip(self.gradient_holders, self.gradBuffer)))
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0

                # self.summary_writer.add_summary(summaries, global_step)
