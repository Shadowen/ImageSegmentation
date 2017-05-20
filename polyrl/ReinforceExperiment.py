import os

import gym
import tensorflow as tf

from polyrl import PolicyEstimator, MonteCarloPolicyGradient
from util import *


class ExperimentPolicyEstimator(PolicyEstimator.PolicyEstimator):
    def _build_graph(self):
        fc_1 = tf.layers.dense(inputs=self.state_pl, units=8, activation=tf.nn.relu)
        return tf.layers.dense(inputs=fc_1, units=self.action_size, activation=None)

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


env = gym.make('Acrobot-v1')

logdir = '/home/wesley/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
reset_dir(logdir)

with tf.Session() as sess:
    tf.Variable(0, name="global_step", trainable=False)
    summary_writer = tf.summary.FileWriter(logdir)
    policy_estimator = ExperimentPolicyEstimator(state_size=env.observation_space.shape, action_size=env.action_space.n,
                                                 tf_session=sess, summary_writer=summary_writer)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    MonteCarloPolicyGradient.reinforce(env=env, sess=sess, policy_estimator=policy_estimator, total_episodes=None,
                                       max_timesteps_per_episode=500, summary_writer=summary_writer)
