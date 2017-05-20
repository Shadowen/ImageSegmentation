# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Originally a Policy Gradient algorithm, upgraded to an Actor-Critic algorithm.

import itertools
import operator
import os
import shutil
from functools import reduce

import gym
import numpy as np
import tensorflow as tf


class PolicyEstimator():
    def __init__(self, learning_rate, state_size, action_size, tf_session, scope='policy_estimator'):
        self.tf_session = tf_session

        with tf.variable_scope(scope):
            self.global_step = tf.train.get_global_step()
            assert self.global_step is not None

            self.state_pl = tf.placeholder(shape=[None, reduce(operator.mul, state_size)], dtype=tf.float32,
                                           name='state')
            fc_1 = tf.layers.dense(inputs=self.state_pl, units=8, activation=tf.nn.relu)
            self.output_op = tf.layers.dense(inputs=fc_1, units=action_size, activation=None)
            self.softmax = tf.nn.softmax(self.output_op)

            self.advantage_pl = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32)

            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_pl,
                                                                                         logits=self.output_op) * self.advantage_pl)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.gradients_ops, self.trainable_variables = zip(*self.optimizer.compute_gradients(self.loss_op))
            self.clipped_gradients_ops, self.gradient_global_norms_op = tf.clip_by_global_norm(self.gradients_ops,
                                                                                               clip_norm=1.0)
            self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients_ops, self.trainable_variables),
                                                           global_step=tf.contrib.framework.get_global_step())

            self.check_ops = [tf.check_numerics(o, 'Error') for o in
                              list(self.gradients_ops) + list(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope))]

            self.gradient_holders = []
            for idx, var in enumerate(self.trainable_variables):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)
            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.trainable_variables),
                                                               global_step=tf.contrib.framework.get_global_step())

            self.gradBuffer = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

            self._create_summaries()

    def _create_summaries(self):
        self.action_probs_summary = tf.summary.histogram('actions_probs', self.softmax)

        self.max_gradient_norm_op = tf.reduce_max([tf.reduce_max(g) for g in self.gradients_ops])
        self.max_gradient_norm_summary_op = tf.summary.scalar('max_gradient_norm', self.max_gradient_norm_op)
        self.global_norm_summary_op = tf.summary.scalar('gradient_global_norm', self.gradient_global_norms_op)
        self.loss_summary_op = tf.summary.scalar('loss', self.loss_op)

        self.training_summaries_op = tf.summary.merge(
            [self.max_gradient_norm_summary_op, self.global_norm_summary_op, self.loss_summary_op])

    def predict(self, state):
        a_dist = self.tf_session.run(self.softmax, feed_dict={self.state_pl: [state]})[0]
        action = np.random.choice(np.arange(len(a_dist)), p=a_dist)
        return action

    def update(self, state, action, advantage, apply_grads):
        # Accumulate gradients
        grads = self.tf_session.run(self.gradients_ops, feed_dict={self.advantage_pl: advantage, self.action_pl: action,
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


def summarize_reward(reward, sess, summary_writer):
    if summarize_reward.__dict__.get('reward_placeholder', None) is None:
        summarize_reward.reward_placeholder = tf.placeholder(dtype=tf.float32, name='reward')
        summarize_reward.reward_summary_op = tf.summary.scalar('reward', summarize_reward.reward_placeholder)

    reward_summary, global_step = sess.run([summarize_reward.reward_summary_op, tf.contrib.framework.get_global_step()],
                                           feed_dict={summarize_reward.reward_placeholder: reward})
    summary_writer.add_summary(reward_summary, global_step)


gamma = 0.99  # The discount rate


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


total_episodes = None
max_timesteps_per_episode = 500

env = gym.make('Acrobot-v1')

logdir = '/home/wesley/data/polygons_reinforce'
if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)
summary_writer = tf.summary.FileWriter(logdir)

with tf.Session() as sess:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator(learning_rate=1e-2, state_size=env.observation_space.shape,
                                       action_size=env.action_space.n, tf_session=sess)

    summary_writer.add_graph(tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    total_reward = []

    for episode_num in range(total_episodes) if total_episodes is not None else itertools.count():
        state = env.reset()
        ep_history = []
        total_episode_reward = 0
        for timestep in range(
                max_timesteps_per_episode) if max_timesteps_per_episode is not None else itertools.count():
            # Get an action from the policy estimator
            action = policy_estimator.predict(state)
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_episode_reward += reward
            reward = 5.0 if done else -0.1
            # Save the transition in the replay memory
            ep_history.append([state, action, reward, next_state, done])
            state = next_state
            print('\rtimestep={}'.format(timestep), end='')

            if done:
                break

        # Update the network.
        ep_history = np.array(ep_history)
        ep_history[:, 2] = discount_rewards(ep_history[:, 2])
        policy_estimator.update(state=np.vstack(ep_history[:, 0]), action=ep_history[:, 1], advantage=ep_history[:, 2],
                                apply_grads=episode_num % 5 == 0 and episode_num != 0)

        summarize_reward(total_episode_reward, sess, summary_writer=summary_writer)
        total_reward.append(total_episode_reward)

        # Update our running tally of scores.
        if episode_num % 100 == 0:
            print("\repisode={}\tglobal_step={}\tavg_reward={}".format(episode_num,
                                                                       sess.run(tf.contrib.framework.get_global_step()),
                                                                       np.mean(total_reward[-100:])))
