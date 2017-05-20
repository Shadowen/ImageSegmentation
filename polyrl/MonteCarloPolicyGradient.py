# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Originally a Policy Gradient algorithm, upgraded to an Actor-Critic algorithm.

import itertools
import operator
from functools import reduce

import gym
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
            self.state_pl = tf.placeholder(shape=[None, reduce(operator.mul, state_size)], dtype=tf.float32,
                                           name='state')
            self._action_logits = self._build_graph()
            self.softmax = tf.nn.softmax(self.action_logits)
            # Loss and training
            self.td_target_pl = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_pl = tf.placeholder(shape=[None], dtype=tf.int32)
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_pl,
                                                                                         logits=self.action_logits) * self.td_target_pl)
            # Ensure everything else is initialized
            for p in [self.train_op]:
                if p is None:
                    raise NotImplementedError()
            # Summaries
            self._create_summaries()

    def _build_graph(self):
        fc_1 = tf.layers.dense(inputs=self.state_pl, units=8, activation=tf.nn.relu)
        return tf.layers.dense(inputs=fc_1, units=self.action_size, activation=None)

    @property
    def action_logits(self):
        return self._action_logits

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
        grads = self.tf_session.run(self.gradients_ops, feed_dict={self.td_target_pl: td_target, self.action_pl: action,
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


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """

    gamma = 0.99  # The discount rate

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def reinforce(env, sess, policy_estimator, total_episodes=None, max_timesteps_per_episode=None):
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
        policy_estimator.update(state=np.vstack(ep_history[:, 0]), action=ep_history[:, 1], td_target=ep_history[:, 2],
                                apply_grads=episode_num % 5 == 0 and episode_num != 0)
        # Reward summary
        global_step = sess.run(tf.contrib.framework.get_global_step())
        summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=total_episode_reward)])
        summary_writer.add_summary(summary, global_step=global_step)

        total_reward.append(total_episode_reward)
        # Update our running tally of scores.
        if episode_num % 100 == 0:
            print("\repisode={}\tglobal_step={}\tavg_reward={}".format(episode_num,
                                                                       sess.run(tf.contrib.framework.get_global_step()),
                                                                       np.mean(total_reward[-100:])))


env = gym.make('Acrobot-v1')

logdir = '/home/wesley/data/polygons_reinforce'
reset_dir(logdir)

with tf.Session() as sess:
    tf.Variable(0, name="global_step", trainable=False)
    summary_writer = tf.summary.FileWriter(logdir)
    policy_estimator = PolicyEstimator(state_size=env.observation_space.shape, action_size=env.action_space.n,
                                       tf_session=sess, summary_writer=summary_writer)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    reinforce(env=env, sess=sess, policy_estimator=policy_estimator, total_episodes=None,
              max_timesteps_per_episode=500)
