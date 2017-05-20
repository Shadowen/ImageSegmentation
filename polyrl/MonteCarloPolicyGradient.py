# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Originally a Policy Gradient algorithm, upgraded to an Actor-Critic algorithm.

import itertools

import gym
import tensorflow as tf

from polyrl.PolicyEstimator import PolicyEstimator
from util import *


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    gamma = 0.99  # The discount rate

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def reinforce(env, sess, policy_estimator, total_episodes=None, max_timesteps_per_episode=None, summary_writer=None):
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
        if summary_writer is not None:
            global_step = sess.run(tf.contrib.framework.get_global_step())
            summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=total_episode_reward)])
            summary_writer.add_summary(summary, global_step=global_step)

        total_reward.append(total_episode_reward)
        # Update our running tally of scores.
        if episode_num % 100 == 0:
            print("\repisode={}\tglobal_step={}\tavg_reward={}".format(episode_num,
                                                                       sess.run(tf.contrib.framework.get_global_step()),
                                                                       np.mean(total_reward[-100:])))


if __name__ == '__main__':
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
