import io
import itertools
import os

import matplotlib

matplotlib.use('agg')
import matplotlib.lines
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imresize

from Dataset import get_train_and_valid_datasets
from convLSTM import ConvLSTMCell
from polyrl import MonteCarloPolicyGradient
from polyrl.polygon_env_end_token import PolygonEnv
from util import *


class ExperimentPolicyEstimator():
    def __init__(self, image_size, action_size, tf_session, summary_writer, max_timesteps):
        self.max_timesteps = max_timesteps
        self.history_length = 2

        self.image_size = image_size
        self.action_size = action_size
        self.tf_session = tf_session
        self.summary_writer = summary_writer

        self.global_step = tf.train.get_global_step()
        assert self.global_step is not None

        with tf.variable_scope('policy_estimator'):
            # Build inference graph
            self.image_pl = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='image')
            self.history_pl = tf.placeholder(tf.float32,
                                             shape=[None, max_timesteps, image_size, image_size, self.history_length])

            self.duration_pl = tf.placeholder(tf.int32, shape=[None], name='duration')
            self._build_graph()
            # Loss and training
            self.td_target_pl = tf.placeholder(tf.float32, shape=[None, self.max_timesteps], name='td_target')
            self.action_taken_pl = tf.placeholder(shape=[None, self.max_timesteps, 2], dtype=tf.int32,
                                                  name='action_taken')
            action_x, action_y = [tf.squeeze(c, axis=2) for c in tf.split(self.action_taken_pl, 2, axis=2)]
            action_dense = action_x * self.action_size + action_y
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_dense,
                                                                                         logits=self.action_logits) * self.td_target_pl)
            # Ensure everything else is initialized
            for p in [self.train_op]:
                if p is None:
                    raise NotImplementedError()
                    # Summaries
                    # self._create_summaries()

    def _build_graph(self):
        self.batch_size = tf.shape(self.image_pl)[0]

        # conv1 [batch, x, y, c]
        conv1 = tf.layers.conv2d(inputs=self.image_pl, filters=16, kernel_size=[5, 5], padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv2 [batch, x, y, c]
        conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                 name='conv2')
        # conv3 [batch, x, y, c]
        conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 name='conv3')
        # tiled [batch, timestep, c]
        tiled = tf.tile(tf.expand_dims(conv3, axis=1), multiples=[1, self.max_timesteps, 1, 1, 1])
        concat = tf.concat([tiled, self.history_pl], axis=4)

        with tf.variable_scope('rnn'):
            # Make placeholder time major for RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_input = tf.transpose(concat, (1, 0, 2, 3, 4))

            rnn_cell = lambda: ConvLSTMCell(shape=[28, 28], filters=16, kernel=[3, 3])
            rnn_layers = 2
            multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(rnn_layers)])
            self._rnn_zero_state_tensor = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self._rnn_initial_state_pl = tuple(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder(tf.float32, shape=[None] + multi_rnn_cell.state_size[i].c.as_list(),
                               name='rnn_initial_state_c'),
                tf.placeholder(tf.float32, shape=[None] + multi_rnn_cell.state_size[i].h.as_list(),
                               name='rnn_initial_state_h')) for i in
                                               range(rnn_layers))
            rnn_output, self._rnn_final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=rnn_input,
                                                                  sequence_length=self.duration_pl,
                                                                  initial_state=self._rnn_initial_state_pl,
                                                                  time_major=True)
            # Make placeholder batch major again after RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2, 3, 4))

        self.action_logits = tf.reshape(tf.layers.dense(inputs=tf.reshape(rnn_output, shape=[-1, 28 * 28 * 16]),
                                                        units=self.action_size * self.action_size + 1,
                                                        name='prediction'),
                                        shape=[-1, self.max_timesteps, self.action_size * self.action_size + 1])

        # Sample
        first_action = tf.squeeze(tf.slice(self.action_logits, begin=[0, 0, 0],
                                           size=[-1, 1, -1]), axis=1)  # The action for the first timestep
        action_sample_dense = tf.squeeze(
            tf.multinomial(logits=first_action - tf.reduce_max(first_action, axis=[1], keep_dims=True), num_samples=1),
            [1])
        self.action_sample_coords = tf.stack(
            [action_sample_dense // self.action_size, action_sample_dense % self.action_size], axis=1)

    def get_rnn_zero_state(self, batch_size):
        return self.tf_session.run(self._rnn_zero_state_tensor, feed_dict={
            self.image_pl: np.empty([batch_size, self.image_size, self.image_size, 3])})

    @lazyproperty
    def train_op(self):
        """
        Define the train_op here. The gradients, clipping, and optimizer should be defined here as well.
        Because this is a @lazyproperty, this will only be called once to initialize. The result will automatically be
        cached in the property.
        :return:
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
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
                                                           global_step=self.global_step)

        self.gradBuffer = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]
        return self._train_op

    def predict(self, image, history, rnn_state):
        """Picks a single action given image and history input. Does not handle batches!"""
        h = np.concatenate([np.expand_dims(np.expand_dims(history, axis=0), axis=1),
                            np.zeros([1, 9, self.image_size, self.image_size, 2])], axis=1)
        action, rnn_final_state = self.tf_session.run(
            [self.action_sample_coords, self._rnn_final_state],
            feed_dict={self.image_pl: [image],
                       self.history_pl: h,
                       self._rnn_initial_state_pl: [rnn_state],
                       self.duration_pl: [1]})
        return action[0], rnn_final_state

    def update(self, images, histories, actions, td_targets, apply_grads):
        """
        Updates the policy estimator according to the policy gradient given state, action, and td-target tuples.
        Only actually applies the gradient back to the estimator if apply_grads=True.
        """
        num_timesteps_given = histories.shape[0]
        # Accumulate gradients
        td = [np.concatenate([td_targets, np.zeros([self.max_timesteps - num_timesteps_given])], axis=0)]
        at = [np.concatenate([actions, np.zeros([self.max_timesteps - num_timesteps_given, 2])], axis=0)]
        h = [np.concatenate([histories, np.zeros(
            [self.max_timesteps - num_timesteps_given, self.image_size, self.image_size, self.history_length])],
                            axis=0)]
        r = self.get_rnn_zero_state(1)
        d = np.ones([1]) * len(histories)
        grads = self.tf_session.run(self.gradients_ops,
                                    feed_dict={self.td_target_pl: td,
                                               self.action_taken_pl: at,
                                               self.image_pl: [images],
                                               self.history_pl: h,
                                               self._rnn_initial_state_pl: r,
                                               self.duration_pl: d})
        for idx, grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        if apply_grads:
            global_step, _ = self.tf_session.run(
                [self.global_step, self.update_batch],
                feed_dict=dict(zip(self.gradient_holders, self.gradBuffer)))
            for ix, grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad * 0

                # self.summary_writer.add_summary(summaries, global_step)

    def _create_summary_image(self, image, true_vertices, predicted_vertices, predicted_mask, iou):
        """
        Plots the image, true vertices, and predicted vertices in matplotlib and returns it as a tf.Summary.Image
        Usage:
            image_summary = tf.Summary(value=[tf.Summary.Value(
                    tag=('image', image=Model._create_summary_image(image, true_vertices, predicted_vertices))])
            summary_writer.add_summary(image_summary, global_step=step)

        :param image: A NumPy array of shape (self.image_size, self.image_size, 3)
        :param true_vertices: a NumPy array of shape (self.max_timesteps, 2)
        :param predicted_vertices: a NumPy array of shape (self.max_timesteps, 2)
        :returns: tf.Summary.Image
        """
        plt.ioff()
        fig = plt.figure()
        ax = plt.gca()
        resized_image = imresize(image, [self.action_size, self.action_size], interp='nearest')
        plt.imshow(resized_image)
        for e, v in enumerate(true_vertices):
            ax.add_artist(plt.Circle(v, radius=0.5, color='lightgreen', alpha=0.5))
        for a, b in iterate_in_ntuples(true_vertices, n=2):
            ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='lawngreen'))
        # History points
        for e, v in enumerate(true_vertices[-self.history_length:]):
            ax.add_artist(plt.Circle(v, radius=0.5, color='lightgreen', alpha=0.5))
            plt.text(v[0], v[1] + e / 2, e - self.history_length + 1, color='lightgreen')
        for e, v in enumerate(predicted_vertices):
            ax.add_artist(plt.Circle(v, radius=0.5, color='salmon', alpha=0.5))
            plt.text(v[0], v[1] + e / 2, e + 1, color='red')
        for a, b in iterate_in_ntuples(predicted_vertices, n=2, loop=False):
            ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='tomato'))
        if np.max(predicted_mask) > 0:
            z = np.zeros_like(predicted_mask)
            plt.show()
            plt.imshow(np.stack([predicted_mask, z, z], axis=2), alpha=0.5)
        # for e, v in enumerate(true_vertices):
        #     plt.text(v[0] - e / 2, v[1], e, color='blue')
        plt.text(0, 0, 'IOU={}'.format(iou), color='red')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return tf.Summary.Image(encoded_image_string=buf.getvalue())

    def validate_iou(self, env, max_timesteps_per_episode=None, summary_prefix='valid', additional_feed_args={}):
        """
        Validates the IoU (Intersection over Union) for the given set of images and vertices.

        :param images: a NumPy np.float array of shape (batch_size, self.image_size, self.image_size, 32)
        :param true_vertices: a NumPy array of shape (batch_size, self.max_timesteps, 2)
        :param summary_prefix: a string to prefix the generated summary names with
        :param additional_feed_args: Dictionary of nonstandard values that should be fed during prediction (ie. dropout)
        :return:
        """
        state = env.reset()
        true_vertices = env.vertices
        image = state[:, :, :3]
        predicted_vertices = []
        total_episode_reward = 0
        rnn_state = policy_estimator.get_rnn_zero_state(batch_size=1)
        for timestep in range(
                max_timesteps_per_episode) if max_timesteps_per_episode is not None else itertools.count():
            # Get an action from the policy estimator
            histories = state[:, :, 3:]
            action, next_rnn_state = policy_estimator.predict(image=image, history=histories, rnn_state=rnn_state)
            # Take a step in the environment
            next_state, reward, done = env.step(action)
            total_episode_reward += reward
            # Save the predicted vertex
            predicted_vertices.append(action)
            state = next_state
            rnn_state = next_rnn_state
            if done:
                break

        predicted_mask = create_shape_mask(
            np.concatenate([true_vertices[-self.history_length:], predicted_vertices], axis=0),
            self.action_size)
        print('Evaluated IOU={}'.format(total_episode_reward))

        # Summary stuff
        image_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=summary_prefix + '/iou_image' if summary_prefix else 'iou_image',
                image=self._create_summary_image(image,
                                                 true_vertices,
                                                 predicted_vertices,
                                                 predicted_mask, total_episode_reward))])
        step = self.tf_session.run(self.global_step)
        self.summary_writer.add_summary(image_summary, global_step=step)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=summary_prefix + '/iou' if summary_prefix else 'iou', simple_value=iou)])
        self.summary_writer.add_summary(summary, global_step=step)


def reinforce(env, sess, policy_estimator, total_episodes=None, max_timesteps_per_episode=None, summary_writer=None):
    total_reward = []
    for episode_num in range(total_episodes) if total_episodes is not None else itertools.count():
        state = env.reset()
        image = state[:, :, :3]
        ep_history = []
        total_episode_reward = 0
        rnn_state = policy_estimator.get_rnn_zero_state(batch_size=1)
        for timestep in range(
                max_timesteps_per_episode) if max_timesteps_per_episode is not None else itertools.count():
            # Get an action from the policy estimator
            histories = state[:, :, 3:]
            action, next_rnn_state = policy_estimator.predict(image=image, history=histories, rnn_state=rnn_state)
            # Take a step in the environment
            next_state, reward, done = env.step(action)
            total_episode_reward += reward
            # Save the transition in the replay memory
            ep_history.append([state, action, reward, next_state, done, rnn_state])
            state = next_state
            rnn_state = next_rnn_state
            print('\rtimestep={}'.format(timestep), end='')

            if done:
                break

        # Update the network if we saw any reward
        if reward > 0:
            print('Updating network! (reward={})'.format(reward))
            ep_history = np.array(ep_history)
            ep_history[:, 2] = MonteCarloPolicyGradient.discount_rewards(ep_history[:, 2])
            h = np.stack([ep_history[i, 0][:, :, 3:] for i in range(ep_history.shape[0])], axis=0)
            policy_estimator.update(images=image, histories=h, actions=np.vstack(ep_history[:, 1]),
                                    td_targets=ep_history[:, 2], apply_grads=episode_num % 5 == 0 and episode_num != 0)
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

        # Validate using IoU
        if episode_num % 100 == 0:
            policy_estimator.validate_iou(env, max_timesteps_per_episode=max_timesteps_per_episode)


max_timesteps = 10

training_env, validation_env = [PolygonEnv(d) for d in
                                get_train_and_valid_datasets('/home/wesley/data/polygons_dataset_2',
                                                             max_timesteps=10,
                                                             image_size=28,
                                                             prediction_size=28,
                                                             history_length=2,
                                                             is_local=True,
                                                             load_max_images=2,
                                                             validation_set_percentage=0.5)]

logdir = '/home/wesley/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
reset_dir(logdir)

with tf.Session() as sess:
    tf.Variable(0, name="global_step", trainable=False)
    summary_writer = tf.summary.FileWriter(logdir)
    policy_estimator = ExperimentPolicyEstimator(image_size=28, action_size=28,
                                                 tf_session=sess, summary_writer=summary_writer,
                                                 max_timesteps=max_timesteps)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    reinforce(env=training_env, sess=sess, policy_estimator=policy_estimator,
              total_episodes=None, max_timesteps_per_episode=max_timesteps,
              summary_writer=summary_writer)
