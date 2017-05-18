import io
import os

import matplotlib
import numpy as np
import tensorflow as tf

from polyrnn import Model
from polyrnn.EndTokenDataset import get_train_and_valid_datasets, _create_history
from polyrnn.convLSTM import ConvLSTMCell

matplotlib.use('agg')
import matplotlib.lines
import matplotlib.pyplot as plt
from scipy.misc import imresize
from polyrnn.util import lazyproperty, create_shape_mask, iterate_in_ntuples


class ExperimentModel(Model.Model):
    def _build_graph(self):
        # conv1 [batch, x, y, c]
        conv1 = tf.layers.conv2d(inputs=self.image_pl / 255, filters=16, kernel_size=[5, 5], padding='same',
                                 activation=tf.nn.relu, name='conv1')
        # conv2 [batch, x, y, c]
        x = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                             name='conv2')
        for n in range(3, 8):
            x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                 name='conv{}'.format(n))
        # tiled [batch, timestep, c]
        tiled = tf.tile(tf.expand_dims(x, axis=1), multiples=[1, self.max_timesteps, 1, 1, 1])
        concat = tf.concat([tiled, self.history_pl], axis=4)

        with tf.variable_scope('rnn'):
            # Make placeholder time major for RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_input = tf.transpose(concat, (1, 0, 2, 3, 4))

            rnn_cell = lambda: ConvLSTMCell(shape=[28, 28], filters=16, kernel=[3, 3])
            rnn_layers = 5
            multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(rnn_layers)])
            self._rnn_zero_state = multi_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            self._rnn_initial_state_pl = tuple(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(self.rnn_zero_state[i].c,
                                            shape=[None] + multi_rnn_cell.state_size[i].c.as_list()),
                tf.placeholder_with_default(self.rnn_zero_state[i].h,
                                            shape=[None] + multi_rnn_cell.state_size[i].h.as_list())) for i in
                                               range(rnn_layers))
            rnn_output, self._rnn_final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=rnn_input,
                                                                  sequence_length=self._duration_pl,
                                                                  initial_state=self._rnn_initial_state_pl,
                                                                  time_major=True)
            # Make placeholder batch major again after RNN. (see https://github.com/tensorflow/tensorflow/pull/5142)
            rnn_output = tf.transpose(rnn_output, (1, 0, 2, 3, 4))
            flat_rnn_output = tf.reshape(rnn_output, shape=[-1, 28 * 28 * 16])

        self.prediction_logits = tf.reshape(
            tf.layers.dense(inputs=flat_rnn_output, units=prediction_size ** 2 + 1, name='prediction'),
            shape=[-1, self.max_timesteps, self.prediction_size ** 2 + 1])

    @lazyproperty
    def rnn_initial_state_pl(self):
        return self._rnn_initial_state_pl

    @lazyproperty
    def rnn_zero_state(self):
        return self._rnn_zero_state

    @lazyproperty
    def rnn_final_state(self):
        return self._rnn_final_state

    @lazyproperty
    def prediction_logits(self):
        return self._prediction_logits

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
        resized_image = imresize(image, [self.prediction_size, self.prediction_size], interp='nearest')
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
            # End token
            if np.all(v == np.array([self.prediction_size, 0])):
                v = true_vertices[-self.history_length + 1]
                ax.add_artist(plt.Circle(v, radius=0.5, color='salmon', alpha=0.5))
                plt.text(v[0], v[1] + e / 2, 'END', color='red')
                break
            ax.add_artist(plt.Circle(v, radius=0.5, color='salmon', alpha=0.5))
            plt.text(v[0], v[1] + e / 2, e + 1, color='red')
        for a, b in iterate_in_ntuples(predicted_vertices, n=2, loop=False):
            # End token
            if np.all(b == np.array([self.prediction_size, 0])):
                ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='tomato'))
                break
            ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='tomato'))
        if np.max(predicted_mask) > 0:
            z = np.zeros_like(predicted_mask)
            plt.show()
            plt.imshow(np.stack([predicted_mask, z, z], axis=2), alpha=0.5)
        plt.text(0, 0, 'IOU={}'.format(iou), color='red')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return tf.Summary.Image(encoded_image_string=buf.getvalue())

    def validate_iou(self, images, true_vertices, summary_prefix='valid', additional_feed_args={}):
        """
        Validates the IoU (Intersection over Union) for the given set of images and vertices.

        :param images: a NumPy np.float array of shape (batch_size, self.image_size, self.image_size, 32)
        :param true_vertices: a NumPy array of shape (batch_size, self.max_timesteps, 2)
        :param summary_prefix: a string to prefix the generated summary names with
        :param additional_feed_args: Dictionary of nonstandard values that should be fed during prediction (ie. dropout)
        :return:
        """
        batch_size = images.shape[0]
        predicted_vertices = np.empty([batch_size, self.max_timesteps, 2])

        # Prepare fake histories
        zs = [np.zeros([self.prediction_size, self.prediction_size, self.history_length]) for _ in
              range(self.max_timesteps - 1)]
        histories = np.array([np.stack(
            [_create_history(v, len(v) - 1, self.history_length, self.prediction_size)] + zs, axis=0)
                              for v in true_vertices])

        step, rnn_state = self.sess.run([self.global_step, self.rnn_zero_state], feed_dict={self.image_pl: images})
        prediction_duration_required = np.ones([batch_size], dtype=np.uint32)
        prediction_durations = np.zeros([batch_size], dtype=np.uint32)
        for i in range(self.max_timesteps):
            out, rnn_state = self.sess.run([self.prediction_max, self.rnn_final_state],
                                           feed_dict={self.image_pl: images,
                                                      self.duration_pl: prediction_duration_required,
                                                      self.history_pl: histories, self.rnn_initial_state_pl: rnn_state,
                                                      **additional_feed_args})
            predicted_vertices[:, i, :] = out[:, 0, :]
            prediction_durations += prediction_duration_required
            for b in range(batch_size):
                if np.all(predicted_vertices[b, i, :] == np.array([self.prediction_size, 0])):
                    # This shape is done; stop predicting
                    prediction_duration_required[b] = 0
                    predicted_vertices[b, i, :] = true_vertices[b][-self.history_length]
                    out[b, 0, :] = 0

            predicted_mask = np.zeros([batch_size, self.prediction_size, self.prediction_size])
            for b in range(batch_size):
                predicted_mask[b, out[b, 0, 1], out[b, 0, 0]] = 1

            histories = np.roll(histories, axis=4, shift=1)
            histories[:, 0, :, :, -1] = predicted_mask

        ious = []
        for i in range(predicted_vertices.shape[0]):
            predicted_mask = create_shape_mask(
                np.concatenate(
                    [true_vertices[i][-self.history_length:], predicted_vertices[i][:prediction_durations[i]]], axis=0),
                self.prediction_size)
            true_mask = create_shape_mask(true_vertices[i], self.prediction_size)
            intersection = np.count_nonzero(np.logical_and(predicted_mask, true_mask))
            union = np.count_nonzero(np.logical_or(predicted_mask, true_mask))
            iou = intersection / union
            ious.append(iou)

            # Summary stuff
            image_summary = tf.Summary(
                value=[tf.Summary.Value(
                    tag=(summary_prefix + '/iou_image_{}' if summary_prefix else 'iou_image_{}').format(i),
                    image=self._create_summary_image(images[i],
                                                     true_vertices[i],
                                                     predicted_vertices[i, 0:prediction_durations[i]],
                                                     predicted_mask, iou))])
            self.summary_writer.add_summary(image_summary, global_step=step)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=summary_prefix + '/iou' if summary_prefix else 'iou', simple_value=np.mean(ious))])
        self.summary_writer.add_summary(summary, global_step=step)


if __name__ == '__main__':
    import shutil

    image_size = 28
    prediction_size = 28
    max_timesteps = 10
    history_length = 2

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.Session() as sess:
        model_dir = '/data/{}/'.format(os.path.splitext(os.path.basename(__file__))[0])
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        train_data, valid_data = get_train_and_valid_datasets('/data/polygons_dataset_3',
                                                              max_timesteps=max_timesteps,
                                                              image_size=image_size,
                                                              prediction_size=prediction_size,
                                                              history_length=history_length, is_local=True,
                                                              load_max_images=100000, validation_set_percentage=0.1)

        model = ExperimentModel(sess, max_timesteps, image_size, prediction_size, history_length, model_dir)
        sess.run(tf.global_variables_initializer())
        model.maybe_restore()

        total_steps = 150000
        for step_num in range(total_steps):
            batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=16)
            model.train(batch_images, batch_d, batch_h, batch_t)

            if step_num % 100 == 0:
                print('Step {}'.format(step_num))
                batch_d, batch_images, batch_h, batch_t, batch_vertices = valid_data.get_batch_for_rnn(batch_size=8)
                model.validate(batch_images, batch_d, batch_h, batch_t)

            if step_num % 1000 == 0:
                batch_images, batch_vertices = train_data.raw_sample(batch_size=8)
                model.validate_iou(batch_images, batch_vertices, summary_prefix='train')

                batch_images, batch_vertices = valid_data.raw_sample(batch_size=8)
                model.validate_iou(batch_images, batch_vertices, summary_prefix='valid')
                model.save()
