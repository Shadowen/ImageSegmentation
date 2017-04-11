from __future__ import division
import sys

sys.path.append('.')
import tensorflow as tf
import operator
from functools import reduce
from supervised_vertices.convLSTM import ConvLSTMCell, flatten, expand
import numpy as np


class RNN_Estimator(object):
    def __init__(self, image_size, use_pretrained=True):
        """ Create an RNN.

        Args:
            init_scale: A float. All weight matrices will be initialized using
                a uniform distribution over [-init_scale, init_scale].
        """

        self._image_size = image_size
        self.input_shape = [self._image_size, self._image_size]

        ## Feed Vars
        # A Tensor of shape [None, 227, 227, 3]
        self._image_input = tf.placeholder(tf.float32, shape=[None] + self.input_shape + [3],
                                           name='inputs')
        # A scalar
        self.iou = tf.placeholder(dtype=tf.float32, shape=[], name='iou')
        self.failures = tf.placeholder(dtype=tf.float32, shape=[], name='failure_rate')

        ## Fetch Vars
        init_scale = 0.1
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._inputs_unrolled = tf.reshape(self._image_input,
                                               shape=[-1, reduce(operator.mul, self._image_input.shape.as_list()[1:])])
            self._create_inference_graph()

            # A Tensor of shape [None, self.prediction_size, self.prediction_size]
            self._targets = tf.placeholder(tf.float32, shape=[None, self.prediction_size, self.prediction_size],
                                           name='targets')
            self._targets_unrolled = tf.reshape(self._targets,
                                                shape=[-1, reduce(operator.mul, self._targets.shape.as_list()[1:])])
            self._create_loss_graph()
            self._create_optimizer(initial_learning_rate=1e-2, num_steps_per_decay=100000,
                                   decay_rate=0.05, max_global_norm=1.0)
        with tf.variable_scope('train'):
            self._training_summaries, self._training_image_summaries = self._build_summaries()
        with tf.variable_scope('valid'):
            self._validation_summaries, self._validation_image_summaries = self._build_summaries()

    def _create_inference_graph(self):
        """ Create a CNN that feeds an RNN. """

        print('Loading AlexNet weights from bvlc_alexnet.npy')
        """AlexNet frontend from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/"""
        net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow'''
            c_i = input.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(input, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(self._image_input, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        upsample_factor = 3
        lstm_image_size = (upsample_factor * maxpool5.shape.as_list()[1], upsample_factor * maxpool5.shape.as_list()[2])
        upsampled = tf.image.resize_nearest_neighbor(maxpool5, lstm_image_size)

        # A Tensor of shape [None, self.prediction_size, self.prediction_size]
        self._cursor_mask = tf.placeholder(tf.float32, shape=[None, *lstm_image_size], name='cursor_mask')
        # A Tensor of shape [None, self.prediction_size, self.prediction_size]
        self._history_mask = tf.placeholder(tf.float32, shape=[None, *lstm_image_size], name='history_mask')
        x = tf.concat([upsampled, tf.expand_dims(self.cursor_mask, axis=3), tf.expand_dims(self.history_mask, axis=3)],
                      axis=3)
        # Make x ready to go into an LSTM
        x = flatten(tf.expand_dims(x, axis=0))

        num_lstm_filters = 32
        lstm_cell = ConvLSTMCell(height=lstm_image_size[0], width=lstm_image_size[1],
                                 filters=num_lstm_filters, kernel=[3, 3])
        self._c_init = tf.zeros([1, lstm_cell.state_size.c], dtype=tf.float32)
        self._h_init = tf.zeros([1, lstm_cell.state_size.h], dtype=tf.float32)
        self._c_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.c], name='c_in')
        self._h_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.h], name='h_in')
        self.seq_length = tf.shape(self._image_input, out_type=tf.int32)[0]
        lstm_outputs, self._lstm_final_state = tf.nn.dynamic_rnn(lstm_cell, x,
                                                                 initial_state=tf.contrib.rnn.LSTMStateTuple(self._c_in,
                                                                                                             self._h_in),
                                                                 # sequence_length=tf.expand_dims(self.seq_length,
                                                                 #                                axis=0)
                                                                 )
        lstm_outputs = tf.squeeze(expand(lstm_outputs, height=lstm_image_size[0], width=lstm_image_size[1],
                                         filters=num_lstm_filters), axis=0)

        # Output Layer
        self._logits = tf.squeeze(tf.layers.conv2d(inputs=lstm_outputs, filters=1, kernel_size=(3, 3), padding='same'),
                                  axis=3)
        self._prediction_size = self._logits.shape.as_list()[1]
        self._logits_unrolled = tf.reshape(self._logits, shape=[-1, self._prediction_size ** 2])
        self._logits = tf.reshape(self._logits_unrolled, shape=[-1, self._prediction_size, self._prediction_size])

        self._softmax_unrolled = tf.nn.softmax(self._logits_unrolled)
        self._softmax = tf.reshape(self._softmax_unrolled, shape=[-1, self._prediction_size, self._prediction_size])

    def _create_loss_graph(self):
        """ Compute cross entropy loss between targets and predictions. Also compute the L2 error. """

        # Loss
        with tf.variable_scope('loss'):
            self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits_unrolled,
                                                                                         labels=self._targets_unrolled),
                                                 name='cross_entropy')
            self._loss = self._cross_entropy

        self._predictions_coords = [tf.mod(tf.argmax(self._logits_unrolled, dimension=1), self._prediction_size),
                                    tf.floordiv(tf.argmax(self._logits_unrolled, dimension=1), self._prediction_size)]
        self._predictions_coords = tf.stack(self._predictions_coords, axis=1)
        self._target_coords = [tf.mod(tf.argmax(self._targets_unrolled, dimension=1), self._prediction_size),
                               tf.floordiv(tf.argmax(self._targets_unrolled, dimension=1), self._prediction_size)]
        self._target_coords = tf.stack(self._target_coords, axis=1)

        # Number of pixels correct
        with tf.variable_scope('accuracy'):
            x_correct, y_correct = tf.split(tf.equal(self._predictions_coords, self._target_coords), 2, axis=1)
            self._accuracy = tf.count_nonzero(tf.logical_and(x_correct, y_correct)) / tf.cast(self.seq_length,
                                                                                              dtype=tf.int64)

        # L2 error
        with tf.variable_scope('error'):
            self._individual_error = tf.sqrt(
                tf.to_float(tf.reduce_sum((self._predictions_coords - self._target_coords) ** 2, 1)))
            self._error = tf.reduce_mean(self._individual_error)
            self._max_error = tf.reduce_max(self._individual_error)

    def _create_optimizer(self, initial_learning_rate, num_steps_per_decay,
                          decay_rate, max_global_norm=1.0):
        """ Create a simple optimizer.

        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.

        Args:
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        Returns:
            An optimizer op for the given loss function
        """

        trainables = tf.trainable_variables()
        self._grads = tf.gradients(self.loss, trainables)
        self._grads, _ = tf.clip_by_global_norm(self._grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(self._grads, trainables)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self._learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, num_steps_per_decay,
                                                         decay_rate,
                                                         staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)

    def _build_summaries(self):
        """Creates summary operations from existing Tensors"""
        learning_rate_summary = tf.summary.scalar('learning_rate', self._learning_rate)
        loss_summary = tf.summary.scalar('loss', self._loss)
        grad_norm_summary = tf.summary.scalar('grad_norm', sum(tf.norm(g) for g in self._grads))
        accuracy_summary = tf.summary.scalar('accuracy', self._accuracy)
        error_summary = tf.summary.scalar('error', self.error)
        max_error_summary = tf.summary.scalar('max_error', self._max_error)
        scalar_summaries = tf.summary.merge(
            [learning_rate_summary, loss_summary, accuracy_summary, error_summary, max_error_summary,
             grad_norm_summary])

        # slices = tf.split(self.inputs, 3, axis=3)
        # flat_images = tf.image.rgb_to_grayscale(tf.concat(slices[:3], axis=3))
        # flat_images = tf.expand_dims(
        #     tf.expand_dims(tf.expand_dims(1 / tf.reduce_max(flat_images, axis=[1, 2, 3]), axis=1), axis=2),
        #     axis=3) * flat_images
        # inputs_with_flat_images = tf.concat([flat_images, slices[-2], slices[-1]], axis=3,
        #                                     name='inputs_with_flat_images')
        # inputs_with_flat_images = tf.concat(slices, axis=3, name='inputs_with_flat_images')
        input_visualization_summary = tf.summary.image('Inputs', self._image_input, max_outputs=20)
        output_visualization_summary = tf.summary.image('Outputs', tf.expand_dims(
            tf.reshape(self.softmax, shape=[-1, self._prediction_size, self._prediction_size]), dim=3), max_outputs=20)
        # target_visualization_summary = tf.summary.image('Targets', tf.expand_dims(self.targets, dim=3), max_outputs=20)
        image_summaries = tf.summary.merge([input_visualization_summary, output_visualization_summary])

        return scalar_summaries, image_summaries

    @property
    def image_input(self):
        """ An float32 placeholder Tensor with shape `[...]`. """
        return self._image_input

    @property
    def cursor_mask(self):
        return self._cursor_mask

    @property
    def history_mask(self):
        return self._history_mask

    @property
    def lstm_init_state(self):
        """ The initial state of the LSTM. """
        return self._c_init, self._h_init

    @property
    def prediction_size(self):
        """ The predictions are [prediction_size, prediction_size]"""
        return self._prediction_size

    @property
    def targets(self):
        """ An n-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets

    @property
    def softmax(self):
        return self._softmax

    @property
    def predictions(self):
        """ An n-D float32 Tensor with shape `[batch_size, max_duration, target_size]`. """
        return self._logits_unrolled

    @property
    def lstm_final_state(self):
        """ The final state of the LSTM after prediction. """
        return self._lstm_final_state

    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        return self._loss

    @property
    def error(self):
        """ A 0-D float32 Tensor. """
        return self._error

    @property
    def train_op(self):
        """ A training operation to optimize the loss function"""
        return self._train_op

    @property
    def training_summaries(self):
        return self._training_summaries

    @property
    def training_image_summaries(self):
        return self._training_image_summaries

    @property
    def validation_summaries(self):
        return self._validation_summaries

    @property
    def validation_image_summaries(self):
        return self._validation_image_summaries


def evaluate_iou(sess, est, dataset, max_timesteps=10, batch_size=None, logdir=None):
    import numpy as np
    from supervised_vertices.Dataset import _create_history_mask, _create_point_mask, _create_shape_mask
    from supervised_vertices.helper import seg_intersect

    batch_size = batch_size if batch_size is not None else len(dataset)

    # IOU
    failed_shapes = 0
    ious = []
    failed_images = []

    for image_number, (image, poly_verts, ground_truth) in enumerate(dataset.raw_sample(batch_size=batch_size)):
        cursor = poly_verts[np.random.randint(len(poly_verts))]
        prediction_vertices = [cursor]
        lstm_c, lstm_h = sess.run(est.lstm_init_state)

        previous_states = []
        previous_softmaxes = []

        # Try to predict the polygon!
        polygon_complete = False
        for t in range(max_timesteps):
            history_mask = _create_history_mask(prediction_vertices, len(prediction_vertices), dataset._prediction_size)
            cursor_mask = _create_point_mask(cursor, dataset._prediction_size)
            state = image
            # state = np.concatenate([image, np.stack([history_mask, cursor_mask], axis=2)], axis=2)
            previous_states.append(state)
            # Feed this one state, but use the previous LSTM state.
            # Basically generate the RNN output one step at a time.
            softmax, pred_coords, (lstm_c, lstm_h) = sess.run(
                [est.softmax, est._predictions_coords, est.lstm_final_state],
                {est.image_input: np.expand_dims(state, axis=0), est.cursor_mask: np.expand_dims(cursor_mask, axis=0),
                 est.history_mask: np.expand_dims(history_mask, axis=0), est._c_in: lstm_c, est._h_in: lstm_h})
            previous_softmaxes.append(softmax[0])
            cursor = tuple(reversed(pred_coords[0].tolist()))

            # Self intersecting shape
            for i in range(1, len(prediction_vertices)):
                does_intersect, intersection = seg_intersect(np.array(prediction_vertices[i - 1]),
                                                             np.array(prediction_vertices[i]),
                                                             np.array(prediction_vertices[-1]), np.array(cursor))
                if does_intersect and not np.all(intersection == prediction_vertices[-1]):
                    # Calculate IOU
                    predicted_polygon = _create_shape_mask(prediction_vertices, dataset.image_size)
                    intersection = np.count_nonzero(predicted_polygon * ground_truth)
                    union = np.count_nonzero(predicted_polygon) + np.count_nonzero(ground_truth) - intersection
                    ious.append(intersection / union) if union != 0 else None
                    polygon_complete = True
                    break
            prediction_vertices.append(cursor)

            if polygon_complete:
                break
        history_mask = _create_history_mask(prediction_vertices, len(prediction_vertices), dataset.image_size)
        cursor_mask = _create_point_mask(cursor, dataset.image_size)
        # state = np.concatenate([image, np.stack([history_mask, cursor_mask], axis=2)], axis=2)
        state = image
        previous_states.append(state)

        if not polygon_complete:
            # If we run for too many time steps
            failed_shapes += 1
            failed_images.append(state)

        # Save some pictures!
        if logdir is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=previous_states[0].shape[-1] + 1, ncols=len(previous_states), sharex=True,
                                   sharey=True)
            plt.axis('off')
            plt.suptitle(
                'image_number={}  '.format(image_number) + (
                    'IOU={}'.format(ious[-1]) if polygon_complete else 'FAILED'))
            for i in range(len(previous_states)):
                for e in range(previous_states[i].shape[-1]):
                    ax[e][i].imshow(previous_states[i][:, :, e], cmap='gray', interpolation='nearest')
            for i in range(len(previous_softmaxes)):
                ax[e + 1][i].imshow(previous_softmaxes[i], cmap='gray', interpolation='nearest')
            plt.savefig('{}/{}.png'.format(logdir, image_number))
            plt.close()

    return failed_shapes / batch_size, (sum(ious) / len(ious)) if len(ious) > 0 else 0, failed_images
