from __future__ import division
import sys

sys.path.append('.')
import tensorflow as tf
import operator
from functools import reduce
from supervised_vertices.convLSTM import ConvLSTMCell, flatten, expand


class RNN_Estimator(object):
    def __init__(self, image_size, input_channels, prediction_size):
        """ Create an RNN.

        Args:
            init_scale: A float. All weight matrices will be initialized using
                a uniform distribution over [-init_scale, init_scale].
        """

        self._image_size = image_size
        self.input_shape = [self._image_size, self._image_size, input_channels]
        self.target_shape = [prediction_size, prediction_size]

        ## Feed Vars
        # A Tensor of shape [None, ...]
        self._inputs = tf.placeholder(tf.float32, shape=[None] + self.input_shape,
                                      name='inputs')
        # A Tensor of shape [None, ...]
        self._targets = tf.placeholder(tf.float32, shape=[None] + self.target_shape,
                                       name='targets')
        # A scalar
        self.iou = tf.placeholder(dtype=tf.float32, shape=[], name='iou')
        self.failures = tf.placeholder(dtype=tf.float32, shape=[], name='failure_rate')

        ## Fetch Vars
        init_scale = 0.1
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._inputs_unrolled = tf.reshape(self._inputs,
                                               shape=[-1, reduce(operator.mul, self._inputs.shape.as_list()[1:])])
            self._create_inference_graph()
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

        x = self._inputs
        filters = [16, 32, 32, 32]
        kernel_sizes = [(5, 5), (5, 5), (3, 3), (3, 3)]
        strides = [(2, 2), (2, 2), (1, 1), (1, 1)]
        for i in range(len(kernel_sizes)):
            x = tf.layers.conv2d(inputs=x, filters=filters[i], name="l{}".format(i + 1), kernel_size=kernel_sizes[i],
                                 strides=strides[i], padding='same', activation=tf.nn.relu)
        # Make x ready to go into an LSTM
        x = flatten(tf.expand_dims(x, axis=0))

        num_lstm_filters = 32
        lstm_cell = ConvLSTMCell(height=self._image_size, width=self._image_size, filters=num_lstm_filters,
                                 kernel=[3, 3])
        self._c_init = tf.zeros([1, lstm_cell.state_size.c], dtype=tf.float32)
        self._h_init = tf.zeros([1, lstm_cell.state_size.h], dtype=tf.float32)
        self._c_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.c], name='c_in')
        self._h_in = tf.placeholder(tf.float32, shape=[1, lstm_cell.state_size.h], name='h_in')
        self.seq_length = tf.shape(self._inputs, out_type=tf.int32)[0]
        lstm_outputs, self._lstm_final_state = tf.nn.dynamic_rnn(lstm_cell, x,
                                                                 initial_state=tf.contrib.rnn.LSTMStateTuple(self._c_in,
                                                                                                             self._h_in),
                                                                 # sequence_length=tf.expand_dims(self.seq_length,
                                                                 #                                axis=0)
                                                                 )
        lstm_outputs = tf.squeeze(expand(lstm_outputs, height=self._image_size, width=self._image_size,
                                         filters=num_lstm_filters), axis=0)

        # Output Layer
        self._logits = tf.squeeze(tf.layers.conv2d(inputs=lstm_outputs, filters=1, kernel_size=(3, 3), padding='same'),
                                  axis=3)
        self._logits_unrolled = tf.reshape(self._logits, shape=[-1, self._image_size ** 2])
        self._logits = tf.reshape(self._logits_unrolled, shape=[-1, self._image_size, self._image_size])

        self._softmax_unrolled = tf.nn.softmax(self._logits_unrolled)
        self._softmax = tf.reshape(self._softmax_unrolled, shape=[-1] + self.target_shape)

    def _create_loss_graph(self):
        """ Compute cross entropy loss between targets and predictions. Also compute the L2 error. """

        # Loss
        with tf.variable_scope('loss'):
            self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits_unrolled,
                                                                                         labels=self._targets_unrolled),
                                                 name='cross_entropy')
            self._loss = self._cross_entropy

        self._predictions_coords = [tf.mod(tf.argmax(self._logits_unrolled, dimension=1), self._image_size),
                                    tf.floordiv(tf.argmax(self._logits_unrolled, dimension=1), self._image_size)]
        self._predictions_coords = tf.stack(self._predictions_coords, axis=1)
        self._target_coords = [tf.mod(tf.argmax(self._targets_unrolled, dimension=1), self._image_size),
                               tf.floordiv(tf.argmax(self._targets_unrolled, dimension=1), self._image_size)]
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

        slices = tf.split(self.inputs, self.input_shape[-1], axis=3)
        # flat_images = tf.image.rgb_to_grayscale(tf.concat(slices[:3], axis=3))
        # flat_images = tf.expand_dims(
        #     tf.expand_dims(tf.expand_dims(1 / tf.reduce_max(flat_images, axis=[1, 2, 3]), axis=1), axis=2),
        #     axis=3) * flat_images
        # inputs_with_flat_images = tf.concat([flat_images, slices[-2], slices[-1]], axis=3,
        #                                     name='inputs_with_flat_images')
        inputs_with_flat_images = tf.concat(slices, axis=3, name='inputs_with_flat_images')
        input_visualization_summary = tf.summary.image('Inputs', inputs_with_flat_images, max_outputs=20)
        output_visualization_summary = tf.summary.image('Outputs', tf.expand_dims(
            tf.reshape(self.softmax, shape=[-1] + self.target_shape), dim=3), max_outputs=20)
        # target_visualization_summary = tf.summary.image('Targets', tf.expand_dims(self.targets, dim=3), max_outputs=20)
        image_summaries = tf.summary.merge([input_visualization_summary, output_visualization_summary])

        return scalar_summaries, image_summaries

    @property
    def inputs(self):
        """ An n-D float32 placeholder with shape `[batch_size, max_duration, input_size]`. """
        return self._inputs

    @property
    def lstm_init_state(self):
        """ The initial state of the LSTM. """
        return self._c_init, self._h_init

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
            history_mask = _create_history_mask(prediction_vertices, len(prediction_vertices), dataset.image_size)
            cursor_mask = _create_point_mask(cursor, dataset.image_size)
            state = np.concatenate([image, np.stack([history_mask, cursor_mask], axis=2)], axis=2)
            previous_states.append(state)
            # Feed this one state, but use the previous LSTM state.
            # Basically generate the RNN output one step at a time.
            softmax, pred_coords, (lstm_c, lstm_h) = sess.run(
                [est.softmax, est._predictions_coords, est.lstm_final_state],
                {est.inputs: np.expand_dims(state, axis=0), est._c_in: lstm_c,
                 est._h_in: lstm_h})
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
        state = np.concatenate([image, np.stack([history_mask, cursor_mask], axis=2)], axis=2)
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
