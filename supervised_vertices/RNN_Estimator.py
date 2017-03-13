from __future__ import division
import tensorflow as tf
import operator
from functools import reduce

image_size = 32  # TODO get rid of this and make it cleaner


class RNN_Estimator(object):
    def __init__(self, max_timesteps, init_scale=0.1):
        """ Create an RNN.

        Args:
            max_timesteps: An integer. The maximum number of timesteps expected from the RNN.
            init_scale: A float. All weight matrices will be initialized using
                a uniform distribution over [-init_scale, init_scale].
        """

        self.max_timesteps = max_timesteps
        self.input_shape = [32, 32, 3]
        self.target_shape = [32, 32]
        self.init_scale = init_scale

        ## Feed Vars
        # A Tensor of shape [None]
        self._seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        # A Tensor of shape [None, max_timesteps, ...]
        self._inputs = tf.placeholder(tf.float32, shape=[None, max_timesteps] + self.input_shape,
                                      name='inputs')
        # A Tensor of shape [None, max_timesteps, ...]
        self._targets = tf.placeholder(tf.float32, shape=[None, max_timesteps] + self.target_shape,
                                       name='targets')
        # A scalar
        self.iou = tf.placeholder(dtype=tf.float32, shape=[], name='iou')
        self.failures = tf.placeholder(dtype=tf.float32, shape=[], name='failure_rate')

        ## Fetch Vars
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._create_inference_graph()
            self._create_loss_graph()
            self._create_optimizer(initial_learning_rate=1e-1, num_steps_per_decay=20000,
                                   decay_rate=0.05, max_global_norm=1.0)
        with tf.variable_scope('train'):
            self._training_summaries, self._training_image_summaries, self._training_iou_summaries = self._build_summaries()
        with tf.variable_scope('valid'):
            self._validation_summaries, self._validation_image_summaries, self._validation_iou_summaries = self._build_summaries()

    def _create_inference_graph(self):
        """ Create a CNN that feeds an RNN. """

        self._batch_size = tf.shape(self.inputs, out_type=tf.int64)[0]

        self._inputs_unrolled = tf.reshape(self._inputs, shape=[-1] + self.input_shape)
        self._targets_unrolled = tf.reshape(self._targets, shape=[-1] + [reduce(operator.mul, self.target_shape)])

        with tf.variable_scope('input_cnn'):
            self._drop_rate = tf.placeholder_with_default(0.0, shape=[])

            with tf.variable_scope('conv1'):
                self._h_conv1 = tf.layers.conv2d(inputs=self._inputs_unrolled, filters=64, kernel_size=[5, 5],
                                                 padding='same', activation=tf.nn.relu)
                self._h_pool1 = tf.layers.max_pooling2d(inputs=self._h_conv1, pool_size=[2, 2], strides=2)

            with tf.variable_scope('conv2'):
                self._h_conv2 = tf.layers.conv2d(inputs=self._h_pool1, filters=128, kernel_size=[5, 5],
                                                 padding='same', activation=tf.nn.relu)
                self._h_pool2 = tf.layers.max_pooling2d(inputs=self._h_conv2, pool_size=[2, 2], strides=2)

            with tf.variable_scope('fc1'):
                fc_size = reduce(operator.mul, self._h_pool2.get_shape().as_list()[1:], 1)
                self._h_pool2_flat = tf.reshape(self._h_pool2, [-1, fc_size])
                self._h_fc1 = tf.layers.dense(inputs=self._h_pool2_flat, units=fc_size, activation=tf.nn.relu)
                self._h_fc1_drop = tf.layers.dropout(inputs=self._h_fc1, rate=self._drop_rate)

            with tf.variable_scope('fc2'):
                self._y_unrolled = tf.layers.dense(inputs=self._h_fc1_drop, units=image_size * image_size,
                                                   activation=tf.nn.relu)
                self._y = tf.reshape(self._y_unrolled, shape=[-1, self.max_timesteps] + [image_size * image_size])

        with tf.variable_scope('recurrent_network'):
            num_rnn_cells = 1024

            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_rnn_cells)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * 3)

            rnn_outputs, self._last_states = tf.nn.dynamic_rnn(cell, self._y, sequence_length=self._seq_length,
                                                               dtype=tf.float32)
            rnn_outputs_unrolled = tf.reshape(rnn_outputs, shape=[-1, num_rnn_cells])
            last_states_unrolled = tf.reshape(self._last_states, shape=[-1, num_rnn_cells])

        with tf.variable_scope('output_layer'):
            fc1 = tf.layers.dense(rnn_outputs_unrolled, units=4096,
                                  activation=tf.nn.relu)
            self._output_unrolled = tf.layers.dense(fc1,
                                                    units=reduce(operator.mul, self.target_shape),
                                                    activation=tf.nn.relu)

            self._output = tf.reshape(self._output_unrolled,
                                      shape=[-1, self.max_timesteps] + self.target_shape)

        self.softmax = tf.reshape(tf.nn.softmax(self.predictions), shape=[-1, image_size, image_size])

    def _create_loss_graph(self):
        """ Compute cross entropy loss between targets and predictions. Also compute the L2 error. """

        # Cross entropy loss
        with tf.variable_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self._output_unrolled,
                                                                  labels=self._targets_unrolled)
            self._loss = tf.reduce_mean(self.losses)

        self._predictions_coords = [tf.mod(tf.argmax(self._output_unrolled, dimension=1), image_size),
                                    tf.floordiv(tf.argmax(self._output_unrolled, dimension=1), image_size)]
        self._predictions_coords = tf.stack(self._predictions_coords, axis=1)
        self._target_coords = [tf.mod(tf.argmax(self._targets_unrolled, dimension=1), image_size),
                               tf.floordiv(tf.argmax(self._targets_unrolled, dimension=1), image_size)]
        self._target_coords = tf.stack(self._target_coords, axis=1)

        # Number of pixels correct
        with tf.variable_scope('accuracy'):
            x_correct, y_correct = tf.split(tf.equal(self._predictions_coords, self._target_coords), 2, axis=1)
            self._accuracy = tf.count_nonzero(tf.logical_and(x_correct, y_correct)) / (
                self._batch_size * self.max_timesteps)

        # L2 error
        with tf.variable_scope('error'):
            self._individual_error = tf.sqrt(
                tf.to_float(tf.reduce_sum((self._predictions_coords - self._target_coords) ** 2, 1)))
            self._error = tf.reduce_mean(self._individual_error)

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
        error_summary = tf.summary.scalar('error', self._error)
        scalar_summaries = tf.summary.merge(
            [learning_rate_summary, loss_summary, accuracy_summary, error_summary, grad_norm_summary])

        input_visualization_summary = tf.summary.image("Inputs", self._inputs_unrolled)
        output_visualization_summary = tf.summary.image("Outputs", tf.expand_dims(self.softmax, dim=3))
        image_summaries = tf.summary.merge(
            [input_visualization_summary, output_visualization_summary])

        failure_summary = tf.summary.scalar('failure_rate', self.failures)
        iou_summary = tf.summary.scalar('iou', self.iou)
        iou_summaries = tf.summary.merge([failure_summary, iou_summary])

        return scalar_summaries, image_summaries, iou_summaries

    @property
    def inputs(self):
        """ An n-D float32 placeholder with shape `[batch_size, max_duration, input_size]`. """
        return self._inputs

    @property
    def targets(self):
        """ An n-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets

    @property
    def sequence_length(self):
        """ A 1-D int32 placeholder with shape `[batch_size]`. """
        return self._seq_length

    @property
    def states(self):
        """ A 2-D float32 Tensor with shape `[batch_size * max_duration, hidden_layer_size]`. """
        return self._states

    @property
    def predictions(self):
        """ An n-D float32 Tensor with shape `[batch_size, max_duration, target_size]`. """
        return self._output

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


# TODO encapsulate this somewhere, hopefully entirely inside TensorFlow if possible
def evaluate_iou(sess, est, dataset):
    from supervised_vertices import generate, analyze
    import itertools
    import numpy as np
    from supervised_vertices.Dataset import _create_image, _create_history_mask, _create_point_mask, _create_shape_mask

    # IOU
    failed_shapes = 0
    num_iou = 0
    ious = []
    batch_size = 50
    for b in range(batch_size):
        vertices, ground_truth = dataset.data[b]

        image = _create_image(ground_truth)
        # Probably should optimize this with a numpy array and clever math. Use np.roll
        start_idx = np.random.randint(len(vertices))
        poly_verts = vertices[start_idx:] + vertices[:start_idx]
        cursor = poly_verts[0]
        verts_so_far = [cursor]

        predictions = []
        inputs = np.zeros([1, 5, 32, 32, 3])
        for t in itertools.count():
            history_mask = _create_history_mask(verts_so_far, len(verts_so_far), image_size)
            cursor_mask = _create_point_mask(cursor, image_size)
            state = np.stack([image, history_mask, cursor_mask], axis=2)

            inputs[0, t, ::] = state
            pred, pred_coords = sess.run([est._output_unrolled, est._predictions_coords],
                                         {est.sequence_length: np.ones([1]) * (t + 1), est.inputs: inputs})
            predictions.append(pred[t])

            cursor = tuple(reversed(pred_coords[t].tolist()))
            verts_so_far.append(cursor)

            distance = np.linalg.norm(np.array(poly_verts[0]) - np.array(cursor))
            if distance < 2:
                predicted_polygon = _create_shape_mask(verts_so_far, image_size)
                iou = analyze.calculate_iou(ground_truth, predicted_polygon)
                ious.append(iou)
                num_iou += 1
                break
            elif t + 1 >= 5:
                failed_shapes += 1
                # num_iou += 1
                # ious.append(0)
                break
    return failed_shapes / batch_size, sum(ious) / num_iou if num_iou > 0 else -1
