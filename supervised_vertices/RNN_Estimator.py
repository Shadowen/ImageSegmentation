import tensorflow as tf
from supervised_vertices.cnn import *
import operator
import io


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

        ## Fetch Vars
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope('model', initializer=initializer):
            self._create_inference_graph()
            self._create_loss_graph()
            self._create_optimizer(initial_learning_rate=1e-1, num_steps_per_decay=1000,
                                   decay_rate=0.1, max_global_norm=1.0)
            self._build_summaries()

    def _create_inference_graph(self):
        """ Create a CNN that feeds an RNN. """

        self._inputs_unrolled = tf.reshape(self._inputs, shape=[-1] + self.input_shape)
        self._targets_flat = tf.reshape(self._targets, shape=[-1] + [reduce(operator.mul, self.target_shape)])

        with tf.variable_scope('input_cnn'):
            self.keep_prob = tf.placeholder_with_default(1.0, [])

            with tf.variable_scope('conv1'):
                self._W_conv1, self._b_conv1 = make_variables(
                    [5, 5, self._inputs_unrolled.get_shape().as_list()[-1], 32])
                self._h_conv1 = tf.nn.relu(conv2d(self._inputs_unrolled, self._W_conv1) + self._b_conv1)
                self._h_pool1 = max_pool_2x2(self._h_conv1)

            with tf.variable_scope('conv2'):
                self._W_conv2, self._b_conv2 = make_variables([5, 5, 32, 64])
                self._h_conv2 = tf.nn.relu(conv2d(self._h_pool1, self._W_conv2) + self._b_conv2)
                self._h_pool2 = max_pool_2x2(self._h_conv2)

            with tf.variable_scope('fc1'):
                fc_size = reduce(operator.mul, self._h_pool2.get_shape().as_list()[1:], 1)
                self._W_fc1, self._b_fc1 = make_variables([fc_size, 1024])
                self._h_pool2_flat = tf.reshape(self._h_pool2, [-1, fc_size])
                self._h_fc1 = tf.nn.relu(tf.matmul(self._h_pool2_flat, self._W_fc1) + self._b_fc1)
                self._h_fc1_drop = tf.nn.dropout(self._h_fc1, self.keep_prob)

            with tf.variable_scope('fc2'):
                self._W_fc2, self.b_fc2 = make_variables([1024, image_size * image_size])
                self._y_unrolled = tf.matmul(self._h_fc1_drop, self._W_fc2) + self.b_fc2

            self._y = tf.reshape(self._y_unrolled, shape=[-1, self.max_timesteps] + [image_size * image_size])

        with tf.variable_scope('recurrent_network'):
            num_rnn_cells = 1024
            cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_rnn_cells)

            outputs, last_states = tf.nn.dynamic_rnn(cell, self._y, sequence_length=self._seq_length,
                                                     dtype=tf.float32)
            self._outputs = tf.reshape(outputs, shape=[-1, num_rnn_cells])
            last_states = tf.reshape(last_states, shape=[-1, num_rnn_cells])

            hidden_layer_size = [reduce(operator.mul, self.target_shape)]
            W_pred = tf.get_variable('W_pred',
                                     shape=[num_rnn_cells] + hidden_layer_size)
            b_pred = tf.get_variable('b_pred', shape=hidden_layer_size,
                                     initializer=tf.constant_initializer(0.0))

            self._predictions_unrolled = tf.add(tf.matmul(self._outputs, W_pred), b_pred, name='predictions')
            self._predictions = tf.reshape(self._predictions_unrolled,
                                           shape=[-1, self.max_timesteps] + self.target_shape)

        self.softmax = tf.reshape(tf.nn.softmax(self.predictions), shape=[-1, image_size, image_size])

    def _create_loss_graph(self):
        """ Compute cross entropy loss between targets and predictions. Also compute the L2 error. """

        # Cross entropy loss
        with tf.variable_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self._outputs, self._targets_flat)
            self._loss = tf.reduce_mean(self.losses)

        # L2 error
        with tf.variable_scope('error'):
            self._y_coords = [tf.mod(tf.argmax(self._predictions_unrolled, dimension=1), image_size),
                              tf.floordiv(tf.argmax(self._predictions_unrolled, dimension=1), image_size)]
            self._y_coords = tf.stack(self._y_coords, axis=1)
            self._target_coords = [tf.mod(tf.argmax(self._targets_flat, dimension=1), image_size),
                                   tf.floordiv(tf.argmax(self._targets_flat, dimension=1), image_size)]
            self._target_coords = tf.stack(self._target_coords, axis=1)
            self._individual_error = tf.sqrt(tf.to_float(tf.reduce_sum((self._y_coords - self._target_coords) ** 2, 1)))
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
        grads = tf.gradients(self.loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self._learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, num_steps_per_decay,
                                                         decay_rate,
                                                         staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)

    def _build_summaries(self):
        """Creates summary operations from existing Tensors"""
        self._learning_rate_summary = tf.summary.scalar('learning_rate', self._learning_rate)
        self._loss_summary = tf.summary.scalar('loss', self._loss)
        self._error_summary = tf.summary.scalar('error', self._error)
        self._training_summaries = tf.summary.merge(
            [self._learning_rate_summary, self._loss_summary, self._error_summary])

        self._input_visualization_summary = tf.summary.image("Inputs", self._inputs_unrolled)
        self._output_visualization_summary = tf.summary.image("Outputs", tf.expand_dims(self.softmax, dim=3))
        self._image_summaries = tf.summary.merge(
            [self._input_visualization_summary, self._output_visualization_summary])

    @property
    def training_summaries(self):
        return self._training_summaries

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
        return self._predictions

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
    def image_summaries(self):
        return self._image_summaries
