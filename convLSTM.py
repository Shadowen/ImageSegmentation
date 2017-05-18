# https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py
import tensorflow as tf


class ConvLSTMCell(tf.contrib.rnn.RNNCell):
    """A LSTM cell with convolutions instead of multiplications.
    Reference:
      Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, shape, filters, kernel, initializer=None, forget_bias=1.0):
        self._kernel = kernel
        self._filters = filters
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._size = tf.TensorShape(shape + [self._filters])

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size

    def __call__(self, x, h, scope=None):
        with tf.variable_scope(scope or self.__class__.__name__):
            previous_memory, previous_output = h

            channels = x.shape[-1].value
            gates = 4 * self._filters
            W_h = tf.get_variable('hidden_to_state_kernel', self._kernel + [self._filters, gates],
                                  initializer=self._initializer)
            W_x = tf.get_variable('input_to_state_kernel', self._kernel + [channels, gates],
                                  initializer=self._initializer)
            y = tf.nn.convolution(previous_output, W_h, 'SAME') + \
                tf.nn.convolution(x, W_x, 'SAME') + \
                tf.get_variable('bias', [gates], initializer=tf.constant_initializer(0.0))

            input_contribution, input_gate, forget_gate, output_gate = tf.split(y, 4, axis=3)

            cell_state = tf.sigmoid(forget_gate) * previous_memory + tf.sigmoid(input_gate) * tf.tanh(input_contribution)
            output = tf.sigmoid(output_gate) * tf.tanh(cell_state)

            return output, tf.contrib.rnn.LSTMStateTuple(cell_state, output)
