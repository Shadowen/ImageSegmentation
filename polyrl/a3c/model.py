import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from convLSTM import ConvLSTMCell


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.layers.conv2d(inputs=x, filters=32, name="l{}".format(i + 1), kernel_size=(3, 3), padding='same',
                                 activation=tf.nn.relu)

        image_size = 32
        size = 32
        lstm = ConvLSTMCell(height=image_size, width=image_size, filters=size, kernel=[3, 3])
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, x, initial_state=state_in, sequence_length=step_size,
                                                     time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(expand(lstm_outputs, height=image_size, width=image_size,
                              filters=size), shape=[-1, image_size ** 2 * size])
        self.logits = tf.layers.dense(inputs=x, units=ac_space, activation=None, name='action')
        self.vf = tf.reshape(tf.layers.dense(inputs=x, units=1, name='value'), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        """ Gets the initial state of the LSTM """
        return self.state_init

    def act(self, ob, c, h):
        """
        Sample an action from the policy. Also returns the value of the current state.
        :param ob: observation
        :param c: initial LSTM state c
        :param h:  initial LSTM state h
        :return: (action, value, *state_out)
        action - action sampled from multinomial distribution
        value - value function at this state
        *state_out - [lstm_c[:1, :], lstm_h[:1, :]] all LSTM intermediate states
        """
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        """ Calculate the estimated value of a given state. """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
