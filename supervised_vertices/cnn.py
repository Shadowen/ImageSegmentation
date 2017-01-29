import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from supervised_vertices import generate


def make_variables(shape):
    initial_weights = tf.truncated_normal(shape, stddev=0.1)
    weight_var = tf.Variable(name='weight', dtype=tf.float32, initial_value=initial_weights)
    initial_biases = tf.constant(0.1, shape=[shape[-1]])
    bias_var = tf.Variable(name='bias', dtype=tf.float32, initial_value=initial_biases)

    return weight_var, bias_var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Estimator():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.keep_prob = tf.placeholder_with_default(1.0, [])
        self.targets = tf.placeholder(tf.int32, shape=[None])
        self.targets_onehot = tf.one_hot(self.targets, 32 * 32)

        with tf.variable_scope('conv1'):
            self.W_conv1, self.b_conv1 = make_variables([5, 5, 3, 32])
            self.h_conv1 = tf.nn.relu(conv2d(self.x, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

        with tf.variable_scope('conv2'):
            self.W_conv2, self.b_conv2 = make_variables([5, 5, 32, 64])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

        with tf.variable_scope('fc1'):
            self.W_fc1, self.b_fc1 = make_variables([8 * 8 * 64, 1024])
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 8 * 8 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        with tf.variable_scope('fc2'):
            self.W_fc2, self.b_fc2 = make_variables([1024, 32 * 32])
            self.y_flat = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
            self.y = tf.reshape(self.y_flat, shape=[-1, 32, 32])

            self.prediction = tf.argmax(self.y_flat, dimension=1)

        # Calculate the loss
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_flat, self.targets)
        self.loss_op = tf.reduce_mean(self.losses)
        self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss_op)


def display_samples(x, truth, prediction):
    print("showing")
    fig, ax = plt.subplots(nrows=len(x), ncols=5)
    [ax[0][e].set_title(t) for e, t in enumerate(['Image', 'History', 'Cursor', 'Truth', 'Prediction'])]
    for i in range(len(x)):
        xi = np.moveaxis(x[i], 2, 0)
        for e in range(3):
            ax[i][e].axis('off')
            ax[i][e].imshow(xi[e], cmap='gray', interpolation='nearest')
        ax[i][3].axis('off')
        ax[i][3].imshow(generate.create_point_mask((truth[i] // 32, truth[i] % 32), image_size=32), cmap='gray',
            interpolation='nearest')
        ax[i][4].axis('off')
        ax[i][4].imshow(generate.create_point_mask((prediction[i] // 32, prediction[i] % 32), image_size=32),
            cmap='gray', interpolation='nearest')
    plt.show()


data = np.load('dataset_polygons.npy')
print('{} polygons loaded.'.format(data.shape[0]))
valid_size = 50
training_data = data[valid_size:]
validation_data = data[:valid_size]
print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))
est = Estimator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 50
    for iteration in range(10001):
        print('Beginning iteration {}'.format(iteration), end='\t')

        batch_indices = np.random.choice(training_data.shape[0], batch_size, replace=False)
        batch_x, batch_t = zip(
            *[generate.create_training_sample(32, vertices, truth) for vertices, truth in training_data[batch_indices]])
        loss, _ = sess.run([est.loss_op, est.train_op], {est.x:batch_x, est.targets:batch_t, est.keep_prob:0.7})
        print(loss, end='\t')

        valid_x, valid_t = zip(
            *[generate.create_training_sample(32, vertices, truth) for vertices, truth in validation_data])
        valid_predictions, valid_loss = sess.run([est.prediction, est.loss_op], {est.x:valid_x, est.targets:valid_t})
        print(valid_loss)

        # if iteration == 10000:
        #     print(valid_predictions == valid_t)
        #     display_samples(valid_x[:10], valid_t[:10], valid_predictions[:10])
