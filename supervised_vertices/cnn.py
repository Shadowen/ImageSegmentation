import numpy as np
import tensorflow as tf
from supervised_vertices import generate
from functools import reduce
import os
from supervised_vertices import analyze

image_size = 32


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
        self.x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 4])
        self.keep_prob = tf.placeholder_with_default(1.0, [])
        self.targets = tf.placeholder(tf.int32, shape=[None, 2])
        self.targets_flat = self.targets[:, 0] * image_size + self.targets[:, 1]
        self.targets_onehot = tf.one_hot(self.targets_flat, image_size * image_size)

        with tf.variable_scope('conv1'):
            self.W_conv1, self.b_conv1 = make_variables([5, 5, self.x.get_shape().as_list()[-1], image_size])
            self.h_conv1 = tf.nn.relu(conv2d(self.x, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

        with tf.variable_scope('conv2'):
            self.W_conv2, self.b_conv2 = make_variables([5, 5, image_size, 64])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

        with tf.variable_scope('fc1'):
            fc_size = reduce(lambda x, y: x * y, self.h_pool2.get_shape().as_list()[1:], 1)
            self.W_fc1, self.b_fc1 = make_variables([fc_size, 1024])
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, fc_size])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        with tf.variable_scope('fc2'):
            self.W_fc2, self.b_fc2 = make_variables([1024, image_size * image_size])
            self.y_flat = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
            self.y = tf.reshape(self.y_flat, shape=[-1, image_size, image_size])
            self.softmax = tf.reshape(tf.nn.softmax(self.y_flat), shape=[-1, image_size, image_size])

        # Calculate the loss
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_flat, self.targets_flat)
        self.loss_op = tf.reduce_mean(self.losses)
        self.loss_summary = tf.summary.scalar('cross_entropy', self.loss_op)
        self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss_op)

        # Accuracy
        self.y_coords = [tf.mod(tf.argmax(self.y_flat, dimension=1), image_size),
                         tf.floordiv(tf.argmax(self.y_flat, dimension=1), image_size)]
        self.y_coords = tf.stack(self.y_coords, axis=1)
        self.target_coords = [tf.to_int64(tf.mod(self.targets_flat, image_size)),
                              tf.to_int64(tf.floordiv(self.targets_flat, image_size))]
        self.target_coords = tf.stack(self.target_coords, axis=1)
        self.individual_accuracy = tf.sqrt(tf.to_float(tf.reduce_sum((self.y_coords - self.target_coords) ** 2, 1)))
        self.accuracy_op = tf.reduce_mean(self.individual_accuracy)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy_op)

        # IOU Histogram
        # self.ious = tf.Variable(0)
        # self.iou_histogram = tf.histogram_summary('IOU_histogram', self.ious)


if __name__ == '__main__':
    data = np.load('dataset_polygons.npy')
    print('{} polygons loaded.'.format(data.shape[0]))
    valid_size = data.shape[0] // 10
    training_data = data[valid_size:]
    validation_data = data[:valid_size]
    del data  # Make sure we don't contaminate the training set
    print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))

    with tf.Session() as sess:
        est = Estimator()
        saver = tf.train.Saver()
        if not os.path.exists(('results')):
            os.makedirs('results/')
        train_writer = tf.train.SummaryWriter('./results/train', sess.graph)
        valid_writer = tf.train.SummaryWriter('./results/valid')

        sess.run(tf.global_variables_initializer())

        batch_size = 50
        for iteration in range(10000):
            if iteration % 50 == 0:
                print('Iteration {}'.format(iteration))

            batch_indices = np.random.choice(training_data.shape[0], batch_size, replace=False)
            batch_x, batch_t = zip(*[generate.create_training_sample(image_size, vertices, truth) for vertices, truth in
                                     training_data[batch_indices]])
            sess.run(est.train_op, {est.x: batch_x, est.targets: batch_t, est.keep_prob: 0.7})

            if iteration % 100 == 0:
                [train_writer.add_summary(s, iteration) for s in sess.run([est.loss_summary, est.accuracy_summary],
                                                                          {est.x: batch_x, est.targets: batch_t,
                                                                           est.keep_prob: 1.0})]

                # Validation set
                valid_x, valid_t = zip(
                    *[generate.create_training_sample(image_size, vertices, truth) for vertices, truth in
                      validation_data])
                [valid_writer.add_summary(s, iteration) for s in sess.run([est.loss_summary, est.accuracy_summary],
                                                                          {est.x: valid_x, est.targets: valid_t,
                                                                           est.keep_prob: 1.0})]

            if iteration % 1000 == 0:
                ious, failed_shapes = analyze.evaluate_iou(validation_data, sess, est)
                # valid_writer.add_summary(sess.run(est.iou_histogram, {est.ious: np.array(ious)}), iteration)
                valid_writer.add_summary(sess.run(tf.scalar_summary('IOU', sum(ious) / len(ious))), iteration)
                valid_writer.add_summary(sess.run(tf.scalar_summary('Failed shapes', failed_shapes / valid_size)),
                                         iteration)

        saver.save(sess, './results/model.ckpt')
