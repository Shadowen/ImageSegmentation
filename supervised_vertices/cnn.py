import numpy as np
import tensorflow as tf
from functools import reduce
import os
import shutil
import operator

# Make local imports work properly when running from terminal
import sys

sys.path.append('.')
from supervised_vertices.analyze import evaluate_iou
from supervised_vertices.Dataset import get_train_and_valid_datasets


class CNN_Estimator():
    def __init__(self, image_size):
        self.image_size = image_size

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 5])
        self.targets = tf.placeholder(tf.int32, shape=[None, self.image_size, self.image_size])
        self.targets_flat = tf.reshape(self.targets, shape=[-1, self.image_size * self.image_size])

        with tf.variable_scope('input_cnn'):
            self.drop_rate = tf.placeholder_with_default(0.0, shape=[])

            with tf.variable_scope('conv1'):
                self._h_conv1 = tf.layers.conv2d(inputs=self.x, filters=16, kernel_size=[5, 5],
                                                 padding='same', activation=tf.nn.relu)
                self._h_pool1 = tf.layers.max_pooling2d(inputs=self._h_conv1, pool_size=[2, 2], strides=2)

            with tf.variable_scope('conv2'):
                self._h_conv2 = tf.layers.conv2d(inputs=self._h_pool1, filters=32, kernel_size=[5, 5],
                                                 padding='same', activation=tf.nn.relu)
                self._h_pool2 = tf.layers.max_pooling2d(inputs=self._h_conv2, pool_size=[2, 2], strides=2)

            with tf.variable_scope('fc1'):
                fc_size = reduce(operator.mul, self._h_pool2.get_shape().as_list()[1:], 1)
                self._h_pool2_flat = tf.reshape(self._h_pool2, [-1, fc_size])
                self._h_fc1 = tf.layers.dense(inputs=self._h_pool2_flat, units=512, activation=tf.nn.relu)
                self._h_fc1_drop = tf.layers.dropout(inputs=self._h_fc1, rate=self.drop_rate)

            with tf.variable_scope('fc2'):
                self.y_flat = tf.layers.dense(inputs=self._h_fc1_drop, units=self.image_size * self.image_size)
                self.y = tf.reshape(self.y_flat, shape=[-1, self.image_size, self.image_size])

        self._create_loss_graph()

    def _create_loss_graph(self):
        # Calculate the loss
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_flat, labels=self.targets_flat)
        self.loss_op = tf.reduce_mean(self.losses)
        self.loss_summary = tf.summary.scalar('cross_entropy', self.loss_op)
        self.learning_rate = tf.maximum(
            tf.train.exponential_decay(0.001, tf.contrib.framework.get_global_step(), 1000, 0.9, staircase=True),
            0.0001)
        self.learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)

        # Accuracy
        self.y_coords = [tf.mod(tf.argmax(self.y_flat, dimension=1), self.image_size),
                         tf.floordiv(tf.argmax(self.y_flat, dimension=1), self.image_size)]
        self.y_coords = tf.stack(self.y_coords, axis=1)
        self.target_coords = [tf.mod(tf.argmax(self.targets_flat, dimension=1), self.image_size),
                              tf.floordiv(tf.argmax(self.targets_flat, dimension=1), self.image_size)]
        self.target_coords = tf.stack(self.target_coords, axis=1)
        self.individual_accuracy = tf.sqrt(tf.to_float(tf.reduce_sum((self.y_coords - self.target_coords) ** 2, 1)))
        self.error_op = tf.reduce_mean(self.individual_accuracy)
        self.error_summary = tf.summary.scalar('error', self.error_op)

        # IOU
        self.iou = tf.placeholder(dtype=tf.float32, shape=[])
        self.iou_summary = tf.summary.scalar('IOU', self.iou)
        self.ious = tf.placeholder(dtype=tf.float32, shape=[None])
        self.iou_histogram = tf.summary.histogram('IOU_histogram', self.ious)
        self.failed_shapes = tf.placeholder(dtype=tf.float32, shape=[])
        self.failed_shapes_summary = tf.summary.scalar('Failed_shapes', self.failed_shapes)


if __name__ == '__main__':

    # training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
    # training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data', local=False)
    training_set, validation_set = get_train_and_valid_datasets('/ais/gobi4/wiki/polyrnn/data/shapes_texture', local=False)

    with tf.Session() as sess:
        global_step_op = tf.Variable(0, name='global_step', trainable=False)
        increment_global_step_op = tf.assign(global_step_op, global_step_op + 1)
        est = CNN_Estimator(image_size=224)
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
        logdir = '/ais/gobi5/polyRL/cnn'
        # logdir = '/tmp/cnn'
        load_from = ''
        latest_checkpoint = tf.train.latest_checkpoint(load_from)
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
            os.makedirs(logdir)
        sess.run(tf.global_variables_initializer())
        if latest_checkpoint:
            print("Loading model checkpoint: {}".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(logdir + '/valid')

        batch_size = 50
        for iteration in range(10000):
            batch_x, batch_t = training_set.get_batch_for_cnn(batch_size)
            loss, _, _, learning_rate_summary, loss_summary = sess.run(
                [est.loss_op, est.train_op, increment_global_step_op, est.learning_rate_summary, est.loss_summary],
                {est.x: batch_x, est.targets: batch_t, est.drop_rate: 0.3})
            if iteration % 50 == 0:
                print('Iteration {}\t Loss={}'.format(iteration, loss))

            if iteration % 100 == 0:
                train_writer.add_summary(learning_rate_summary, iteration)
                train_writer.add_summary(loss_summary, iteration)
                # Validation set
                valid_x, valid_t = validation_set.get_batch_for_cnn(batch_size=batch_size)
                [valid_writer.add_summary(s, iteration) for s in sess.run([est.loss_summary, est.error_summary],
                                                                          {est.x: valid_x, est.targets: valid_t,
                                                                           est.drop_rate: 0})]

            if iteration % 1000 == 0:
                ious, failed_shapes = evaluate_iou(validation_set, sess, est)
                valid_writer.add_summary(sess.run(est.iou_histogram, {est.ious: np.array(ious)}), iteration)

                valid_writer.add_summary(sess.run(est.iou_summary, {est.iou: sum(ious) / len(validation_set)}),
                                         iteration)
                valid_writer.add_summary(
                    sess.run(est.failed_shapes_summary, {est.failed_shapes: failed_shapes / len(validation_set)}),
                    iteration)

                saver.save(sess, logdir + '/model')
