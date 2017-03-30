import sys

sys.path.append('.')
from supervised_vertices.RNN_Estimator import *
import tensorflow as tf
import os
import shutil
from supervised_vertices.Dataset import get_train_and_valid_datasets


def train(sess, model, training_set, validation_set, num_optimization_steps, logdir='./logs'):
    """ Train.

    Args:
        sess: A Session.
        model: A Model.
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """

    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(num_optimization_steps):
        print('\rStep %d.' % (step + 1), end='')
        durations, inputs, targets, _ = training_set.get_sample_for_rnn(max_timesteps=5)
        training_summaries, training_image_summaries, _ = sess.run(
            [model.training_summaries, model.training_image_summaries, model.train_op],
            {model.inputs: inputs,
             model.targets: targets})
        summary_writer.add_summary(training_summaries, global_step=step)
        summary_writer.add_summary(training_image_summaries, global_step=step)

        # Validate
        if step % 100 == 0:
            durations, inputs, targets, _ = validation_set.get_sample_for_rnn(max_timesteps=5)
            validation_summaries, validation_image_summaries = sess.run(
                [model.validation_summaries, model.validation_image_summaries],
                {model.inputs: inputs, model.targets: targets})
            summary_writer.add_summary(validation_summaries, global_step=step)
            summary_writer.add_summary(validation_image_summaries, global_step=step)

        # IOU
        if step % 1000 == 0:
            avg_failed_shapes, avg_iou = evaluate_iou(sess, model, validation_set, batch_size=100)
            iou_summaries = tf.Summary()
            iou_summaries.value.add(tag='valid/failed_shapes', simple_value=avg_failed_shapes)
            iou_summaries.value.add(tag='valid/iou', simple_value=avg_iou)
            summary_writer.add_summary(iou_summaries, step)

        # Save
        if step % 1000 == 0:
            saver.save(sess, logdir + '/model.ckpt')


if __name__ == '__main__':
    logdir = 'rnn'
    do_train = False
    restart_training = False

    training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
    model = RNN_Estimator()
    with tf.Session() as sess:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)

        if restart_training and os.path.exists(logdir):
            shutil.rmtree(logdir)
            os.makedirs(logdir)
        if do_train:
            train(sess, model, training_set, validation_set, num_optimization_steps=100000, logdir=logdir)
        elif tf.train.latest_checkpoint(logdir):
            saver.restore(sess, save_path=logdir + '/model.ckpt')
