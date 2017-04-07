import sys

sys.path.append('.')
from supervised_vertices.RNN_Estimator import *
import tensorflow as tf
import os
import shutil
from supervised_vertices.Dataset import get_train_and_valid_datasets


def train(sess, model, training_set, validation_set, max_timesteps, num_optimization_steps, logdir='./logs'):
    """ Train.

    Args:
        sess: A Session.
        model: A Model.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """

    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(num_optimization_steps):
        print('\rStep %d.' % (step + 1), end='')
        durations, inputs, targets, _ = training_set.get_sample_for_rnn()
        training_summaries, training_image_summaries, _ = sess.run(
            [model.training_summaries, model.training_image_summaries, model.train_op],
            {model.inputs: inputs,
             model.targets: targets})
        summary_writer.add_summary(training_summaries, global_step=step)
        summary_writer.add_summary(training_image_summaries, global_step=step)

        # Validate
        if step % 100 == 0:
            durations, inputs, targets, _ = validation_set.get_sample_for_rnn()
            validation_summaries, validation_image_summaries = sess.run(
                [model.validation_summaries, model.validation_image_summaries],
                {model.inputs: inputs, model.targets: targets})
            summary_writer.add_summary(validation_summaries, global_step=step)
            summary_writer.add_summary(validation_image_summaries, global_step=step)

        # IOU
        if step % 1000 == 0:
            avg_failed_shapes, avg_iou, failed_images = evaluate_iou(sess, model, validation_set,
                                                                     batch_size=min(100, len(validation_set)),
                                                                     max_timesteps=max_timesteps)
            iou_summaries = tf.Summary()
            iou_summaries.value.add(tag='valid/failed_shapes', simple_value=avg_failed_shapes)
            iou_summaries.value.add(tag='valid/iou', simple_value=avg_iou)
            summary_writer.add_summary(iou_summaries, step)

        # Save
        if step % 1000 == 0:
            saver.save(sess, logdir + '/model.ckpt')


if __name__ == '__main__':
    # Settings
    logdir = '/data/convlstm_4_sided'  # Where to save the checkpoints and output files
    do_train = True  # Should we run the training steps?
    restart_training = True  # Turn this on to delete any existing directory
    is_local = True  # Turn this on for training on the CS cluster

    # Parameters
    image_size = 32
    prediction_size = 32
    max_timesteps = 10
    num_steps = 30000

    if is_local:
        input_channels = 3
        print('Loading data from numpy archive...')
        training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy',
                                                                    image_size=image_size,
                                                                    input_channels=input_channels,
                                                                    prediction_size=prediction_size,
                                                                    is_local=True)
        print('Done!')
    else:
        logdir = '/ais/gobi5/polyRL/' + logdir
        input_channels = 5
        print('Loading data from JSON...')
        training_set, validation_set = get_train_and_valid_datasets('/ais/gobi4/wiki/polyrnn/data/shapes_texture/',
                                                                    image_size=image_size,
                                                                    input_channels=input_channels,
                                                                    prediction_size=prediction_size,
                                                                    is_local=False)
        # logdir = '/home/wesley/data/' + logdir
        # training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data/',
        #                                                             image_size=image_size,
        #                                                             input_channels=input_channels,
        #                                                             prediction_size=prediction_size,
        #                                                             is_local=False)
        print('Done!')
    model = RNN_Estimator(image_size=image_size, input_channels=input_channels, prediction_size=prediction_size)
    with tf.Session() as sess:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)

        if restart_training and os.path.exists(logdir):
            print('Resetting training directory at `{}`.'.format(logdir))
            shutil.rmtree(logdir)
            os.makedirs(logdir)
        if do_train:
            print('Training started!')
            train(sess, model, training_set, validation_set, max_timesteps=max_timesteps, num_optimization_steps=num_steps,
                  logdir=logdir)
            print('Training complete!')
        elif tf.train.latest_checkpoint(logdir):
            print('Restoring model from `' + logdir + '/model.ckpt`...', end=' ')
            saver.restore(sess, save_path=logdir + '/model.ckpt')
            print('Success!')

            print('Evaluating IOU...', end=' ')
            avg_failed_shapes, avg_iou, failed_images = evaluate_iou(sess, model, validation_set,
                                                                     max_timesteps=max_timesteps, batch_size=None,
                                                                     logdir=logdir)
            print('Average failed shapes={}\]t Average IOU={}'.format(avg_failed_shapes, avg_iou))
