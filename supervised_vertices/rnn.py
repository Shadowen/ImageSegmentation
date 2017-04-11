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
    lstm_c, lstm_h = sess.run(model.lstm_init_state)
    for step in range(num_optimization_steps):
        print('\rStep %d.' % (step + 1), end='')
        durations, images, cursors, histories, targets, _ = training_set.get_sample_for_rnn()
        training_summaries, training_image_summaries, _ = sess.run(
            [model.training_summaries, model.training_image_summaries, model.train_op],
            {model.image_input: images, model.cursor_mask: cursors, model.history_mask: histories,
             model.targets: targets, model._c_in: lstm_c, model._h_in: lstm_h})
        summary_writer.add_summary(training_summaries, global_step=step)
        summary_writer.add_summary(training_image_summaries, global_step=step)

        # Validate
        if step % 100 == 0:
            durations, images, cursors, histories, targets, _ = validation_set.get_sample_for_rnn()
            validation_summaries, validation_image_summaries = sess.run(
                [model.validation_summaries, model.validation_image_summaries],
                {model.image_input: images, model.cursor_mask: cursors, model.history_mask: histories,
                 model.targets: targets, model._c_in: lstm_c, model._h_in: lstm_h})
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
    logdir = 'alexnet_convlstm_3/'  # Where to save the checkpoints and output files
    do_train = True  # Should we run the training steps?
    restart_training = True  # Turn this on to delete any existing directory
    is_local = True  # Turn this on for training on the CS cluster
    use_pretrained = True  # Use pretrained weights?

    # Parameters
    image_size = 224
    max_timesteps = 10
    num_steps = 100000

    model = RNN_Estimator(image_size=image_size, use_pretrained=use_pretrained)
    if is_local:
        print('Loading dataset from numpy archive...')
        training_set, validation_set = get_train_and_valid_datasets('polygons_5-sided.npy',
                                                                    image_size=image_size,
                                                                    prediction_size=model.prediction_size,
                                                                    is_local=True)
        print('Done!')
    else:
        logdir = '/ais/gobi5/polyRL/' + logdir
        print('Loading dataset from JSON...')
        training_set, validation_set = get_train_and_valid_datasets('/ais/gobi4/wiki/polyrnn/data/shapes_texture/',
                                                                    image_size=image_size,
                                                                    prediction_size=model.prediction_size,
                                                                    is_local=False)
        # logdir = '/home/wesley/data/' + logdir
        # training_set, validation_set = get_train_and_valid_datasets('/home/wesley/data/',
        #                                                             image_size=image_size,
        #                                                             input_channels=input_channels,
        #                                                             prediction_size=prediction_size,
        #                                                             is_local=False)
        print('Done!')
    with tf.Session() as sess:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)

        if restart_training and os.path.exists(logdir):
            print('Resetting training directory at `{}`.'.format(logdir))
            shutil.rmtree(logdir)
            os.makedirs(logdir)
        if do_train:
            print('Training started!')
            train(sess, model, training_set, validation_set, max_timesteps=max_timesteps,
                  num_optimization_steps=num_steps,
                  logdir=logdir)
            print('Training complete!')
        elif tf.train.latest_checkpoint(logdir):
            print('Restoring model from `' + logdir + '/model.ckpt`...', end=' ')
            saver.restore(sess, save_path=logdir + '/model.ckpt')
            print('Success!')

            print('Evaluating IOU...', end=' ')
            avg_failed_shapes, avg_iou, failed_images = evaluate_iou(sess, model, validation_set,
                                                                     max_timesteps=max_timesteps, batch_size=20,
                                                                     logdir=logdir)
            print('Average failed shapes={}\]t Average IOU={}'.format(avg_failed_shapes, avg_iou))
