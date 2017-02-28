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
        optimizer: An Optimizer.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(num_optimization_steps):
        print('\rStep %d.' % (step + 1), end='')
        durations, inputs, targets = training_set.get_batch(batch_size=50, max_timesteps=5)
        training_summaries, training_image_summaries, _ = sess.run(
            [model.training_summaries, model.training_image_summaries, model.train_op],
            {model.sequence_length: durations, model.inputs: inputs,
             model.targets: targets})
        summary_writer.add_summary(training_summaries, global_step=step)
        summary_writer.add_summary(training_image_summaries, global_step=step)

        # Validate
        if step % 100 == 0:
            durations, inputs, targets = validation_set.get_batch(batch_size=50, max_timesteps=5)
            validation_summaries, validation_image_summaries = sess.run(
                [model.validation_summaries, model.validation_image_summaries],
                {model.sequence_length: durations, model.inputs: inputs,
                 model.targets: targets})
            summary_writer.add_summary(validation_summaries, global_step=step)
            summary_writer.add_summary(validation_image_summaries, global_step=step)

        # Save
        if step % 1000 == 0:
            saver.save(sess, logdir + '/model.ckpt')


if __name__ == '__main__':
    training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
    model = RNN_Estimator(max_timesteps=5, init_scale=0.1)
    with tf.Session() as sess:
        logdir = 'rnn_4'
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)

        do_train = True
        if do_train:
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.makedirs(logdir)
            train(sess, model, training_set, validation_set, num_optimization_steps=100000, logdir=logdir)

        saver.restore(sess, save_path=logdir + '/model.ckpt')
