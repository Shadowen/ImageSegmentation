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

    summary_writer = tf.train.SummaryWriter(logdir=logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(num_optimization_steps):
        durations, inputs, targets = training_set.get_batch(batch_size=50, max_timesteps=5)

        training_summaries, _ = sess.run([model.training_summaries, model.train_op],
                                         {model.sequence_length: durations, model.inputs: inputs,
                                          model.targets: targets})
        summary_writer.add_summary(training_summaries, global_step=step)

        image_summaries = sess.run(model.image_summaries, {model.sequence_length: durations, model.inputs: inputs})
        summary_writer.add_summary(image_summaries, global_step=step)
        print('\rStep %d.' % (step + 1), end='')


if __name__ == '__main__':
    training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
    model = RNN_Estimator(max_timesteps=5, init_scale=0.1)
    with tf.Session() as sess:
        saver = tf.train.Saver()

    train(sess, model, training_set, validation_set, num_optimization_steps=100000,
          logdir='./logdir')

    saver.save(sess, './results/model.ckpt')
