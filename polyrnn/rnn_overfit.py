import numpy as np
import tensorflow as tf

from polyrnn.Dataset import get_train_and_valid_datasets

image_size = 28
prediction_size = 28
max_timesteps = 5
history_length = 1

global_step = tf.Variable(0, name='global_step', trainable=False)
train_data, valid_data = get_train_and_valid_datasets('/home/wesley/data/tiny-polygons', max_timesteps=max_timesteps,
                                                      image_size=image_size, prediction_size=prediction_size,
                                                      history_length=history_length, is_local=True, load_max_images=1)

image_pl = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='image')  # [batch, x, y, c]
batch_size = tf.shape(image_pl)[0]
image_flat = tf.reshape(image_pl, shape=[-1, image_size * image_size * 3])  # [batch, x * y * c]
duration_pl = tf.placeholder(tf.int32, shape=[None], name='duration')  # [batch]
fc1 = tf.layers.dense(inputs=image_flat, units=128, activation=tf.nn.relu, name='fc1')  # [batch, c]
tiled_fc1 = tf.tile(tf.expand_dims(fc1, axis=1), multiples=[1, max_timesteps, 1])  # [batch, timestep, c]

with tf.variable_scope('rnn'):
    rnn_cell = tf.contrib.rnn.BasicRNNCell(prediction_size ** 2)
    rnn_zero_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    rnn_initial_state_pl = tf.placeholder_with_default(rnn_zero_state, shape=[None, rnn_cell.state_size])
    rnn_output, last_rnn_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=tiled_fc1, sequence_length=duration_pl,
                                                   initial_state=rnn_initial_state_pl)
predictions = tf.layers.dense(inputs=rnn_output, units=prediction_size ** 2, name='prediction')
predictions_sparse = tf.cast(tf.argmax(predictions, axis=2), dtype=tf.int32)

with tf.variable_scope('target'):
    targets_pl = tf.placeholder(tf.int32, shape=[None, max_timesteps, 2], name='target')
    targets_x, targets_y = [tf.squeeze(c, axis=2) for c in tf.split(targets_pl, 2, axis=2)]
    targets_sparse = targets_x * prediction_size + targets_y

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_sparse, logits=predictions))
accuracy_op = tf.count_nonzero(
    tf.logical_and(tf.equal(targets_sparse, predictions_sparse),
                   tf.sequence_mask(duration_pl, max_timesteps)), dtype=tf.int32) / tf.reduce_sum(duration_pl)

# Optimizer
trainables = [t for t in tf.trainable_variables() if 'fc1' not in t.name]
grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, trainables), clip_norm=1.0)
train_op = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(grads, trainables), global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Make sure target splitting is correct
    batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1)
    tx, ty = sess.run([targets_x, targets_y], feed_dict={targets_pl: batch_t})
    assert np.all(batch_t == np.stack([tx, ty], axis=2))
    # Make sure target condensing is correct
    ts = sess.run(targets_sparse, feed_dict={targets_pl: batch_t})
    assert np.all(np.squeeze(batch_t[:, :, 0] * prediction_size + batch_t[:, :, 1]) == np.squeeze(ts))

    total_steps = 100
    print('Duration={}'.format(batch_d))
    print('Batch_t={}'.format(ts))
    for _ in range(total_steps):
        batch_d, batch_images, batch_h, batch_t, batch_vertices = train_data.get_batch_for_rnn(batch_size=1)
        step, loss, accuracy, _ = sess.run([global_step, loss_op, accuracy_op, train_op],
                                           feed_dict={image_pl: batch_images, duration_pl: batch_d,
                                                      targets_pl: batch_t})
        print('Step {}/{}\tLoss={}\tAccuracy={}'.format(step, total_steps, loss, accuracy))

    assert accuracy == 1.0
