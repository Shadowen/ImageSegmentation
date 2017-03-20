import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def create_cnn():
    image = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    previous_mask = tf.placeholder(tf.float32, shape=[None, 32, 32])
    ground_truth_mask = tf.placeholder(tf.float32, shape=[None, 32, 32])
    actions = tf.placeholder(tf.int32, shape=[None, 1024])
    targets = tf.placeholder(tf.float32, shape=[None])

    input_tensor = tf.stack(tf.unstack(image, axis=3) + [previous_mask], axis=3)
    batch_size = tf.shape(input_tensor)[0]

    with tf.variable_scope('conv1'):
        W_conv1, b_conv1 = make_variables([5, 5, 4, 32])
        h_conv1 = tf.nn.relu(conv2d(input_tensor, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2'):
        W_conv2, b_conv2 = make_variables([5, 5, 32, 64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('fc1'):
        W_fc1, b_fc1 = make_variables([8 * 8 * 64, 1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope('fc2'):
        W_fc2, b_fc2 = make_variables([1024, 32 * 32])
        y_flat = tf.matmul(h_fc1, W_fc2) + b_fc2
        y = tf.reshape(y_flat, shape=[-1, 32, 32])

    predicted_idx = tf.argmax(y_flat, dimension=1)
    predicted_mask = tf.one_hot(predicted_idx, depth=1024)
    ground_truth_mask_flat = tf.reshape(ground_truth_mask, shape=[-1, 1024])
    previous_mask_flat = tf.reshape(previous_mask, shape=[-1, 1024])
    previous_tp_mask_flat = previous_mask_flat * ground_truth_mask_flat
    tp_mask = ground_truth_mask_flat * predicted_mask
    fp_mask = (1 - ground_truth_mask_flat) * predicted_mask
    new_tp_mask = (ground_truth_mask_flat - previous_tp_mask_flat) * predicted_mask

    reward = tf.reduce_sum(new_tp_mask) - tf.reduce_sum(fp_mask)

    gather_indices = tf.range(batch_size) * tf.shape(y_flat)[1] + actions
    action_predictions = tf.gather(tf.reshape(y_flat, [-1]), gather_indices)
    # Calculate the loss
    loss = tf.reduce_mean(tf.squared_difference(targets, action_predictions))
    train_op = tf.train.RMSPropOptimizer().minimize(loss)

    return image, previous_mask, ground_truth_mask, actions, targets, predicted_mask, tp_mask, fp_mask, new_tp_mask, \
           reward, loss, train_op


with np.load('dataset_simple.npz') as data:
    input_images, ground_truth_verts, ground_truth_masks = data['arr_0']
    input_images = np.repeat(np.expand_dims(input_images, axis=3), 3, axis=3)

image, previous_mask, ground_truth_mask, actions, targets, predicted_mask, tp_mask, fp_mask, new_tp_mask, reward, \
loss, train_op = create_cnn()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    baseline = sess.run(reward, {
        image:input_images[:1], previous_mask:np.zeros([100, 32, 32])[:1], ground_truth_mask:ground_truth_masks[:1]
    })
    print(np.max(baseline))

    predictions = sess.run(predicted_mask, {
        image:input_images[:1], previous_mask:np.zeros([100, 32, 32])[:1]
    }).reshape([-1, 32, 32])
    plt.imshow(input_images[0] * 0.5, interpolation='nearest')
    plt.imshow(predictions[0], alpha=0.5, cmap='gray', interpolation='nearest')
    plt.show()
