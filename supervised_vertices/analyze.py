import numpy as np
import io
import itertools


def display_sample(x, truth=None, prediction=None, return_tensor=False):
    return display_samples([x], ground_truths=[truth] if truth is not None else None,
                           predictions=[prediction] if prediction is not None else None, return_tensor=return_tensor)


def display_samples(x, ground_truths=None, predictions=None, return_tensor=False):
    from supervised_vertices.Dataset import _create_shape_mask, _create_point_mask, _create_history_mask, _create_image

    num_columns = x[0].shape[-1] + (1 if ground_truths is not None else 0) + (1 if predictions is not None else 0)
    fig, ax = plt.subplots(nrows=max(len(x), 2), ncols=num_columns, figsize=(10, 2 * max(len(x), 2)))
    [ax[0][e].set_title(t) for e, t in zip(range(num_columns),
                                           ['Image', 'History', 'Cursor', 'Valid mask'][:x[0].shape[-1]] + (
                                               ['Truth'] if ground_truths is not None else []) + (
                                               ['Prediction'] if predictions is not None else []))]
    for i in range(len(x)):
        for e in range(x[i].shape[-1]):
            ax[i][e].axis('off')
            ax[i][e].imshow(x[i][:, :, e], cmap='gray', interpolation='nearest')
        if ground_truths is not None:
            e += 1
            ax[i][e].axis('off')
            ax[i][e].imshow(_create_point_mask(ground_truths[i], image_size=image_size), cmap='gray',
                            interpolation='nearest')
        if predictions is not None:
            e += 1
            ax[i][e].axis('off')
            ax[i][e].imshow(predictions[i], cmap='gray', interpolation='nearest')

    if return_tensor:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return tf.expand_dims(tf.image.decode_png(buf.getvalue(), channels=1), 0)
    else:
        plt.ion()
        plt.show(block=True)


def visual_eval(vertices, ground_truth):
    from supervised_vertices.Dataset import _create_shape_mask, _create_point_mask, _create_history_mask, _create_image

    plt.ion()
    # Create training examples out of it
    image = _create_image(ground_truth)

    start_idx = 0
    poly_verts = vertices[start_idx:] + vertices[:start_idx]
    # Probably should optimize this with a numpy array and clever math. Use np.roll

    cursor = poly_verts[0]
    verts_so_far = [cursor]

    states = []
    predictions = []
    for i in itertools.count():
        player_mask = _create_history_mask(verts_so_far, len(verts_so_far), image_size)
        cursor_mask = _create_point_mask(cursor, image_size)
        valid_mask = _create_valid_mask(verts_so_far, len(verts_so_far), image_size)
        state = np.stack([image, player_mask, cursor_mask, valid_mask], axis=2)
        states.append(state)

        y, y_coords = sess.run([est.y, est.y_coords], {est.x: [state], est.keep_prob: 1.0})
        predictions.append(y[0])

        cursor = tuple(reversed(y_coords[0].tolist()))
        verts_so_far.append(cursor)

        distance = np.linalg.norm(np.array(poly_verts[0]) - np.array(cursor))
        if distance < 2 or i > 5:
            break

    predicted_polygon = _create_shape_mask(verts_so_far, image_size)
    print('IOU={}'.format(calculate_iou(ground_truth, predicted_polygon)))

    # plt.figure()
    # plt.imshow(np.stack([image, predicted_polygon, np.zeros_like(image)], axis=2), cmap='gray',
    # interpolation='nearest')
    # plt.show()
    display_samples(states, predictions=predictions)


def evaluate_iou(dataset, sess, est, show_images=False, show_intermediate_images=False):
    import numpy as np

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1, a2, b1, b2):
        if np.all(a1 == b1) or np.all(a1 == b2):
            return True, a1
        if np.all(a2 == b1) or np.all(a2 == b2):
            return True, a2

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        if denom == 0:
            return False, None

        num = np.dot(dap, dp)
        intersection = (num / denom.astype(float)) * db + b1
        does_intersect = min(a1[0], a2[0]) <= intersection[0] <= max(a1[0], a2[0]) and min(b1[0], b2[0]) <= \
                                                                                       intersection[
                                                                                           0] <= max(b1[0], b2[0])
        return does_intersect, intersection

    import matplotlib.pyplot as plt
    # Make relative imports work from terminal
    import sys
    sys.path.append('.')
    from supervised_vertices.Dataset import _create_image, _create_history_mask, _create_point_mask, _create_shape_mask

    failed_shapes = 0
    ious = []
    for polygon_number, (vertices, ground_truth) in enumerate(dataset.data):
        # Create training examples out of it
        image = dataset.images[polygon_number] if dataset.images is not None else np.expand_dims(
            _create_image(ground_truth), axis=2)

        start_idx = np.random.randint(len(vertices))
        poly_verts = np.roll(vertices, shift=start_idx, axis=0)

        cursor = poly_verts[0]
        verts_so_far = [cursor]

        for timestep in itertools.count():
            history_mask = _create_history_mask(verts_so_far, len(verts_so_far), dataset.image_size)
            cursor_mask = _create_point_mask(cursor, dataset.image_size)

            state = np.concatenate([image, np.expand_dims(history_mask, axis=2), np.expand_dims(cursor_mask, axis=2)],
                                   axis=2)
            if show_intermediate_images:
                plt.imshow(state, interpolation='nearest')
                plt.show(block=True)

            y, y_coords = sess.run([est.y, est.y_coords], {est.x: [state]})
            cursor = tuple(reversed(y_coords[0].tolist()))

            if timestep >= 7:
                failed_shapes += 1
                break
            # Self intersecting shape
            should_break = False
            for i in range(1, len(verts_so_far)):
                does_intersect, intersection = seg_intersect(np.array(verts_so_far[i - 1]),
                                                             np.array(verts_so_far[i]),
                                                             np.array(verts_so_far[-1]), np.array(cursor))
                if does_intersect and not np.all(intersection == verts_so_far[-1]):
                    predicted_polygon = _create_shape_mask(verts_so_far, dataset.image_size)
                    iou = calculate_iou(ground_truth, predicted_polygon)
                    ious.append(iou)
                    should_break = True
                    break
            if should_break:
                break
            verts_so_far.append(cursor)

        if show_intermediate_images:
            plt.imshow(state, interpolation='nearest')
            plt.show(block=True)
        if show_images:
            state = np.concatenate(
                [image, np.expand_dims(history_mask, axis=2), np.expand_dims(predicted_polygon, axis=2)],
                axis=2)
            plt.imshow(state, interpolation='nearest')
            plt.suptitle('steps={}, IOU={}%'.format(timestep + 1, iou * 100))
            plt.show(block=True)

    return ious, failed_shapes


def calculate_iou(p1, p2):
    """ Calculates the intersection over union of the two masks. """
    intersection = np.count_nonzero(p1 * p2)
    union = np.count_nonzero(p1) + np.count_nonzero(p2) - intersection
    return intersection / union


def visualize_weights(sess, est):
    W_conv1 = sess.run(est.W_conv1)
    fig, ax = plt.subplots(3, 5)
    for i in range(ax.shape[0]):
        for e in range(ax.shape[1]):
            ax[i][e].axis('off')
            ax[i][e].imshow(W_conv1[:, :, i, e], cmap='gray', interpolation='nearest')


if __name__ == '__main__':
    import tensorflow as tf
    from supervised_vertices.Dataset import get_train_and_valid_datasets
    from supervised_vertices.cnn import CNN_Estimator
    import os

    training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')

    logdir = 'cnn_points'
    with tf.Session() as sess:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        est = CNN_Estimator(image_size=32)
        saver = tf.train.Saver()
        if not os.path.exists((logdir + '/iou')):
            os.makedirs(logdir + '/iou')

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))

        ious, failed = evaluate_iou(validation_set, sess, est, show_intermediate_images=False, show_images=True)
        print('IOU={}\tFailed={}'.format(sum(ious) / len(validation_set), failed / len(validation_set)))

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from supervised_vertices.Dataset import get_train_and_valid_datasets
#     import tensorflow as tf
#     from supervised_vertices.RNN_Estimator import RNN_Estimator
#
#     model = RNN_Estimator(max_timesteps=5, init_scale=0.1)
#     with tf.Session() as sess:
#         logdir = 'rnn_2'
#         saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
#
#         saver.restore(sess, save_path=logdir + '/model.ckpt')
#
#         training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
#         d, x, t = training_set.get_batch_for_rnn(max_timesteps=5)
#         o = sess.run(model.predictions, feed_dict={model.sequence_length: d, model.inputs: x, model.targets: t})
#
#         batch_size, max_timesteps, _, _, _ = x.shape
#         for b in range(batch_size):
#             fig, ax = plt.subplots(nrows=5, ncols=max_timesteps, figsize=(10, 2 * max(len(x), 2)))
#             [ax[0][t].set_title(d[b]) for t in range(max_timesteps)]
#
#             for timestep in range(x[b].shape[0]):
#                 for i in range(x[b].shape[-1]):
#                     ax[i][timestep].axis('off')
#                     ax[i][timestep].imshow(x[b, timestep, :, :, i], cmap='gray', interpolation='nearest')
#
#                 i += 1
#                 ax[i][timestep].axis('off')
#                 ax[i][timestep].imshow(t[b, timestep, :, :], cmap='gray', interpolation='nearest')
#
#                 i += 1
#                 ax[i][timestep].axis('off')
#                 ax[i][timestep].imshow(o[b, timestep, :, :], cmap='gray', interpolation='nearest')
#
#             plt.show(block=True)
