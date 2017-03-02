import numpy as np
import io
import itertools
from supervised_vertices import generate

image_size = 32


def display_sample(x, truth=None, prediction=None, return_tensor=False):
    return display_samples([x], ground_truths=[truth] if truth is not None else None,
                           predictions=[prediction] if prediction is not None else None, return_tensor=return_tensor)


def display_samples(x, ground_truths=None, predictions=None, return_tensor=False):
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
            ax[i][e].imshow(generate.create_point_mask(ground_truths[i], image_size=image_size), cmap='gray',
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
    plt.ion()
    # Create training examples out of it
    image = generate.create_image(ground_truth)

    start_idx = 0
    poly_verts = vertices[start_idx:] + vertices[:start_idx]
    # Probably should optimize this with a numpy array and clever math. Use np.roll

    cursor = poly_verts[0]
    verts_so_far = [cursor]

    states = []
    predictions = []
    for i in itertools.count():
        player_mask = generate.create_history_mask(verts_so_far, len(verts_so_far), image_size)
        cursor_mask = generate.create_point_mask(cursor, image_size)
        valid_mask = generate.create_valid_mask(verts_so_far, len(verts_so_far), image_size)
        state = np.stack([image, player_mask, cursor_mask, valid_mask], axis=2)
        states.append(state)

        y, y_coords = sess.run([est.y, est.y_coords], {est.x: [state], est.keep_prob: 1.0})
        predictions.append(y[0])

        cursor = tuple(reversed(y_coords[0].tolist()))
        verts_so_far.append(cursor)

        distance = np.linalg.norm(np.array(poly_verts[0]) - np.array(cursor))
        if distance < 2 or i > 5:
            break

    predicted_polygon = generate.create_shape_mask(verts_so_far, image_size)
    print('IOU={}'.format(calculate_iou(ground_truth, predicted_polygon)))

    # plt.figure()
    # plt.imshow(np.stack([image, predicted_polygon, np.zeros_like(image)], axis=2), cmap='gray',
    # interpolation='nearest')
    # plt.show()
    display_samples(states, predictions=predictions)


def evaluate_iou(dataset, sess, est):
    failed_shapes = 0
    num_iou = 0
    ious = []
    for polygon_number, (vertices, ground_truth) in enumerate(dataset):
        # Create training examples out of it
        image = generate.create_image(ground_truth)

        start_idx = 0
        poly_verts = vertices[start_idx:] + vertices[:start_idx]
        # Probably should optimize this with a numpy array and clever math. Use np.roll

        cursor = poly_verts[0]
        verts_so_far = [cursor]

        states = []
        predictions = []
        for i in itertools.count():
            history_mask = generate.create_history_mask(verts_so_far, len(verts_so_far), image_size)
            # history_mask = np.zeros_like(image)
            cursor_mask = generate.create_point_mask(cursor, image_size)
            valid_mask = generate.create_valid_mask(np.array(verts_so_far), len(verts_so_far), image_size)  # TODO hacky

            state = np.stack([image, history_mask, cursor_mask, valid_mask], axis=2)
            states.append(state)

            y, y_coords = sess.run([est.y, est.y_coords], {est.x: [state], est.keep_prob: 1.0})
            predictions.append(y[0])

            cursor = tuple(reversed(y_coords[0].tolist()))
            verts_so_far.append(cursor)

            distance = np.linalg.norm(np.array(poly_verts[0]) - np.array(cursor))
            if distance < 2:
                predicted_polygon = generate.create_shape_mask(verts_so_far, image_size)
                iou = calculate_iou(ground_truth, predicted_polygon)
                ious.append(iou)
                num_iou += 1
                break
            elif i >= 7:
                failed_shapes += 1
                # num_iou += 1
                # ious.append(0)
                break
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


# if __name__ == '__main__':
#     data = np.load('dataset_polygons.npy')
#     print('{} polygons loaded.'.format(data.shape[0]))
#     valid_size = data.shape[0] // 10
#     training_data = data[valid_size:]
#     validation_data = data[:valid_size]
#     del data  # Make sure we don't contaminate the training set
#     print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))
#
#     with tf.Session() as sess:
#         est = cnn.CNN_Estimator()
#         saver = tf.train.Saver()
#         # if not os.path.exists(('results')):
#         #     os.makedirs('results/')
#         # train_writer = tf.train.SummaryWriter('./results/train', sess.graph)
#         # valid_writer = tf.train.SummaryWriter('./results/valid')
#         #
#         sess.run(tf.global_variables_initializer())
#
#         saver.restore(sess, 'results/model.ckpt')
#         #
#         for i in range(50):
#             # visualize_weights(sess, est)
#             visual_eval(*validation_data[i])
#
#         ious, failed = evaluate_iou(validation_data, sess, est)
#         print('IOU = {}, failed = {}'.format(sum(ious) / len(ious), failed))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from supervised_vertices.Dataset import get_train_and_valid_datasets
    import tensorflow as tf
    from supervised_vertices.RNN_Estimator import RNN_Estimator

    model = RNN_Estimator(max_timesteps=5, init_scale=0.1)
    with tf.Session() as sess:
        logdir = 'rnn_2'
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)

        saver.restore(sess, save_path=logdir + '/model.ckpt')

        training_set, validation_set = get_train_and_valid_datasets('dataset_polygons.npy')
        d, x, t = training_set.get_batch(max_timesteps=5)
        o = sess.run(model.predictions, feed_dict={model.sequence_length: d, model.inputs: x, model.targets: t})

        batch_size, max_timesteps, _, _, _ = x.shape
        for b in range(batch_size):
            fig, ax = plt.subplots(nrows=5, ncols=max_timesteps, figsize=(10, 2 * max(len(x), 2)))
            [ax[0][t].set_title(d[b]) for t in range(max_timesteps)]

            for timestep in range(x[b].shape[0]):
                for i in range(x[b].shape[-1]):
                    ax[i][timestep].axis('off')
                    ax[i][timestep].imshow(x[b, timestep, :, :, i], cmap='gray', interpolation='nearest')

                i += 1
                ax[i][timestep].axis('off')
                ax[i][timestep].imshow(t[b, timestep, :, :], cmap='gray', interpolation='nearest')

                i += 1
                ax[i][timestep].axis('off')
                ax[i][timestep].imshow(o[b, timestep, :, :], cmap='gray', interpolation='nearest')

            plt.show(block=True)
