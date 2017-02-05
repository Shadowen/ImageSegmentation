import tensorflow as tf
import numpy as np
import io
import itertools
import matplotlib.pyplot as plt
from supervised_vertices import generate
from supervised_vertices import cnn
import os

image_size = 32


def display_sample(x, truth=None, prediction=None, return_tensor=False):
    return display_samples([x], ground_truths=[truth] if truth is not None else None,
        predictions=[prediction] if prediction is not None else None, return_tensor=return_tensor)


def display_samples(x, ground_truths=None, predictions=None, return_tensor=False):
    num_columns = 3 + (1 if ground_truths is not None else 0) + (1 if predictions is not None else 0)
    fig, ax = plt.subplots(nrows=len(x), ncols=num_columns, figsize=(10, 2 * len(x)))
    [ax[0][e].set_title(t) for e, t in zip(range(num_columns),
        ['Image', 'History', 'Cursor'] + (['Truth'] if ground_truths is not None else []) + (
            ['Prediction'] if predictions is not None else []))]
    for i in range(len(x)):
        for e in range(3):
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
        state = np.stack([image, player_mask, cursor_mask], axis=2)
        states.append(state)

        y, y_coords = sess.run([est.softmax, est.y_coords], {est.x:[state], est.keep_prob:1.0})
        predictions.append(y[0])

        cursor = tuple(reversed(y_coords[0].tolist()))
        verts_so_far.append(cursor)

        distance = np.linalg.norm(np.array(poly_verts[0]) - np.array(cursor))
        if distance < 2 or i > 5:
            break

    predicted_polygon = generate.create_shape_mask(verts_so_far, image_size)
    print('IOU={}'.format(calculate_iou(ground_truth, predicted_polygon)))

    plt.figure()
    plt.imshow(np.stack([image, predicted_polygon, np.zeros_like(image)], axis=2), cmap='gray', interpolation='nearest')
    plt.show()
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
            player_mask = generate.create_history_mask(verts_so_far, len(verts_so_far), image_size)
            cursor_mask = generate.create_point_mask(cursor, image_size)
            state = np.stack([image, player_mask, cursor_mask], axis=2)
            states.append(state)

            y, y_coords = sess.run([est.y, est.y_coords], {est.x:[state], est.keep_prob:1.0})
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


if __name__ == '__main__':
    data = np.load('dataset_polygons.npy')
    print('{} polygons loaded.'.format(data.shape[0]))
    valid_size = data.shape[0] // 10
    training_data = data[valid_size:]
    validation_data = data[:valid_size]
    del data  # Make sure we don't contaminate the training set
    print('{} for training. {} for validation.'.format(len(training_data), len(validation_data)))

    with tf.Session() as sess:
        est = cnn.Estimator()
        saver = tf.train.Saver()
        if not os.path.exists(('results')):
            os.makedirs('results/')
        train_writer = tf.train.SummaryWriter('./results/train', sess.graph)
        valid_writer = tf.train.SummaryWriter('./results/valid')

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, 'results/model.ckpt')
        for i in range(18, 50):
            print(validation_data[i][0])
            visual_eval(*validation_data[i])

            # valid_x, valid_t = zip(
            #     *[generate.create_training_sample(image_size, vertices, truth) for vertices,
            # truth in validation_data])
            # valid_y = sess.run(est.y, {est.x:valid_x, est.keep_prob:1.0})
            # valid_writer.add_summary(sess.run(
            #     tf.image_summary('samples', display_samples(valid_x[:10], valid_t[:10], valid_y[:10],
            # return_tensor=True))))
            # evaluate_iou(validation_data, valid_writer)
