from util import *


class PolygonEnv():
    def __init__(self, dataset):
        self.dataset = dataset

    def _make_state(self):
        # TODO May need to do some resizing here as self.image and self.histories are not guaranteed to be the same size
        return np.concatenate([self.image, self.histories], axis=2)

    def reset(self):
        self.done = False
        self.image, self.vertices = [x[0] for x in self.dataset.raw_sample(batch_size=1)]
        self.predicted_vertices = self.vertices[:self.dataset._history_length].tolist()
        assert len(self.predicted_vertices) == self.dataset._history_length
        self.histories = create_history(vertices=self.predicted_vertices,
                                        end_idx=len(self.predicted_vertices) - 1,
                                        history_length=self.dataset._history_length,
                                        image_size=self.dataset._prediction_size)
        return self._make_state()

    def step(self, action):
        """
        :param action:
        :return: tuple(state, reward, done)
         where
            state is a NumPy array of shape (prediction_size, prediction_size, 3 + history_length)
            reward is a float
            done is a boolean
        """
        if self.done:
            raise NotImplementedError("Didn't reset a finished environment!")

        self.predicted_vertices.append(action)

        predicted_mask = create_point_mask(action, self.dataset._prediction_size)
        self.histories = np.roll(self.histories, axis=2, shift=1)
        self.histories[:, :, -1] = predicted_mask

        if np.all(action == self.vertices[0]):
            reward = self._calculate_iou()
            self.done = True
            return self._make_state(), reward, True

        return self._make_state(), 0, False

    def _calculate_iou(self):
        predicted_mask = create_shape_mask(np.array(self.predicted_vertices), self.dataset._prediction_size)
        true_mask = create_shape_mask(self.vertices, self.dataset._prediction_size)
        intersection = np.count_nonzero(np.logical_and(predicted_mask, true_mask))
        union = np.count_nonzero(np.logical_or(predicted_mask, true_mask))
        return intersection / union


if __name__ == '__main__':
    from Dataset import get_train_and_valid_datasets
    import matplotlib.pyplot as plt
    import matplotlib.lines


    def show_state(state):
        fig, ax = plt.subplots()
        plt.imshow(state[:, :, 0:3])
        # Ground truth
        for e, v in enumerate(training_env.vertices):
            ax.add_artist(plt.Circle(v, 0.5, color='lightgreen'))
            plt.text(v[0], v[1], e, color='forestgreen')
        for a, b in iterate_in_ntuples(training_env.vertices, n=2):
            ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='forestgreen'))
        # Predicted
        for e, v in enumerate(training_env.predicted_vertices):
            ax.add_artist(plt.Circle(v, 0.5, color='salmon'))
            plt.text(v[0] + 0.5, v[1] + 0.5, e, color='red')
        for a, b in iterate_in_ntuples(training_env.predicted_vertices, n=2, loop=False):
            ax.add_line(matplotlib.lines.Line2D([a[0], b[0]], [a[1], b[1]], color='red'))


    training_env, validation_env = [PolygonEnv(d) for d in
                                    get_train_and_valid_datasets('/home/wesley/docker_data/polygons_dataset_3',
                                                                 max_timesteps=10,
                                                                 image_size=28,
                                                                 prediction_size=28,
                                                                 history_length=2,
                                                                 is_local=True,
                                                                 load_max_images=2,
                                                                 validation_set_percentage=0.5)]
    state = training_env.reset()
    show_state(state)
    for timestep in range(training_env.dataset._history_length, len(training_env.vertices) + 1):
        print("t={}".format(timestep))
        a = training_env.vertices[timestep % len(training_env.vertices)]
        state, reward, done = training_env.step(action=[a[0], a[1]])
        print(reward)
        show_state(state)
    plt.show(block=True)
