import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def visualize_cam_line(weights, ecg, save_path, show_flag=False):
    x_coordinate = np.linspace(0, len(ecg), len(ecg))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    title = save_path.split('/')[-1].split('_')[0]
    plt.title(title)

    # Scale weights to 0, 0.7
    scaled_weights = [(w - min(weights)) * 0.7 / (max(weights) - min(weights)) for w in weights]

    for x, weight in zip(x_coordinate, scaled_weights):
        plt.axvspan(x - 0.5, x + 0.5, color=cm.Oranges(weight))  # pylint: disable=no-member

    # Major ticks every 20, minor ticks every 5

    # TODO: also map y ticks to the same scale as ecg paper
    major_ticks = np.arange(0, len(ecg), 500 * 0.04 * 5)
    minor_ticks = np.arange(0, len(ecg), 500 * 0.04)

    # Set ticks scale
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # Set ticks line shape
    ax.grid(which='major')
    ax.grid(which='minor', ls='--')

    plt.plot(x_coordinate, ecg, color='b')
    if show_flag:
        plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_cam_scatter(colors, ecg, save_path, show_flag=False):
    # TODO: Add grid

    x_coordinate = np.linspace(0, len(ecg), len(ecg))
    title = save_path.split('/')[-1].split('_')[0]
    plt.title(title)

    plt.scatter(x_coordinate, ecg, c=colors)
    if show_flag:
        plt.show()
    plt.savefig(save_path)
    plt.close()
