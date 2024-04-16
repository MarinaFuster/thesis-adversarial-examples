"""
This file will include functions to plot results and gather
conclusions from those plots.
"""
import matplotlib.pyplot as plt
import numpy as np


def get_label(n):
    return str(n) if n % 5 == 0 else ''


def plot_zoom_multiple_lines(
        values,
        labels,
        colors,
        title,
        x_legend,
        y_legend,
        filename
):

    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    # example zoom for last 80 epochs out of 100 epochs
    # make sure you include in values only data from 20 to 100
    x_ticks = np.arange(0, 41, 1)
    x_labels = [get_label(x) for x in np.arange(35, 76, 1)]
    plt.xticks(x_ticks, x_labels)

    for X, label, color in zip(values, labels, colors):
        plt.plot(X, color=color, label=label)
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'../results/{filename}')
    plt.close()


def str_to_float(file_row):
    return float(file_row.split('\n')[0])


def plot_multiple_lines(
        values,
        labels,
        colors,
        title,
        x_legend,
        y_legend,
        filename
):

    plt.title(title)
    plt.xlabel(x_legend)
    plt.ylabel(y_legend)

    for X, label, color in zip(values, labels, colors):
        plt.plot(X, color=color, label=label)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'../results/{filename}')
    plt.close()

# gray colors
# darkgray
# silver
# lightgray
# whitesmoke
