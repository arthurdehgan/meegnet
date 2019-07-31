"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.io import loadmat
from params import FREQ_DICT, STATE_LIST, SAVE_PATH, WINDOW, OVERLAP


FIG_PATH = SAVE_PATH + "figures/"
SAVE_PATH = SAVE_PATH + "results/"
NAME_COSP = "cosp"
NAME_COV = "cov"
PREFIX = "bootstrapped_subsamp_"
MOY = "moy" in NAME_COSP
SUBSAMP = "subsamp" in NAME_COSP.split("_")
PERM = True
PVAL = 0.001

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
COLORS = ["#C2C2C2"] + list(sns.color_palette("deep"))
WIDTH = 0.90
GRAPH_TITLE = "Gender classifications"

RESOLUTION = 300


def autolabel(ax, rects, thresh):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()
        if height > thresh:
            color = "green"
        else:
            color = "black"

        if height != 0:
            ax.text(
                rect.get_x() + width / 2.0,
                width + 1.0 * height,
                "%d" % int(height),
                ha="center",
                va="bottom",
                color=color,
                size=14,
            )
    return ax


# barplot parameters
def visualisation(pval):
    scoring = "acc"
    labels = list(FREQ_DICT.keys())
    labels = ["Covariance"] + labels
    groups = STATE_LIST

    nb_labels = len(labels)
    dat, stds = [], []
    thresholds = []
    for state in groups:
        temp_std, temp, temp_thresh = [], [], []
        for lab in labels:
                file_name =  SAVE_PATH + f"{algo}_results.mat" # TODO
            try:
                data = loadmat(file_name)
                n_rep = int(data["n_rep"])
                data = np.asarray(data[scoring][0]) * 100
                n_cv = int(len(data) / n_rep)
            except IOError:
                print(file_name, "not found.")
            except KeyError:
                print(file_name, "key error")

            temp.append(np.mean(data))
            std_value = np.std(data)
            temp_std.append(std_value)
        dat.append(temp)
        stds.append(temp_std)

    fig = plt.figure(figsize=(10, 5))  # size of the figure

    # Generating the barplot (do not change)
    ax = plt.axes()
    temp = 0
    offset = 0.4
    for group in range(len(groups)):
        bars = []
        t = thresholds[group]
        data = dat[group]
        std_val = stds[group]
        for i, val in enumerate(data):
            pos = i + 1
            if i == 1:
                temp += offset  # offset for the first bar
            color = COLORS[i]
            bars.append(ax.bar(temp + pos, val, WIDTH, color=color, yerr=std_val[i]))
            start = (
                (temp + pos * WIDTH) / 2 + 1 - WIDTH
                if pos == 1 and temp == 0
                else temp + pos - len(data) / (2 * len(data) + 1)
            )
            end = start + WIDTH
            ax.plot([start, end], [t, t], "k--", label="p < {}".format(PVAL))
            # ax = autolabel(ax, bars[i], t)

        temp += pos + 1

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
    ax.set_title(GRAPH_TITLE)
    ax.set_xticklabels(groups)
    ax.set_xticks(
        [
            ceil(nb_labels / 2) + offset + i * (1 + offset + nb_labels)
            for i in range(len(groups))
        ]
    )
    # labels[-1] = labels[-1][:-1]
    labels = ["CNN"] + algos
    # ax.legend(bars, labels, frameon=False)
    ax.legend(
        bars,
        labels,
        # loc="upper center",
        # bbox_to_anchor=(0.5, -0.05),
        fancybox=False,
        shadow=False,
        # ncol=len(labels),
    )

    file_name = f"barplot.png" # TODO
    save_path = FIG_PATH + file_name
    print(save_path)
    fig.savefig(save_path, dpi=RESOLUTION)
    plt.close()


if __name__ == "__main__":
    visualisation(PVAL)
