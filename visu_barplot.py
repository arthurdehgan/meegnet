"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = list()


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


def _simple_barplot(ax, bars, vals, stds, thresholds, pval, width, colors, offset):
    for j, val in enumerate(vals):
        pos = j + 1
        bars.append(
            ax.bar(
                offset + pos,
                val,
                width,
                color=colors[j],
                yerr=stds[j] if stds is not None else stds,
            )
        )

        if thresholds is not None:
            start = (
                (offset + pos * width) / 2 + 1 - width
                if pos == 1 and offset == 0
                else offset + pos - len(vals) / (2 * len(vals) + 1)
            )
            end = start + width
            ax.plot(
                [start, end],
                [thresholds[j], thresholds[j]],
                "k--",
                label="p < {}".format(pval) if pval != 0 and j == 0 else "",
            )

    return ax, bars


def generate_barplot(
    ylabel,
    labels,
    values,
    thresholds=None,
    stds=None,
    title="",
    groups=None,
    mini=0.5,
    maxi=1,
    pval=0.01,
    width=0.9,
    autolabel=False,
    colors=list(sns.color_palette("deep")),
):
    """Generates a barplot for groups of data.

    Parameters
    ----------
    ylabel:
        The label for the y axis
    labels:
        The labels of individual bars in each group, will be the x axis labels
        if no groups
    values:
        The values, list of list of values if groups. A list of values otherwise.
    thresholds:
        The thresholds for each bar. If groups, a list of lists of thresholds
    pval:
        The value of p that corresponds to the thresholds (for the legend)
    stds:
        The standard deviation, list of list of values if groups. A list of
        stds otherwise.
    title:
        The title for the graph
    groups:
        The labels on the x axis for each group
    mini:
        The minimum value for the scale on the y axis
    maxi:
        The maximum value for the scale on the y axis
    width:
        The width of the bars. 1 means bars will touch each others
    autolabel:
        if True, will display value of the bar on top of the bar

    Returns
    -------

    """
    ax = plt.axes()
    n_labels = len(labels)
    if groups is not None:
        n_groups = len(groups)
        bars = []
        for i in range(n_groups):
            ax, bars = _simple_barplot(
                ax,
                bars,
                values[i],
                stds[i],
                thresholds[i],
                pval,
                width,
                colors,
                len(bars) + i,
            )

        ax.set_xticklabels(groups)
        ax.set_xticks(
            [ceil(n_labels / 2) + i * (1 + n_labels) for i in range(len(groups))]
        )
    else:
        ax, bars = _simple_barplot(
            ax, [], values, stds, thresholds, pval, width, colors, offset=0
        )
        ax.set_xticklabels([0] + labels)

    if autolabel:
        ax = autolabel(ax, bars, thresholds)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=mini, top=maxi)
    ax.legend(bars, labels, fancybox=False, shadow=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


if __name__ == "__main__":
    RESOLUTION = 300

    # Data creation
    ylabel = "Random"
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    thresholds = [[0.5, 1, 1.5], [1.5, 2, 2.5], [2.5, 3, 3.5]]
    labels = ["A", "B", "C"]
    groups = ["Group1", "Group2", "Group3"]
    stds = [[0.1, 0.3, 0.5], [0.3, 0.3, 0.1], [0.4, 0.3, 0.2]]
    pval = 0.0001

    # Create the barplot
    generate_barplot(
        ylabel,
        labels,
        values,
        thresholds,
        stds=stds,
        groups=groups,
        pval=pval,
        mini=0,
        maxi=10,
    )

    # Saving the barplot
    file_name = f"test1.png"
    plt.savefig(file_name, dpi=RESOLUTION)
    plt.close()

    # Data creation
    ylabel = "Random"
    values = [1, 2, 3, 4, 5, 6]
    labels = ["A", "B", "C", "a", "b", "c"]
    thresholds = [0.5, 1, 1.5, 1.5, 2, 2.5]
    pval = 0.0001

    # Create the barplot
    generate_barplot(ylabel, labels, values, thresholds, pval=pval, mini=0, maxi=7)

    # Saving the barplot
    file_name = f"test2.png"
    plt.savefig(file_name, dpi=RESOLUTION)
    plt.close()
