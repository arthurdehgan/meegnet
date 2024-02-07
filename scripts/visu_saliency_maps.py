import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from meegnet.parsing import parser
from meegnet.viz import load_info
import mne
import seaborn as sns


def generate_saliency_figure(
    saliencies: dict,
    save_path: str = "",
    suffix: str = "",
    title: str = "",
    sensors: list = [""],
    eventclf=False,
    sfreq=500,
    cmap="coolwarm",
    stim_tick=75,
):
    """
    Generates a figure visualizing saliency maps for MEG data.

    This function creates a grid of images showing the saliency maps for different
    types of stimuli (e.g., image and sound) and sensor channels (e.g., MAG, GRAD1, GRAD2).
    It also plots a topomap for the maximum saliency index along with a color bar.

    Parameters
    ----------
    saliencies : dict
        Dictionary containing saliency maps. Keys should correspond to different types
        of stimuli (e.g., "image", "sound"), and values should be numpy arrays of shape
        3 x sensors x samples, representing the saliency maps for each channel.
    save_path : str, optional
        Path to save the generated figure. Default is an empty string, which means the
        figure will not be saved automatically.
    suffix : str, optional
        Suffix to append to the filename when saving the figure. Default is an empty string.
    title : str, optional
        Title for the figure. Default is an empty string.
    sensors : list, optional
        List of sensor types to include in the visualization. Default is [""].
    eventclf : bool, optional
        Flag indicating whether the saliency maps are for event classification. If True,
        the y-axis ticks and labels are adjusted accordingly. Default is False.
    sfreq : int, optional
        Sampling frequency for computation of  xticks. Default is  500.
    cmap : str, optional
        Colormap to use for displaying the topo- and saliency maps. Default is "coolwarm".
    stim_tick : int, optional
        Tick position for the stimulus event in the time axis. Default is  75.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that the input saliency maps are normalized to have zero mean and
    unit variance. The colormap used for displaying the saliency maps is "coolwarm" by default,
    but can be changed with the `cmap` parameter.

    The function creates a grid layout with a subplot for each sensor channel and a subplot for the
    topomap. The grid layout is dynamically adjusted based on the number of sensors.

    The `eventclf` flag allows for adjustments in the y-axis ticks and labels for event classification
    scenarios, where the stimulus event is marked with a specific tick position.

    The function does not handle exceptions that may occur during the plotting process, such as issues with
    file I/O or invalid input data.
    """

    if suffix != "" and not suffix.endswith("_"):
        suffix += "_"
    n_blocs = len(sensors)  # number of blocs of figures in a line
    n_lines = len(saliencies)  # number of lines for the pyplot figure
    n_cols = n_blocs * 3 + 1  # number of columns for the pyplot figure
    grid = GridSpec(n_lines, n_cols)
    fig = plt.figure(figsize=(n_cols * 2, n_lines * 2))
    plt.title(title)
    plt.axis("off")
    axes = []
    tick_ratio = int(1000 / sfreq)
    for i, label in enumerate(saliencies.keys()):
        gradient = saliencies[label]
        gradient -= gradient.mean()
        gradient /= np.abs(gradient).max()
        for j, sensor_type in zip(range(0, n_blocs * 3, n_blocs), sensors):
            idx = j // 3
            grads = gradient[idx]
            segment_length = grads.shape[1]
            n_sensors = grads.shape[0]
            vmax = grads.max()
            vmin = grads.min()
            max_idx = np.argmax(np.mean(np.abs(grads), axis=0))
            axes.append(fig.add_subplot(grid[i, j : j + 2]))
            plt.imshow(
                grads,
                interpolation="nearest",
                aspect=1,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            axes[-1].spines["top"].set_visible(False)
            axes[-1].spines["right"].set_visible(False)
            axes[-1].yaxis.tick_right()

            if eventclf:
                x_ticks = sorted(
                    [0, stim_tick, int(segment_length / 2), segment_length] + [max_idx]
                )
                ticks_values = [(x_tick - stim_tick) * tick_ratio for x_tick in x_ticks]
                plt.axvline(x=stim_tick, color="black", linestyle="--", linewidth=1)
                plt.axvline(x=max_idx, color="green", linestyle="--", linewidth=1)
            else:
                x_ticks = [0, int(segment_length / tick_ratio), segment_length]
                ticks_values = [x_tick * tick_ratio for x_tick in x_ticks]
            plt.xticks(x_ticks, ticks_values, fontsize=8)
            plt.yticks([0, n_sensors], [n_sensors, 0])

            if j == 0:
                axes[-1].text(-50, 50, label, ha="left", va="center", rotation="vertical")
            if idx == n_blocs - 1:
                axes[-1].yaxis.set_label_position("right")
                plt.ylabel("sensors")
            if i == 0:
                plt.title(sensor_type)
            if i == 1:
                plt.xlabel("time (ms)")
            axes.append(fig.add_subplot(grid[i, j + 2]))
            data = grads[:, max_idx]
            info = load_info()

            im, _ = mne.viz.plot_topomap(
                data.ravel(),
                info,
                res=128,
                cmap=cmap,
                vlim=(vmin, vmax) if vmax > vmin else (None, None),
                show=False,
                contours=1,
                extrapolate="local",
                axes=axes[-1],
            )
            if idx == n_blocs - 1:
                axes.append(fig.add_subplot(grid[i, n_blocs * 3]))
                fig.colorbar(
                    im,
                    ax=axes[-1],
                    location="right",
                    shrink=0.9,
                    ticks=(vmin, 0, vmax),
                )
                axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{suffix}saliencies.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    mne.set_log_level(False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.98,
        help="the confidence needed for a trial to be selected for visualisation",
    )
    args = parser.parse_args()

    random_subject = args.seed

    #################################################
    ### Create network name depending on the args ###
    #################################################

    if args.fold is not None:
        fold = args.fold + 1
    else:
        fold = 1
    name = f"{args.model_name}_{args.seed}_fold{fold}_{args.sensors}"
    suffixes = ""
    if args.net_option == "custom_net":
        if args.batchnorm:
            suffixes += "_BN"
        if args.maxpool != 0:
            suffixes += f"_maxpool{args.maxpool}"

        name += f"_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
        name += suffixes

    #########################################
    ### Get saliencies found in data-path ###
    #########################################

    sal_path = os.path.join(args.data_path, "saliency_maps", name)
    file_list = os.listdir(sal_path)
    # TODO might want to use participants.csv for subject list :
    subjects = np.unique([file.split("_")[0] for file in file_list])[: args.max_subj]

    if args.eventclf:
        labels = ["visual", "auditory"]
        all_saliencies = {labels[0]: [], labels[1]: []}
    elif args.subclf:
        labels = subjects
        all_saliencies = {sub: [] for sub in subjects}
    else:
        labels = ["male", "female"]
        all_saliencies = {labels[0]: [], labels[1]: []}

    #########################
    ### HARD CODED VALUES ###
    #########################

    # TODO change design to create an object with info about the dataset and pass it from script to script
    sensors = ["MAG", "PLANNAR1", "PLANNAR2"]
    cmap = "coolwarm"
    stim_tick = 75

    # Some tested aletrnatives for the colormap:
    # cmap = sns.color_palette("icefire", as_cmap=True)
    # cmap = sns.color_palette("coolwarm", as_cmap=True, center="dark")
    # cmap = "inferno"
    # cmap = "seismic"

    ######################################
    ### LOAD DATA AND GENERATE FIGURES ###
    ######################################

    # label contains same information as sub for subclf but we only load files that have label == sub
    for i, sub in enumerate(subjects):
        sub_saliencies = {}
        for label in all_saliencies.keys():
            if args.subclf:
                if label != sub:
                    continue
            saliencies = {}
            nofile = False
            for saliency_type in ("pos", "neg"):
                lab = "" if args.subclf else f"_{label}"
                sal_file = os.path.join(
                    sal_path,
                    f"{sub}{lab}_{saliency_type}_sal_{args.confidence}confidence.npy",
                )
                if os.path.exists(sal_file):
                    try:
                        saliencies[saliency_type] = np.load(sal_file)
                    except IOError:
                        logging.warning(f"Error loading {sal_file}")
                        nofile = True
                        continue
                else:
                    nofile = True
                    continue
            if nofile:
                continue
            # TODO look into this line of code: do we want just pos ? pos - neg ? or absolute value / maximum ?
            sub_saliencies[label] = saliencies["pos"] - saliencies["neg"]
            if sub_saliencies[label].size == 0:
                continue
            all_saliencies[label].append(sub_saliencies[label].mean(axis=0))
            if args.subclf:
                break  # we only need to add one label per subject so we get out of the loop

        if i == random_subject:
            suffix = f"{sub}_single_trial"
            if args.subclf:
                suffix += "_subclf"
            elif args.eventclf:
                suffix += "_eventclf"
            else:
                suffix += "_sexclf"
            generate_saliency_figure(
                {key: val[0] for key, val in sub_saliencies.items()},
                save_path=args.save_path,
                suffix=suffix,
                sensors=sensors,
                title=f"saliencies for a single trial of subject {sub}",
                eventclf=args.eventclf,
                cmap=cmap,
                stim_tick=75,
            )
            suffix = f"{sub}_all_trials"
            if args.subclf:
                suffix += "_subclf"
            elif args.eventclf:
                suffix += "_eventclf"
            else:
                suffix += "_sexclf"
            generate_saliency_figure(
                {key: np.mean(val, axis=0) for key, val in sub_saliencies.items()},
                save_path=args.save_path,
                suffix=suffix,
                sensors=sensors,
                title=f"saliencies for the averaged trials of subject {sub}",
                eventclf=args.eventclf,
                cmap=cmap,
                stim_tick=75,
            )

    final_dict = {key: np.mean(val, axis=0) for key, val in all_saliencies.items()}
    if args.subclf:
        final_dict = {"all_subj": np.mean(list(final_dict.values()), axis=0)}
        labels = ["all_subj"]
        suffix = "subclf"
    elif args.eventclf:
        suffix = "eventclf"
    else:
        suffix = "sexclf"

    generate_saliency_figure(
        final_dict,
        save_path=args.save_path,
        suffix=suffix,
        sensors=sensors,
        title="saliencies averaged across all subjects",
        eventclf=args.eventclf,
        cmap=cmap,
        stim_tick=75,
    )
