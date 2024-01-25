import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from meegnet.parsing import parser
from meegnet.viz import load_info, generate_topomap
import mne
import seaborn as sns

mne.set_log_level(False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def generate_saliency_figure(
    saliencies: dict,
    save_path: str = "",
    suffix: str = "",
    title: str = "",
    eventclf=False,
):
    """generate_saliency_figure.

    Parameters
    ----------
    saliencies : dict
        saliency maps: must be a dictionnary contining keys "image" and "sound", values must be numpy array
        of shape 3 x height x width
    save_path : str
        the path where to save the figure
    save_path : str
        the path to get the data from (for psd computation only)
    suffix : str
        suffix
    sfreq : int
        sampling frequency for psd and data loading
    labels : list
        labels
    """
    if suffix != "" and not suffix.endswith("_"):
        suffix += "_"
    grid = GridSpec(2, 10)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.axis("off")
    axes = []
    for i, label in enumerate(saliencies.keys()):
        gradient = saliencies[label]
        if np.isnan(gradient):
            continue
        gradient -= gradient.mean()
        gradient /= np.abs(gradient).max()
        # Change this part, hard codded = bad and we want some flexibility depending on the use of sensors
        channels = "MAG"  # ("MAG", "GRAD1", "GRAD2")
        for j, chan in zip(range(0, 9, 3), channels):
            idx = j // 3
            grads = gradient[idx]
            vmax = grads.max()
            vmin = grads.min()
            highest = np.argmax(np.mean(np.abs(grads), axis=0))
            # cmap = sns.color_palette("icefire", as_cmap=True)
            # cmap = sns.color_palette("coolwarm", as_cmap=True, center="dark")
            # cmap = "inferno"
            cmap = "coolwarm"
            # cmap = "seismic"
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
            plt.yticks([], [])
            if eventclf:
                plt.xticks([0, 75, 200, 400], [-150, 0, 250, 650], fontsize=8)
                # plt.yticks([0, 102], [102, 0])
                plt.axvline(x=75, color="black", linestyle="--", linewidth=1)
                plt.axvline(x=highest, color="green", linestyle="--", linewidth=1)
            else:
                plt.xticks([0, 200, 400], [0, 1000, 2000], fontsize=8)
            if j == 0:
                axes[-1].text(-50, 50, label, ha="left", va="center", rotation="vertical")
            if idx == 2:
                axes[-1].yaxis.set_label_position("right")
                plt.ylabel("sensors")
            if i == 0:
                plt.title(chan)
            if i == 1:
                plt.xlabel("time (ms)")
            axes.append(fig.add_subplot(grid[i, j + 2]))
            data = grads[:, highest]
            info = load_info()
            im = generate_topomap(data, info, vmin=vmin, vmax=vmax, cmap=cmap)
            if idx == 2:
                axes.append(fig.add_subplot(grid[i, 9]))
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
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.98,
        help="the confidence needed for a trial to be selected for visualisation",
    )
    args = parser.parse_args()

    random_subject = args.seed

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

    sal_path = os.path.join(args.data_path, "saliency_maps", name)
    file_list = os.listdir(sal_path)
    # might want to use participants.csv for subject list :
    subjects = np.unique([file.split("_")[0] for file in file_list])[: args.max_subj]
    all_subjects_saliencies = {"image": [], "sound": []}

    if args.eventclf:
        labels = ["visual", "auditory"]
        all_subjects_saliencies = {labels[0]: [], labels[1]: []}
    elif args.subclf:
        labels = subjects
        all_subjects_saliencies = {sub: [] for sub in subjects}
    else:
        labels = ["male", "female"]
        all_subjects_saliencies = {labels[0]: [], labels[1]: []}

    # label contains same information as sub for subclf but we only load files that have label == sub
    for i, sub in enumerate(subjects):
        current_sub = {}
        for label in all_subjects_saliencies.keys():
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
                        if not saliencies[saliency_type]:
                            logging.info(f"No saliencies found in file : {sal_file}")
                            continue
                    except:
                        print(f"Error loading {sal_file}")
                        nofile = True
                        continue
                    if saliencies[saliency_type].shape[1:] != (3, 102, 400):
                        nofile = True
                        continue
                    elif np.isnan(saliencies[saliency_type]).any():
                        print(f"nan found in {sal_file}")
                        nofile = True
                        continue
                else:
                    nofile = True
                    continue
            if nofile:
                continue
            current_sub[label] = saliencies["pos"] - saliencies["neg"]
            if current_sub[label].size == 0:
                continue
            all_subjects_saliencies[label].append(current_sub[label].mean(axis=0))
            if args.subclf:
                break  # we only need to add one label per subject so get out of the loop

        if i == random_subject:
            suffix = f"{sub}_single_trial"
            if args.subclf:
                suffix += "_subclf"
            elif args.eventclf:
                suffix += "_eventclf"
            else:
                suffix += "_sexclf"

            if list(current_sub.values())[0].size == 0:
                continue
            generate_saliency_figure(
                {key: val[0] for key, val in current_sub.items()},
                save_path=args.save_path,
                suffix=suffix,
                title=f"saliencies for a single trial of subject {sub}",
                eventclf=args.eventclf,
            )
            suffix = f"{sub}_all_trials"
            if args.subclf:
                suffix += "_subclf"
            elif args.eventclf:
                suffix += "_eventclf"
            else:
                suffix += "_sexclf"
            generate_saliency_figure(
                {key: np.mean(val, axis=0) for key, val in current_sub.items()},
                save_path=args.save_path,
                suffix=suffix,
                title=f"saliencies for the averaged trials of subject {sub}",
                eventclf=args.eventclf,
            )

    final_dict = {key: np.mean(val, axis=0) for key, val in all_subjects_saliencies.items()}
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
        title="saliencies averaged across all subjects",
        eventclf=args.eventclf,
    )
