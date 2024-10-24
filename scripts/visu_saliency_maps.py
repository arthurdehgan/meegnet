import os
import mne
import warnings
import logging
from collections import defaultdict
import numpy as np
import configparser
import pandas as pd
from meegnet.parsing import parser, save_config
from meegnet.viz import generate_saliency_figure
from meegnet_functions import prepare_logging, get_name, load_info


LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

LOG.warning("This script is deprecated, prefer using the jupyter notebook instead")

warnings.filterwarnings("ignore")


def get_saliency_data(saliency_dict, option):
    if option in ("pos", "neg"):
        return saliency_dict[option]
    else:
        saliencies = {}
        if option == "sum":
            operation = lambda a, b: a + b
        else:
            operation = lambda a, b: a - b
        for lab, pos in saliency_dict["pos"].items():
            saliencies[lab] = operation(np.array(pos), np.array(saliency_dict["neg"][lab]))
    return saliencies


if __name__ == "__main__":
    mne.set_log_level(False)

    ###############
    ### PARSING ###
    ###############

    args = parser.parse_args()
    save_config(vars(args), args.config)

    fold = None if args.fold == -1 else int(args.fold)
    if args.clf_type == "eventclf":
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run event classification"

    if args.clf_type == "eventclf":
        labels = ["visual", "auditory"]
    else:
        labels = []

    name = get_name(args)

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)
    if args.clf_type == "subclf":
        data_path = os.path.join(args.save_path, f"downsampled_{args.sfreq}")
        n_subjects = len(os.listdir(data_path))
        n_outputs = min(n_subjects, args.max_subj)
        lso = False
    else:
        n_outputs = 2
        lso = True

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        prepare_logging("gradcam_computations", args, LOG, fold)

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    visu_path = os.path.join(args.save_path, "visualizations")
    if not os.path.exists(visu_path):
        os.makedirs(visu_path)
    sal_path = os.path.join(args.save_path, "saliency_maps", name)
    if not os.path.exists(sal_path):
        os.makedirs(sal_path)

    ####################
    ### LOADING DATA ###
    ####################

    csv_file = os.path.join(args.save_path, f"participants_info.csv")
    dataframe = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]
    np.random.seed(args.seed)
    random_subject_idx = np.random.choice(np.arange(len(subj_list)))

    #########################
    ### HARD CODED VALUES ###
    #########################

    # TODO add those to a TOML file, either config or default_values
    sensors = ["MAG", "PLANAR1", "PLANAR2"]
    cmap = "coolwarm"
    stim_tick = 75
    saliency_types = ("pos", "neg")
    saliency_options = ("pos", "neg", "sum", "diff")

    # Some tested aletrnatives for the colormap:
    # cmap = sns.color_palette("icefire", as_cmap=True)
    # cmap = sns.color_palette("coolwarm", as_cmap=True, center="dark")
    # cmap = "inferno"
    # cmap = "seismic"

    #################
    ### MAIN LOOP ###
    #################

    all_saliencies = defaultdict(lambda: defaultdict(lambda: []))

    LOG.info(f"Generating figure for sensors: {sensors}")
    LOG.info(f"For the {args.clf_type} classification")

    # label contains same information as sub for subclf but we only load files that have label == sub
    for i, sub in enumerate(subj_list):
        sub_saliencies = defaultdict(lambda: {})
        for label in labels:
            if args.clf_type == "subclf":
                if label != sub:
                    continue
            nofile = False

            for saliency_type in saliency_types:
                lab = "" if args.clf_type == "subclf" else f"_{label}"
                saliency_file = os.path.join(
                    sal_path,
                    f"{sub}{lab}_{saliency_type}_sal_{args.confidence}confidence.npy",
                )
                if os.path.exists(saliency_file):
                    try:
                        saliencies = np.load(saliency_file)
                        sub_saliencies[saliency_type][label] = saliencies
                    except IOError:
                        logging.warning(f"Error loading {saliency_file}")
                        nofile = True
                        continue
                else:
                    nofile = True
                    continue
                if len(saliencies.shape) == 3:
                    saliencies = saliencies[np.newaxis, ...]  # If only one saliency in file
                elif len(saliencies.shape) != 4:
                    nofile = True
                    continue
                all_saliencies[saliency_type][label].append(saliencies.mean(axis=0))

            if nofile:
                continue
            if args.clf_type == "subclf":
                break  # we only need to add one label per subject so we get out of the loop

        skip = False
        for option in saliency_options:
            if i == random_subject_idx:
                data_dict = get_saliency_data(sub_saliencies, option)
                for val in data_dict.values():
                    if val.size == 0:
                        skip = True
                        break
                temp = {
                    key: val[np.random.choice(np.arange(len(val)))]
                    for key, val in data_dict.items()
                }
                out_path = generate_saliency_figure(
                    temp,
                    info=load_info(),
                    save_path=visu_path,
                    suffix=f"{args.clf_type}_{sub}_single_trial_{option}",
                    sensors=sensors,
                    title=f"{option} saliencies for a single trial of subject {sub}",
                    clf_type=args.clf_type,
                    cmap=cmap,
                    stim_tick=stim_tick,
                )
                logging.info(f"Figure generated: {out_path}")
                temp = {key: np.mean(val, axis=0) for key, val in data_dict.items()}
                out_path = generate_saliency_figure(
                    temp,
                    info=load_info(),
                    save_path=visu_path,
                    suffix=f"{args.clf_type}_{sub}_all_trials_{option}",
                    sensors=sensors,
                    title=f"{option} saliencies for the averaged trials of subject {sub}",
                    clf_type=args.clf_type,
                    cmap=cmap,
                    stim_tick=stim_tick,
                )
                logging.info(f"Figure generated: {out_path}")
        if skip:
            random_subject_idx += 1
            continue

    for label in labels:
        for saliency_type in saliency_types:
            if type(all_saliencies[saliency_type][label]) == list:
                all_saliencies[saliency_type][label] = np.array(
                    all_saliencies[saliency_type][label]
                )
    for option in saliency_options:
        data_dict = get_saliency_data(all_saliencies, option)
        # np.newaxis here is a quick fix to a problem that might stick with other clf types
        final_dict = {key: np.mean(val, axis=0)[np.newaxis] for key, val in data_dict.items()}
        if args.clf_type == "subclf":
            final_dict = {"all_subj": np.mean(list(final_dict.values()), axis=0)}
            labels = ["all_subj"]

        out_path = generate_saliency_figure(
            final_dict,
            info=load_info(),
            save_path=visu_path,
            suffix=f"{args.clf_type}_{option}",
            sensors=sensors,
            title=f"{option} saliencies averaged across all subjects",
            clf_type=args.clf_type,
            cmap=cmap,
            stim_tick=stim_tick,
        )
        logging.info(f"Figure generated: {out_path}")
