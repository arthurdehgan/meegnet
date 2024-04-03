import os
import logging
from collections import defaultdict
import numpy as np
from meegnet.parsing import parser, save_config
from meegnet.viz import generate_saliency_figure
import mne
import seaborn as sns
import warnings

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

    args = parser.parse_args()
    logging.info(parser.format_values())
    save_config(vars(args), args.config)

    #################################################
    ### Create network name depending on the args ###
    #################################################

    if args.fold != -1:
        fold = args.fold + 1
    else:
        fold = 1
    name = f"{args.model_name}_{args.seed}_fold{fold}_{args.sensors}"

    suffixes = args.clf_type
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

    visu_path = os.path.join(args.save_path, "visualizations")
    if not os.path.exists(visu_path):
        os.makedirs(visu_path)
    sal_path = os.path.join(args.save_path, "saliency_maps", name)
    file_list = os.listdir(sal_path)

    if args.clf_type == "eventclf":
        labels = ["visual", "auditory1", "auditory2", "auditory3"]
    else:
        labels = []

    subjects = np.unique(
        [
            file.split("_")[0]
            for file in file_list
            if set(file.split("_")).intersection(set(labels)) != set()
        ]
    )[: args.max_subj]
    random_subject = np.random.randint(len(subjects))

    # TODO not working
    if args.clf_type == "subclf":
        labels = subjects

    all_saliencies = defaultdict(lambda: defaultdict(lambda: []))

    #########################
    ### HARD CODED VALUES ###
    #########################

    # TODO add those to a TOML file, either config or default_values
    sensors = ["MAG"]  # , "PLANNAR1", "PLANNAR2"]
    cmap = "coolwarm"
    stim_tick = 75
    saliency_types = ("pos", "neg")
    saliency_options = ("pos", "neg", "sum", "diff")

    # Some tested aletrnatives for the colormap:
    # cmap = sns.color_palette("icefire", as_cmap=True)
    # cmap = sns.color_palette("coolwarm", as_cmap=True, center="dark")
    # cmap = "inferno"
    # cmap = "seismic"

    ###############
    ### LOGGING ###
    ###############

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    logging.info(f"Generating figure for sensors: {sensors}")
    logging.info(f"For the {args.clf_type} classification")

    ######################################
    ### LOAD DATA AND GENERATE FIGURES ###
    ######################################

    # label contains same information as sub for subclf but we only load files that have label == sub
    for i, sub in enumerate(subjects):
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
                    print(saliency_file)
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
            if i == random_subject:
                data_dict = get_saliency_data(sub_saliencies, option)
                for val in data_dict.values():
                    if val.size == 0:
                        skip = True
                        break
                out_path = generate_saliency_figure(
                    {key: val[0] for key, val in data_dict.items()},
                    info_path=args.raw_path,
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
                    info_path=args.raw_path,
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
            random_subject += 1
            continue

    for label in labels:
        for saliency_type in saliency_types:
            if type(all_saliencies[saliency_type][label]) == list:
                all_saliencies[saliency_type][label] = np.concatenate(
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
            info_path=args.raw_path,
            save_path=visu_path,
            suffix=f"{args.clf_type}_{option}",
            sensors=sensors,
            title=f"{option} saliencies averaged across all subjects",
            clf_type=args.clf_type,
            cmap=cmap,
            stim_tick=stim_tick,
        )
        logging.info(f"Figure generated: {out_path}")
