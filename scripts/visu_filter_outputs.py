import os
import toml
from collections.abc import Iterable
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import logging
from mne.viz import plot_topomap
from meegnet_functions import load_single_subject
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.viz import load_info


LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


if __name__ == "__main__":

    #########################
    ### HARD CODED VALUES ###
    #########################

    LABELS = ["visual", "auditory"]  # image is label 0 and sound label 1
    CHANNELS = ("GRAD", "GRAD2", "MAG")

    ###############
    ### PARSING ###
    ###############

    args = parser.parse_args()
    save_config(vars(args), args.config)
    with open("default_values.toml", "r") as f:
        default_values = toml.load(f)

    fold = None if args.fold == -1 else int(args.fold)
    if args.clf_type == "eventclf":
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run event classification"

    if args.feature == "bins":
        trial_length = default_values["TRIAL_LENGTH_BINS"]
    elif args.feature == "bands":
        trial_length = default_values["TRIAL_LENGTH_BANDS"]
    elif args.feature == "temporal":
        trial_length = default_values["TRIAL_LENGTH_TIME"]

    if args.clf_type == "subclf":
        trial_length = int(args.segment_length * args.sfreq)

    if args.sensors == "MAG":
        n_channels = default_values["N_CHANNELS_MAG"]
    elif args.sensors == "GRAD":
        n_channels = default_values["N_CHANNELS_GRAD"]
    else:
        n_channels = default_values["N_CHANNELS_OTHER"]

    input_size = (
        (1, n_channels, trial_length)
        if args.flat
        else (
            n_channels // default_values["N_CHANNELS_MAG"],
            default_values["N_CHANNELS_MAG"],
            trial_length,
        )
    )

    name = f"{args.clf_type}_{args.model_name}_{args.seed}_{args.sensors}"
    suffixes = ""
    if args.net_option == "custom_net":
        if args.batchnorm:
            suffixes += "_BN"
        if args.maxpool != 0:
            suffixes += f"_maxpool{args.maxpool}"

        name += f"_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
        name += suffixes

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
        log_name = f"{args.model_name}_{args.seed}_{args.sensors}"
        if fold is not None:
            log_name += f"_fold{args.fold}"
        log_name += "_filter_computations.log"
        log_file = os.path.join(args.save_path, log_name)
        logging.basicConfig(filename=log_file, filemode="a")
        LOG.info(f"Starting logging in {log_file}")

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    viz_path = os.path.join(args.save_path, "visualizations")
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    netviz_path = os.path.join(viz_path, name)
    if not os.path.exists(netviz_path):
        os.makedirs(netviz_path)

    #####################
    ### LOADING MODEL ###
    #####################

    if args.model_path is None:
        model_path = args.save_path
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        logging.info(f"{model_path} does not exist. Creating folders")
        os.makedirs(model_path)

    my_model = Model(name, args.net_option, input_size, n_outputs, save_path=args.save_path)
    my_model.from_pretrained()
    # my_model.load()

    model_weights = my_model.feature_weights

    #############################################
    ### Genrating feature importance topomaps ###
    #############################################

    info = load_info(args.raw_path, args.datatype)

    plt.figure(figsize=(20, 17))
    for i, filtr in enumerate(model_weights[0]):
        plt.subplot(10, 10, i + 1)
        im, _ = plot_topomap(
            filtr[0, :, :].ravel(),
            info,
            res=128,
            show=False,
            contours=1,
            extrapolate="local",
        )
        plt.axis("off")
    plt.savefig(os.path.join(viz_path, name, "filters.png"))
    plt.close()

    ######################
    ### FILTER OUTPUTS ###
    ######################

    dataframe = (
        pd.read_csv(
            os.path.join(args.save_path, f"participants_info.csv"),
            index_col=0,
        )
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    # Incrementing and changing subject in case there is an error with loading subject data
    data = []
    while data == []:
        data = load_single_subject("random", n_samples, lso, args).data

    input_tensor = data.to(torch.float).cpu()

    np.random.seed(args.seed)
    random_sample = input_tensor[np.random.choice(np.arange(len(input_tensor)))][np.newaxis, :]
    random_sample = random_sample.cuda()

    results = [my_model.net.feature_extraction[0](random_sample)]
    for layer in my_model.net.feature_extraction[1:]:
        results.append(layer(results[-1]))
    outputs = results

    #############################
    ### FILTER VISUALISATIONS ###
    #############################

    for layer_idx, out in enumerate(outputs):
        plt.figure(figsize=(30, 30))
        layer_viz = out[0].data.cpu()
        # If layer is flatten or FC layer, skip it
        if len(layer_viz.shape) <= 1:
            continue
        for i, filt in enumerate(layer_viz):
            if i == 100:  # we will visualize only 10x10 blocks from each layer
                break
            plt.subplot(10, 10, i + 1)
            plt.plot(np.arange(len(filt[0])), filt[0])
        logging.info(f"Saving layer {layer_idx} feature maps...")
        plt.savefig(os.path.join(viz_path, name, f"layer_{layer_idx}.png"))
        plt.close()
