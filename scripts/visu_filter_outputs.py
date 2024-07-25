import os
import configparser
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
import logging
from mne.viz import plot_topomap
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet_functions import load_single_subject, prepare_logging, get_name, get_input_size, load_info


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
    CHANNELS = ("MAG", "GRAD", "GRAD2")

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

    input_size = get_input_size(args)
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
        prepare_logging("filter_computations", args, LOG, fold)

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

    info = load_info()

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
