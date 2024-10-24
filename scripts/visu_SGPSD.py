import os
import toml
from collections.abc import Iterable
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from torch import nn
import logging
from mne.viz import plot_topomap

# from scipy.signal import welch
from meegnet_functions import load_single_subject
from meegnet.parsing import parser, save_config
from meegnet.utils import load_checkpoint, compute_psd, cuda_check
from meegnet.dataloaders import BANDS, load_data
from pytorch_grad_cam import GuidedBackpropReLUModel
from meegnet.viz import make_gif
from meegnet.viz import (
    save_gradient_images,
    convert_to_grayscale,
    get_positive_negative_saliency,
)
from meegnet.network import create_net
from meegnet_functions import load_info


LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

LOG.warning("This script is deprecated, prefer using the jupyter notebook instead")

def compute_save_guided_bprop(net, X, y):
    GBP = GuidedBackpropReLUModel(net)
    guided_grads = GBP.generate_gradients(X, y)
    target_name = "image" if y == 0 else "sound"
    file_name_to_export = f"../figures/{target_name}"
    save_gradient_images(guided_grads, file_name_to_export + "_Guided_BP_color")
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    save_gradient_images(grayscale_guided_grads, file_name_to_export + "_Guided_BP_gray")
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + "_pos_sal")
    save_gradient_images(neg_sal, file_name_to_export + "_neg_sal")
    return guided_grads, target_name


def generate_topograd_map(
    guided_grads, target_name, info, cmap="viridis", viz_path=".", spectral=False
):
    for k, chan in enumerate(CHANNELS):
        imlist = []
        data = guided_grads[k] if len(guided_grads.shape) == 3 else guided_grads
        for step in range(data.shape[-1]):
            if spectral:
                text = BANDS[step]
                # vmin, vmax = data[:, step].min(), data[:, step].max()
                pre = text
            else:
                text = "t=" + str(-150 + (step * 2))
                # vmin, vmax = data.min(), data.max()
                pre = f"t{step}"

            plt.text(
                0.12,
                0.13,
                text,
                fontsize=20,
                color="black",
            )
            _, _ = plot_topomap(data[:, step], info, cmap=cmap, show=False, contours=0)
            data_path = os.path.join(viz_path, chan, target_name)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            file_path = os.path.join(data_path, f"{pre}_topograd_{target_name}_{chan}.png")
            imlist.append(file_path)
            plt.axis("off")
            plt.savefig(file_path)
            plt.close()
        if not spectral:
            make_gif(imlist)


if __name__ == "__main__":

    ### HARD CODED VALUES ###
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

    )
    save_config(vars(args), args.config)
    with open("default_values.toml", "r") as f:
        default_values = toml.load(f)

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

    model_weights = my_model.get_feature_weights()

    #############################################
    ### Genrating feature importance topomaps ###
    #############################################

    info = load_info()

    plt.figure(figsize=(20, 17))
    for i, filtr in enumerate(model_weights[0]):
        plt.subplot(10, 10, i + 1)
        im, _ = plot_topomap(
            filtr[0, :, :].detach().cpu().ravel(),
            info,
            res=128,
            show=False,
            contours=1,
            extrapolate="local",
        )
        plt.axis("off")
    plt.savefig(os.path.join(viz_path, name, "filters.png"))
    plt.close()

    #######################################################
    ### Generating visualization after each conv layer ####
    #######################################################

    dataframe = (
        pd.read_csv(
            os.path.join(args.save_path, f"participants_info.csv"),
            index_col=0,
        )
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    # Taking first subject form list will be random since the dataframe has been rd sampled
    i = 0
    data = []
    # Incrementing and changing subject in case there is an error with loading subject data
    while data == []:
        sub = subj_list[i]
        data = load_single_subject(sub, n_samples, lso, args).data
        i += 1

    logging.info(f"loading random subject {sub}")
    input_tensor = data.to(torch.float)  # can be multiple images

    np.random.seed(args.seed)
    random_sample = np.random.choice(input_tensor)[np.newaxis, :]

    results = [my_model.net.feature_extraction[0](random_sample)]
    for layer in my_model.net.feature_extraction[1:]:
        results.append(layer[i](results[-1]))
    for layer in my_model.net.classif[1:]:
        results.append(layer[i](results[-1]))
    outputs = results

    #############################
    ### FILTER VISUALISATIONS ###
    #############################

    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer]
        layer_viz = layer_viz.data.cpu()
        for i, filt in enumerate(layer_viz):
            if i == 100:  # we will visualize only 10x10 blocks from each layer
                break
            plt.subplot(10, 10, i + 1)
            plt.plot(np.arange(len(filt[0])), filt[0])
        logging.info(f"Saving layer {num_layer} feature maps...")
        plt.savefig(os.path.join(viz_path, name, f"layer_{num_layer}.png"))
        plt.close()

    ##########################
    ### LOADING SALIENCIES ###
    ##########################

    # Always compute for both pos and neg saliencies, disregarding the args.saliency_type value
    saliency_types = ("pos", "neg")

    # all_options_bv are embedded lists with dimenstions:
    # saliencies x labels x subjects x sensors x frequency bands
    all_options_bv = []

    for saliency_type in saliency_types:
        save_path = os.path.join(
            args.save_path,
            "saliency_based_psd",
            name,
            saliency_type + f"_{args.confidence}confidence",
        )
        files = os.listdir(save_path)
        bands_values = [[] for _ in labels]
        for file in files:
            bands_values
            file_lab = file.split(".")[0].split("_")[-1]
            lab_idx = labels.index(file_lab)
            bv = np.load(os.path.join(save_path, file))
            bands_values[lab_idx].append(np.array(bv))
        for i, lab in enumerate(labels):
            bands_values[i] = np.array(np.mean(bands_values[i], axis=0))
            tname = f"All_Subjects_{saliency_type}_{lab}"
            generate_topograd_map(
                bands_values[i], tname, info, viz_path=viz_path, spectral=True
            )
        all_options_bv.append(bands_values)
    all_options_bv = np.array(all_options_bv)

    ######################################
    ### GENERATING SALIENCY-BASED PSDs ###
    ######################################

    print("psd-based saliencies")
    for k, band in enumerate(BANDS):
        for i, chan in enumerate(CHANNELS):
            grid = GridSpec(len(labels) + 1, 4)
            fig = plt.figure()
            # fig.clf()
            axes = []
            for j, sal in enumerate(("pos", "neg", "diff", "sum")):
                if sal in ("pos", "neg"):
                    data = all_options_bv[j]
                elif sal == "diff":
                    data = all_options_bv[0] - all_options_bv[1]
                else:
                    data = all_options_bv[0] + all_options_bv[1]

                for l, label in enumerate(labels):
                    axes.append(fig.add_subplot(grid[l, j]))
                    if len(CHANNELS) > 1:
                        topo_data = data[l][i][:, k]
                    else:
                        topo_data = data[l][:, k]
                    im, _ = plot_topomap(
                        topo_data,
                        info,
                        show=False,
                        contours=0,
                        res=300,
                        cmap="viridis",
                        axes=axes[-1],
                    )

                    if l == 0:
                        plt.title(sal)
                    if j == 0:
                        axes[-1].text(-0.25, 0, label, va="center", rotation="vertical")

            figure_path = os.path.join(viz_path, f"saliency_guided_{band}_psd.png")
            plt.savefig(figure_path, dpi=300)
            plt.show()
            plt.close()
