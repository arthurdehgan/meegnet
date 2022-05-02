import os
from collections.abc import Iterable
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
from joblib import Parallel, delayed, parallel_backend

# from scipy.signal import welch
from camcan.parsing import parser
from camcan.params import TIME_TRIAL_LENGTH
from camcan.utils import load_checkpoint, compute_psd
from camcan.dataloaders import (
    load_passive_sub_events,
    load_sub,
    BANDS,
)
from camcan.viz import generate_topomap, load_info, GuidedBackprop, make_gif
from camcan.misc_functions import (
    save_gradient_images,
    convert_to_grayscale,
    get_positive_negative_saliency,
)
from run import create_net

DEVICE = "cpu"
LABELS = ["image", "sound"]  # image is label 0 and sound label 1


def compute_the_good_stuff(
    net, trial, y, w_size, fs, use_windows=False, sal_option="pos"
):
    X = torch.Tensor(trial[np.newaxis])
    X.requires_grad_(True)
    # If confidence is good enough we use the trial for visualization
    confidence = torch.nn.Softmax(dim=1)(net(X)).max()
    chan_data = None
    if confidence >= 0.95:
        if use_windows:
            GBP = GuidedBackprop(net)
            guided_grads = GBP.generate_gradients(X, y)
            pos_saliency, neg_saliency = get_positive_negative_saliency(guided_grads)
            if sal_option == "pos":
                saliency = pos_saliency
            elif sal_option == "neg":
                saliency = neg_saliency
            else:
                saliency = pos_saliency + neg_saliency
            chan_data = []
            for chan in saliency:
                windows_idx = choose_best_window(chan)
                transformed_data = []
                for j, index in enumerate(windows_idx):
                    if isinstance(index, Iterable):
                        tmp = []
                        for idx in index:
                            tmp.append(
                                compute_psd(
                                    chan[j, idx : idx + w_size].reshape(1, w_size),
                                    fs=fs,
                                )
                            )
                        bands = np.mean(tmp, axis=0)
                    else:
                        bands = compute_psd(
                            chan[j, index : index + w_size].reshape(1, w_size),
                            fs=fs,
                        )
                    transformed_data.append(bands)
                chan_data.append(np.array(transformed_data).squeeze())
            chan_data = np.array(chan_data)
        else:
            chan_data = compute_psd(trial, fs=fs)
    return chan_data


def choose_best_window(data, fs=500, w_size=300):
    """
    data: array
        Must be of size k x n_samples. k can be sensor dimension or trial dimension.
    w_size: int
        The size of the window in ms
    """
    masks = [dat >= (np.mean(dat) + np.std(dat) / 2) for dat in data]
    w_size = int(w_size * fs / 1000)
    best_window_idx = []
    for mask in masks:
        windows = np.array([mask[i : i + w_size] for i in range(len(mask) - w_size)])
        values = [sum(window) for window in windows]
        best_window_index = np.where(values == max(values))[0]
        if len(best_window_index) > 1:
            idx_range = best_window_index[-1] - best_window_index[0]
            if idx_range <= 150 * fs / 1000:  # 150ms betweeen first and last window
                # Then best would be in the middle of all this
                best = int(len(best_window_index) / 2)
                best_window_idx.append(best_window_index[best])
            elif idx_range <= 300 * fs / 1000:  # 300ms
                best_window_idx.append((best_window_index[0], best_window_index[-1]))
            else:
                best = int(len(best_window_index) / 2)
                best_window_idx.append(
                    (best_window_index[0], best, best_window_index[-1])
                )
        else:
            best_window_idx.append(best_window_index[0])
    return best_window_idx


def compute_save_guided_bprop(net, X, y):
    GBP = GuidedBackprop(net)
    guided_grads = GBP.generate_gradients(X, y)
    target_name = "image" if y == 0 else "sound"
    file_name_to_export = f"../figures/{target_name}"
    save_gradient_images(guided_grads, file_name_to_export + "_Guided_BP_color")
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    save_gradient_images(
        grayscale_guided_grads, file_name_to_export + "_Guided_BP_gray"
    )
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + "_pos_sal")
    save_gradient_images(neg_sal, file_name_to_export + "_neg_sal")
    return guided_grads, target_name


def generate_topograd_map(guided_grads, target_name, spectral=False):
    for k, chan in enumerate(("GRAD", "GRAD2", "MAG")):
        imlist = []
        data = guided_grads[k]
        for step in range(data.shape[-1]):
            if spectral:
                text = BANDS[step]
                vmin, vmax = data[:, step].min(), data[:, step].max()
                pre = text
            else:
                text = "t=" + str(-150 + (step * 2))
                vmin, vmax = data.min(), data.max()
                pre = f"t{step}"

            plt.text(
                0.12,
                0.13,
                text,
                fontsize=20,
                color="black",
            )
            _ = generate_topomap(data[:, step], info, vmin=vmin, vmax=vmax)
            dpath = f"../figures/topograds/{chan}/{target_name}/"
            if not os.path.exists(dpath):
                os.makedirs(dpath)
            imname = dpath + f"{pre}_topograd_{target_name}_{chan}.png"
            imlist.append(imname)
            plt.axis("off")
            plt.savefig(imname)
            plt.close()
        if not spectral:
            make_gif(imlist)


def load_data(args, sub="CC321464"):
    # change path depending on the data type. TODO Hard coded
    # Add an option to change this
    data_path = args.path
    if args.dattype == "passive":
        data_path += "downsampled_500/"
        return load_passive_sub_events(data_path, sub)
    else:
        dataframe = pd.read_csv(f"{data_path}trials_df_clean.csv", index_col=0)
        data_path += "downsamlped_200"
        X = load_sub(data_path, sub)
        y = dataframe[dataframe["subs"] == sub]["sex"].iloc(0)

    return X, y


if __name__ == "__main__":
    parser.add_argument(
        "--use-windows",
        action="store_true",
        help="wether or not to use saliency windows generated by guided backprop.",
    )
    parser.add_argument(
        "--topomaps",
        action="store_true",
        help="wether or not to generate topomaps for the spatial layer.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Will generate all programmed visualizations.",
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="for topograd and backprop, if set will do everything in spectral domain after welch transform. if not set, temporal.",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        help="wether or not to generate the backprop filter.",
    )
    parser.add_argument(
        "--topograd",
        action="store_true",
        help="wether or not to generate the topomaps of the guided backprop.",
    )
    parser.add_argument(
        "--all-subj",
        action="store_true",
        help="compute for all subjs instead of just one",
    )
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="wether or not to generate the outputs of the conv layers.",
    )
    parser.add_argument(
        "--saliency",
        default="pos",
        choices=["pos", "neg", "both"],
        type=str,
        help="chooses whether to use positive saliency, negative saliency or the sum of them",
    )
    args = parser.parse_args()
    if args.all:
        args.topograd = True
        args.outputs = True

    if args.topograd:
        args.backprop = True
        args.spectral = True

    suffixes = ""
    if args.batchnorm:
        suffixes += "_BN"
    if args.maxpool != 0:
        suffixes += f"_maxpool{args.maxpool}"

    if args.feature == "bins":
        trial_length = 241
    if args.feature == "bands":
        trial_length = 5
    elif args.feature == "temporal":
        trial_length = TIME_TRIAL_LENGTH
    elif args.feature == "cov":
        # TODO
        pass
    elif args.feature == "cosp":
        # TODO
        pass

    if args.sensors == "MAG":
        n_channels = 102
    elif args.sensors == "GRAD":
        n_channels = 204
    elif args.sensors == "ALL":
        n_channels = 306

    input_size = (n_channels // 102, 102, trial_length)

    # WARNING: using an older version of networks: fold was saved from 0 to 4 instead of 1 to 5 !! TODO
    name = f"{args.model_name}_{args.seed}_fold{args.fold}_{args.sensors}_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
    name += suffixes

    if args.subclf:
        # TODO
        raise "not yet implemented for subclf"
    else:
        n_outputs = 2

    info = load_info(args.path)

    model_filepath = "../models/" + name + ".pt"
    net = create_net(args.net_option, name, input_size, n_outputs, args)
    epoch, net_state, optimizer_state = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)
    net.to(DEVICE)

    print(net)
    model_weights = []
    conv_layers = []
    model_children = list(net.children())

    counter = 0
    for seq in model_children:
        for layer in seq:
            if type(layer) == nn.Conv2d:
                counter += 1
                model_weights.append(layer.weight)
                conv_layers.append(layer)

    ##########################
    ### Genrating topomaps ###
    ##########################

    if args.topomaps:
        plt.figure(figsize=(20, 17))
        for i, filtr in enumerate(model_weights[0]):
            plt.subplot(10, 10, i + 1)
            _ = generate_topomap(filtr[0, :, :].detach(), info)
            plt.axis("off")
        plt.savefig("../figures/filter.png")

    #######################################################
    ### Generating visualization after each conv layer ####
    #######################################################

    if args.outputs:
        X = torch.Tensor(load_data(args)[0])

        # Generating outputs after forward pass of each conv layer
        # TODO check if we should add maxpooling to this, since it
        # is usually done during forward pass
        results = [conv_layers[0](X)]
        for i in range(1, len(conv_layers)):
            results.append(conv_layers[i](results[-1]))
        outputs = results

        # Generating the visualization of those inputs
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            for i, filt in enumerate(layer_viz):
                if i == 100:  # we will visualize only 10x10 blocks from each layer
                    break
                plt.subplot(10, 10, i + 1)
                plt.plot(np.arange(len(filt[0])), filt[0])
            print(f"Saving layer {num_layer} feature maps...")
            plt.savefig(f"../figures/layer_{num_layer}.png")
            # plt.show()
            plt.close()

    ########################
    ### Backprop filters ###
    ########################

    if args.topograd:
        if args.all_subj:
            dataframe = pd.read_csv(
                f"{args.path}clean_participant_new.csv", index_col=0
            )
            subj_list = dataframe["participant_id"]
            subj_list = subj_list.sample(frac=1).reset_index(drop=True)
            if args.max_subj is not None:
                subj_list = subj_list.loc[: args.max_subj]
        else:
            subj_list = ["CC510433"]
        bands_values = [[], []]
        w_size = 300  # Parameter TODO put somewhere else when factoring in fucntions
        fs = 500  # Parameter TODO put somewhere else when factoring in fucntions
        w_size = int(w_size * fs / 1000)
        files = [f"../data/psd_{lab}.npy" for lab in LABELS]

        if all(os.path.exists(file) for file in files) and not args.use_windows:
            for i, file in enumerate(files):
                bands_values[i] = np.load(file)
        else:
            for sub in subj_list:
                examples, targets = load_data(args, sub)
                if targets is None:
                    continue
                print(sub)
                targets = np.array(targets)
                examples = np.array(examples)
                for targ in np.unique(targets):
                    chan_data = Parallel(n_jobs=-1)(
                        delayed(compute_the_good_stuff)(
                            net,
                            trial,
                            y,
                            w_size,
                            fs,
                            use_windows=args.use_windows,
                            sal_option=args.saliency,
                        )
                        for trial, y in zip(
                            examples[targets == targ], targets[targets == targ]
                        )
                    )
                    chan_data = [e for e in chan_data if e is not None]
                    bands_values[targ] += chan_data
            print(
                f"used {len(bands_values[0])} image trials, and {len(bands_values[1])} sound trials."
            )
            name = "../data/psd"
            name = (
                name + f"_{args.saliency}_saliency_windows"
                if args.use_windows
                else name
            )
            for i, bv in enumerate(bands_values):
                np.save(name + f"_{LABELS[i]}.npy", bv)
            bands_values = [np.array(bv).mean(axis=0) for bv in bands_values]

        for i, bv in enumerate(bands_values):
            if not args.use_windows:
                tname = "psd_" + LABELS[i]
            else:
                tname = LABELS[i]
            tname += f"_{args.saliency}_saliency"
            generate_topograd_map(bv, tname, spectral=args.spectral)
