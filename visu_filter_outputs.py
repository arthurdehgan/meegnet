"""In this script, we load a network and a camcan dataset example.

- We extract all the conv layers and their weights.
- We then generate a figure of topomaps for all the weights of layer 0 projected on the head using mne viz package

Next this script will also generate all the output of filters for a given input.
We will also perform spectral analysis on those.

"""
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from scipy.signal import welch
from parsing import parser
from params import TIME_TRIAL_LENGTH
from cnn import create_net
from utils import load_checkpoint
from dataloaders import create_datasets, create_loader, extract_bands
from viz import generate_topomap, load_info, GuidedBackprop, make_gif
from misc_functions import (
    save_gradient_images,
    convert_to_grayscale,
    get_positive_negative_saliency,
)

DEVICE = "cpu"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def load_data(args):
    # change path depending on the data type. TODO Hard coded
    # Add an option to change this
    data_path = args.path
    if args.dattype == "passive":
        data_path += "downsampled_500/"
    else:
        data_path += "downsamlped_200"

    # Using dataloaders in order to load data the same way
    # we did during training and testing
    datasets = create_datasets(
        data_path,
        args.train_size,
        args.max_subj,
        args.sensors,
        args.feature,
        seed=args.seed,
        printmem=args.printmem,
        n_samples=args.n_samples,
        dattype=args.dattype,
        load_events=args.eventclf,
    )
    dataloader = create_loader(
        datasets[0],
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
    )
    for X, y in dataloader:
        break

    return X, y


if __name__ == "__main__":
    parser.add_argument(
        "--topomaps",
        action="store_true",
        help="wether or not to generate topomaps for the spatial layer.",
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
        "--outputs",
        action="store_true",
        help="wether or not to generate the outputs of the conv layers.",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if notest option is true",
    )
    args = parser.parse_args()

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

    model_filepath = "models/" + name + ".pt"
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
        plt.savefig("figures/filter.png")

    #######################################################
    ### Generating visualization after each conv layer ####
    #######################################################

    if args.outputs:
        X, _ = load_data(args)

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
            print(layer_viz.size())
            for i, filt in enumerate(layer_viz):
                if i == 100:  # we will visualize only 10x10 blocks from each layer
                    break
                plt.subplot(10, 10, i + 1)
                plt.plot(np.arange(len(filt[0])), filt[0])
            print(f"Saving layer {num_layer} feature maps...")
            plt.savefig(f"figures/layer_{num_layer}.png")
            # plt.show()
            plt.close()

    ########################
    ### Backprop filters ###
    ########################

    if args.backprop:
        label_set = []
        args.seed += 27
        while len(label_set) < 2:
            X, y = load_data(args)
            args.seed += 1
            if y not in label_set:
                label_set.append(y)
                print(y)  # for eventclf 0 is image, audio is 1
                X.requires_grad_(True)

                GBP = GuidedBackprop(net)
                # Get gradients
                guided_grads = GBP.generate_gradients(X, y)
                target_name = "image" if y == 0 else "sound"
                file_name_to_export = f"figures/{target_name}"
                # Save colored gradients
                save_gradient_images(
                    guided_grads, file_name_to_export + "_Guided_BP_color"
                )
                # Convert to grayscale
                grayscale_guided_grads = convert_to_grayscale(guided_grads)
                # Save grayscale gradients
                save_gradient_images(
                    grayscale_guided_grads, file_name_to_export + "_Guided_BP_gray"
                )
                # Positive and negative saliency maps
                pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
                save_gradient_images(pos_sal, file_name_to_export + "_pos_sal")
                save_gradient_images(neg_sal, file_name_to_export + "_neg_sal")
                print("Guided backprop completed")

            if args.topograd:
                for i, chan in enumerate(("GRAD", "GRAD2", "MAG")):
                    imlist = []
                    data = guided_grads[i]
                    if args.spectral:
                        # TODO Dont hardcode sampling freq !
                        f, data = welch(data, fs=500)
                        data = extract_bands(data, f)
                    vmin, vmax = data.min(), data.max()
                    for timestamp in range(data.shape[-1]):
                        _ = generate_topomap(
                            data[:, timestamp], info, vmin=vmin, vmax=vmax
                        )
                        imname = (
                            f"figures/t{timestamp}_topograd_{target_name}_{chan}.png"
                        )
                        # we took data at 500Hz fr0m 150ms before stim to 650 after
                        if args.spectral:
                            text = BANDS[timestamp]
                        else:
                            text = "t=" + str(-150 + (timestamp * 2))
                        plt.text(
                            0.12,
                            0.13,
                            text,
                            fontsize=20,
                            color="black",
                        )
                        imlist.append(imname)
                        plt.axis("off")
                        plt.savefig(imname)
                        plt.close()
                    if not args.spectral:
                        make_gif(imlist)
