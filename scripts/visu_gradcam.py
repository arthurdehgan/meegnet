import os
import toml
import logging
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from meegnet.parsing import parser, save_config
from meegnet.utils import load_checkpoint, cuda_check
from meegnet.network import create_net
from meegnet.dataloaders import load_data
from meegnet.viz import plot_masked_epoch
import cv2

# from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam import GradCAM

#     HiResCAM,
#     ScoreCAM,
#     GradCAMPlusPlus,
#     AblationCAM,
#     XGradCAM,
#     EigenCAM,
#     FullGrad,
# )
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = cuda_check()


if __name__ == "__main__":

    args = parser.parse_args()
    save_config(vars(args), args.config)
    with open("default_values.toml", "r") as f:
        default_values = toml.load(f)

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        log_name = args.model_name
        if args.fold != -1:
            log_name += f"_fold{args.fold}"
        log_name += "_saliency_computations.log"
        logging.basicConfig(
            filename=os.path.join(args.save_path, log_name),
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

    ###############################
    ### EXTRACTING PARSER INFO ###
    ###############################

    if args.clf_type == "eventclf":
        labels = [
            "visual",
            "auditory1",
            "auditory2",
            "auditory3",
        ]  # image is label 0 and sound label 1
    else:
        labels = []

    if args.feature == "bins":
        trial_length = default_values["TRIAL_LENGTH_BINS"]
    elif args.feature == "bands":
        trial_length = default_values["TRIAL_LENGTH_BANDS"]
    elif args.feature == "temporal":
        trial_length = default_values["TRIAL_LENGTH_TIME"]

    if args.sensors == "MAG":
        n_channels = default_values["N_CHANNELS_MAG"]
        chan_index = [0]
    elif args.sensors == "GRAD":
        n_channels = default_values["N_CHANNELS_GRAD"]
        chan_index = [1, 2]
    else:
        n_channels = default_values["N_CHANNELS_OTHER"]
        chan_index = [0, 1, 2]

    input_size = (n_channels // 102, 102, trial_length)

    if args.fold != -1:
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

    if args.clf_type == "subclf":
        n_outputs = min(643, args.max_subj)
    else:
        n_outputs = 2

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    viz_path = os.path.join(args.save_path, "visualizations")
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    #####################################
    ### LOADING NETWORK AND DATA INFO ###
    #####################################

    model_filepath = os.path.join(args.save_path, name + ".pt")
    net = create_net(args.net_option, name, input_size, n_outputs, DEVICE, args)
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    dataframe = (
        pd.read_csv(
            os.path.join(args.save_path, f"participants_info_{args.datatype}.csv"),
            index_col=0,
        )
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]
    n_sub = len(subj_list)

    #################
    ### MAIN LOOP ###
    #################

    for i, layer in enumerate(net.feature_extraction):
        target_layers = [layer]
        gradcam = GradCAM(model=net, target_layers=target_layers)

        all_cams = []
        all_trials = None
        for sub in subj_list:
            data, targets = load_data(
                dataframe.loc[dataframe["sub"] == sub],
                args.save_path,
                epoched=args.epoched,
                seed=args.seed,
                sfreq=args.sfreq,
                chan_index=chan_index,
                datatype=args.datatype,
                clf_type=args.clf_type,
                n_samples=None if args.n_samples == -1 else args.n_samples,
            )
            if data is None:
                logging.info(f"data from {sub} is empty.")
                continue
            input_tensor = data.to(torch.float)  # can be multiple images

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = gradcam(input_tensor=input_tensor, targets=None)

            # In this example grayscale_cam has only one image in the batch:
            all_cams.append(np.mean(grayscale_cam, 0))
            if all_trials is None:
                all_trials = data.mean(axis=0)[0] / n_sub
            else:
                all_trials += data.mean(axis=0)[0] / n_sub

        all_cams = np.array(all_cams)
        cams = np.mean(all_cams, axis=0)
        cams_img = cv2.merge([cams, cams, cams])

        img = Image.fromarray(np.uint8(255 * cams_img))
        img.save(os.path.join(viz_path, f"gradcam_masklayer{i}.png"))

        plot_masked_epoch(all_trials, cams)
        plt.savefig(os.path.join(viz_path, f"average_trial_layer{i}.png"), dpi=400)
