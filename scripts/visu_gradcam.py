import os
import logging
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.viz import plot_masked_epoch
from meegnet_functions import load_single_subject, prepare_logging, get_input_size, get_name


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

LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

if __name__ == "__main__":

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
        prepare_logging("gradcam_computations", args, LOG, fold)

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    viz_path = os.path.join(args.save_path, "visualizations")
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

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
    net = my_model.net
    # my_model.load()

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
            data = load_single_subject(sub, n_samples, lso, args).data
            if data == []:
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
