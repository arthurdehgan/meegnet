import os
import logging
import toml
import torch
import numpy as np
import pandas as pd
from meegnet.parsing import parser
from meegnet.params import TIME_TRIAL_LENGTH
from meegnet.utils import load_checkpoint, cuda_check
from meegnet.network import create_net
from meegnet.dataloaders import load_data
from meegnet.util_viz import (
    get_positive_negative_saliency,
    compute_saliency_based_psd,
)
from pytorch_grad_cam import GuidedBackpropReLUModel

# from joblib import Parallel, delayed

DEVICE = cuda_check()


def compute_saliency_maps(
    data,
    targets,
    sal_path,
    GBP,
    threshold,
    w_size,
    sfreq,
    eventclf=False,
    subclf=False,
    compute_psd=False,
):
    # existing_paths = []
    # for j, sal_type in enumerate(("pos", "neg")):
    #     if not args.eventclf:
    #         for label in labels:
    #             lab = "" if args.subclf else f"_{label}"
    #             sal_filepath = os.path.join(
    #                 sal_path,
    #                 f"{sub}{lab}_{sal_type}_sal_{args.confidence}confidence.npy",
    #             )
    #             existing_paths.append(os.path.exists(sal_filepath))
    #     else:
    #         for i, label in enumerate(labels):
    #             sal_filepath = os.path.join(
    #                 sal_path,
    #                 f"{sub}_{labels[i]}_{sal_type}_sal_{args.confidence}confidence.npy",
    #             )
    #             existing_paths.append(os.path.exists(sal_filepath))
    # if all(existing_paths):
    #     return

    # Load all trials and corresponding labels for a specific subject.
    if eventclf:
        target_saliencies = [[[], []], [[], []]]
        target_psd = [[[], []], [[], []]]
    else:
        target_saliencies = [[], []]
        target_psd = [[], []]

    # For each of those trial with associated label:
    for trial, label in zip(data, targets):
        X = trial[np.newaxis].type(torch.FloatTensor).to(DEVICE)
        if len(X.shape) < 4:
            X = X[np.newaxis, :]
        # Compute predictions of the trained network, and confidence
        preds = torch.nn.Softmax(dim=1)(net(X)).detach().cpu()
        pred = preds.argmax().item()
        confidence = preds.max()
        if subclf:
            label = int(dataframe[dataframe["sub"] == sub].index[0])

        # If the confidence reaches desired treshhold (given by args.confidence)
        if confidence >= threshold and pred == label:
            # Compute Guided Back-propagation for given label projected on given data X
            guided_grads = GBP(X.to(DEVICE), label)
            guided_grads = np.rollaxis(guided_grads, 2, 0)
            # Compute saliencies
            pos_saliency, neg_saliency = get_positive_negative_saliency(guided_grads)

            # Depending on the task, add saliencies in lists
            if eventclf:
                target_saliencies[label][0].append(pos_saliency)
                target_saliencies[label][1].append(neg_saliency)
                if compute_psd:
                    target_psd[label][0].append(
                        compute_saliency_based_psd(pos_saliency, trial, w_size, sfreq)
                    )
                    target_psd[label][1].append(
                        compute_saliency_based_psd(neg_saliency, trial, w_size, sfreq)
                    )
            else:
                target_saliencies[0].append(pos_saliency)
                target_saliencies[1].append(neg_saliency)
                if compute_psd:
                    target_psd[0].append(
                        compute_saliency_based_psd(pos_saliency, trial, w_size, sfreq)
                    )
                    target_psd[1].append(
                        compute_saliency_based_psd(neg_saliency, trial, w_size, sfreq)
                    )
    # With all saliencies computed, we save them in the specified save-path
    for j, sal_type in enumerate(("pos", "neg")):
        if eventclf:
            for i, label in enumerate(labels):
                sal_filepath = os.path.join(
                    sal_path,
                    f"{sub}_{labels[i]}_{sal_type}_sal_{threshold}confidence.npy",
                )
                np.save(sal_filepath, np.array(target_saliencies[i][j]))
                if compute_psd:
                    psd_filepath = os.path.join(
                        psd_path,
                        f"{sub}_{labels[i]}_{sal_type}_psd_{threshold}confidence.npy",
                    )
                    np.save(psd_filepath, np.array(target_psd[i][j]))
        else:
            lab = "" if subclf else f"_{labels[label]}"
            sal_filepath = os.path.join(
                sal_path,
                f"{sub}{lab}_{sal_type}_sal_{threshold}confidence.npy",
            )
            np.save(sal_filepath, np.array(target_saliencies[j]))
            if compute_psd:
                lab = "" if subclf else f"_{labels[label]}"
                psd_filepath = os.path.join(
                    psd_path,
                    f"{sub}{lab}_{sal_type}_psd_{threshold}confidence.npy",
                )
                np.save(psd_filepath, np.array(target_psd[j]))


if __name__ == "__main__":

    args = parser.parse_args()
    args_dict = vars(args)
    toml_string = toml.dumps(args_dict)
    with open(args.config, "w") as toml_file:
        toml.dump(args_dict, toml_file)

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        log_name = args.model_name
        if args.fold is not None:
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
    ### TRANSLATING PARSER INFO ###
    ###############################

    if args.eventclf:
        labels = ["visual", "auditory"]  # image is label 0 and sound label 1
    elif args.subclf:
        labels = []
    else:
        labels = ["male", "female"]

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
        chan_index = 0
    elif args.sensors == "GRAD":
        n_channels = 204
        chan_index = [1, 2]
    elif args.sensors == "ALL":
        n_channels = 306
        chan_index = [0, 1, 2]

    input_size = (n_channels // 102, 102, trial_length)

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

    if args.subclf:
        n_outputs = min(643, args.max_subj)
    else:
        n_outputs = 2

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    if args.compute_psd:
        psd_path = os.path.join(args.save_path, "saliency_based_psds", name)
        if not os.path.exists(psd_path):
            os.makedirs(psd_path)

    sal_path = os.path.join(args.save_path, "saliency_maps", name)
    if not os.path.exists(sal_path):
        os.makedirs(sal_path)

    #####################################
    ### LOADING NETWORK AND DATA INFO ###
    #####################################

    model_filepath = os.path.join(args.save_path, name + ".pt")
    net = create_net(args.net_option, name, input_size, n_outputs, DEVICE, args)
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    folder = "psd" if args.feature in ["bins", "bands"] else f"downsampled_{args.sfreq}"
    csv_file = os.path.join(args.data_path, folder, f"participants_info_{args.datatype}.csv")
    dataframe = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    #################
    ### MAIN LOOP ###
    #################

    GBP = GuidedBackpropReLUModel(net, device=DEVICE)
    for i, sub in enumerate(subj_list):
        data, targets = load_data(
            dataframe.loc[dataframe["sub"] == sub],
            args.data_path,
            epoched=args.epoched,
            seed=args.seed,
            sfreq=args.sfreq,
            chan_index=chan_index,
            datatype=args.datatype,
            eventclf=args.eventclf,
        )
        if data is None or targets is None:
            continue
        compute_saliency_maps(
            data,
            targets,
            sal_path,
            GBP,
            args.confidence,
            args.w_size,
            args.sfreq,
            eventclf=args.eventclf,
            subclf=args.subclf,
            compute_psd=args.compute_psd,
        )
