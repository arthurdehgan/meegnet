import os
import logging
import configparser
import torch
import numpy as np
import pandas as pd
from meegnet_functions import load_single_subject
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.utils import cuda_check
from meegnet.viz import (
    get_positive_negative_saliency,
    compute_saliency_based_psd,
)
from pytorch_grad_cam import GuidedBackpropReLUModel


LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


def compute_saliency_maps(
    dataset,
    labels,
    sub,
    sal_path,
    net,
    threshold,
    w_size,
    sfreq,
    clf_type="",
    compute_psd=False,
):

    device = cuda_check()
    GBP = GuidedBackpropReLUModel(net, device=device)

    # Load all trials and corresponding labels for a specific subject.
    data = dataset.data
    targets = dataset.labels
    if clf_type == "eventclf":
        target_saliencies = [[[], []], [[], []]]
        target_psd = [[[], []], [[], []]]
    else:
        target_saliencies = [[], []]
        target_psd = [[], []]

    # For each of those trial with associated label:
    for trial, label in zip(data, targets):
        X = trial
        while len(X.shape) < 4:
            X = X[np.newaxis, :]
        X = X.to(device)
        # Compute predictions of the trained network, and confidence
        preds = torch.nn.Softmax(dim=1)(net(X)).detach().cpu()
        pred = preds.argmax().item()
        confidence = preds.max()
        label = int(label)

        # If the confidence reaches desired treshhold (given by args.confidence)
        if confidence >= threshold and pred == label:
            # Compute Guided Back-propagation for given label projected on given data X
            guided_grads = GBP(X.to(device), label)
            guided_grads = np.rollaxis(guided_grads, 2, 0)
            # Compute saliencies
            pos_saliency, neg_saliency = get_positive_negative_saliency(guided_grads)

            # Depending on the task, add saliencies in lists
            if clf_type == "eventclf":
                target_saliencies[label][0].append(pos_saliency)
                target_saliencies[label][1].append(neg_saliency)
                if compute_psd:
                    target_psd[label][0].append(
                        compute_saliency_based_psd(pos_saliency, X, w_size, sfreq)
                    )
                    target_psd[label][1].append(
                        compute_saliency_based_psd(neg_saliency, X, w_size, sfreq)
                    )
            else:
                target_saliencies[0].append(pos_saliency)
                target_saliencies[1].append(neg_saliency)
                if compute_psd:
                    target_psd[0].append(
                        compute_saliency_based_psd(pos_saliency, X, w_size, sfreq)
                    )
                    target_psd[1].append(
                        compute_saliency_based_psd(neg_saliency, X, w_size, sfreq)
                    )
    # With all saliencies computed, we save them in the specified save-path
    n_saliencies = 0
    n_saliencies += sum([len(e) for e in target_saliencies[0]])
    n_saliencies += sum([len(e) for e in target_saliencies[1]])
    LOG.info(f"{n_saliencies} saliency maps computed for {sub}")
    for j, sal_type in enumerate(("pos", "neg")):
        if clf_type == "eventclf":
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
            lab = "" if clf_type == "subclf" else f"_{labels[label]}"
            sal_filepath = os.path.join(
                sal_path,
                f"{sub}{lab}_{sal_type}_sal_{threshold}confidence.npy",
            )
            np.save(sal_filepath, np.array(target_saliencies[j]))
            if compute_psd:
                lab = "" if clf_type == "subclf" else f"_{labels[label]}"
                psd_filepath = os.path.join(
                    psd_path,
                    f"{sub}{lab}_{sal_type}_psd_{threshold}confidence.npy",
                )
                np.save(psd_filepath, np.array(target_psd[j]))


if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    args = parser.parse_args()
    save_config(vars(args), args.config)
    default_values = configparser.ConfigParser()
    default_values.read("../default_values.ini")
    default_values = default_values["config"]

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
    if args.clf_type == "eventclf":
        labels = ["visual", "auditory"]  # image is label 0 and sound label 1
    elif args.clf_type == "subclf":
        labels = []

    if args.sensors == "MAG":
        n_channels = default_values["N_CHANNELS_MAG"]
    elif args.sensors == "GRAD":
        n_channels = default_values["N_CHANNELS_GRAD"]
    else:
        n_channels = default_values["N_CHANNELS_OTHER"]

    input_size = (n_channels // 102, 102, trial_length)

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
        log_name = f"saliencies_{args.model_name}_{args.seed}_{args.sensors}"
        if fold is not None:
            log_name += f"_fold{args.fold}"
        log_name += ".log"
        log_file = os.path.join(args.save_path, log_name)
        logging.basicConfig(filename=log_file, filemode="a")
        LOG.info(f"Starting logging in {log_file}")

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

    #####################
    ### LOADING MODEL ###
    #####################

    if args.model_path is None:
        model_path = args.save_path
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        LOG.info(f"{model_path} does not exist. Creating folders")
        os.makedirs(model_path)

    my_model = Model(name, args.net_option, input_size, n_outputs, save_path=args.save_path)
    my_model.from_pretrained()
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

    #################
    ### MAIN LOOP ###
    #################

    for sub in subj_list:
        dataset = load_single_subject(sub, n_samples, lso, args)
        compute_saliency_maps(
            dataset,
            labels,
            sub,
            sal_path,
            my_model.net,
            args.confidence,
            args.w_size,
            args.sfreq,
            args.clf_type,
            args.compute_psd,
        )
