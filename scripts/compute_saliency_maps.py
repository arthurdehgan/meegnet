import threading
import multiprocessing
import os
import logging
import toml
import torch
import numpy as np
import pandas as pd
from meegnet.parsing import parser, save_config
from meegnet.utils import load_checkpoint, cuda_check
from meegnet.network import create_net
from meegnet.dataloaders import load_data
from meegnet.util_viz import (
    get_positive_negative_saliency,
    compute_saliency_based_psd,
)
from pytorch_grad_cam import GuidedBackpropReLUModel


DEVICE = cuda_check()


def produce_data(queue, sub, args, disk_semaphore):
    with disk_semaphore:
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
            queue.put(tuple([None] * 3 + [0]))
            return
        else:
            queue.put((data, targets, sub, 1))
            return


def process_data(arguments):
    queue, labels, sal_path, net, args = arguments
    while True:
        data, targets, sub, stop = queue.get()
        if stop is None:
            break
        if data is None:
            continue
        compute_saliency_maps(
            data,
            targets,
            labels,
            sub,
            sal_path,
            net,
            args.confidence,
            args.w_size,
            args.sfreq,
            args.clf_type,
            args.compute_psd,
        )


def compute_saliency_maps(
    data,
    targets,
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
    GBP = GuidedBackpropReLUModel(net, device=DEVICE)

    # Load all trials and corresponding labels for a specific subject.
    if clf_type == "eventclf":
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
        if clf_type == "subclf":
            label = int(dataframe[dataframe["sub"] == sub].index[0])

        # If the confidence reaches desired treshhold (given by args.confidence)
        if confidence >= threshold and pred == label:
            # Compute Guided Back-propagation for given label projected on given data X
            guided_grads = GBP(X.to(DEVICE), label)
            guided_grads = np.rollaxis(guided_grads, 2, 0)
            # Compute saliencies
            pos_saliency, neg_saliency = get_positive_negative_saliency(guided_grads)

            # Depending on the task, add saliencies in lists
            if clf_type == "eventclf":
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
            filename=log_name,
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

    if args.clf_type == "eventclf":
        labels = [
            "visual",
            "auditory1",
            "auditory2",
            "auditory3",
        ]  # image is label 0 and sound label 1
    elif args.clf_type == "subclf":
        labels = []
    else:
        labels = ["male", "female"]

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

    if args.model_path is None:
        model_path = args.save_path
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        logging.info(f"{model_path} does not exist. Creating folders")
        os.makedirs(model_path)

    model_filepath = os.path.join(model_path, name + ".pt")
    net = create_net(args.net_option, name, input_size, n_outputs, DEVICE, args)
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    csv_file = os.path.join(args.save_path, f"participants_info_{args.datatype}.csv")
    dataframe = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    #######################
    ### PRODUCE CONSUME ###
    #######################

    # Setting start method to spawn for CUDA compatibility.
    multiprocessing.set_start_method("spawn")
    # Define the maximum number of threads that can read from the disk at once
    MAX_DISK_READERS = 1
    # Define the maximum size of the queue
    MAX_QUEUE_SIZE = 20  # Adjust this value based on your memory constraints
    NUM_CONSUMERS = 6

    # Create a bounded queue with the maximum size
    queue = multiprocessing.Manager().Queue(maxsize=MAX_QUEUE_SIZE)
    # Create a semaphore with the maximum number of readers
    disk_semaphore = threading.Semaphore(MAX_DISK_READERS)

    # Start the producer threads
    threads = []
    for sub in subj_list:
        t = threading.Thread(
            target=produce_data,
            args=(queue, sub, args, disk_semaphore),
        )
        t.start()
        threads.append(t)

    # Start the consumer processes
    with multiprocessing.Pool(processes=NUM_CONSUMERS) as pool:
        pool.map(
            process_data,
            [(queue, labels, sal_path, net, args)] * NUM_CONSUMERS,
        )

    # Wait for all producer threads to finish
    for t in threads:
        t.join()

    # Signal the consumer processes to exit
    for _ in range(NUM_CONSUMERS):
        queue.put(tuple([None] * 4))

    pool.close()
    pool.join()
