import threading
import multiprocessing
import os
import logging
import toml
import torch
import numpy as np
import pandas as pd
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.dataloaders import Dataset, RestDataset
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


def produce_data(queue, sub, args, disk_semaphore):
    with disk_semaphore:
        if args.datatype == "rest":
            dataset = RestDataset(
                window=args.segment_length,
                overlap=args.overlap,
                sfreq=args.sfreq,
                n_subjects=args.max_subj,
                n_samples=n_samples,
                sensortype=args.sensors,
                lso=lso,
                random_state=args.seed,
            )
        else:
            dataset = Dataset(
                sfreq=args.sfreq,
                n_subjects=args.max_subj,
                n_samples=n_samples,
                sensortype=args.sensors,
                lso=lso,
                random_state=args.seed,
            )
        dataset.load(args.save_path, one_sub=sub)
        if len(dataset) == 0:
            logging.info(f"data from {sub} is empty.")
            queue.put(tuple([None] * 2 + [0]))
            return
        else:
            queue.put((dataset, sub, 1))
            return


def process_data(arguments):
    queue, labels, sal_path, net, args = arguments
    while True:
        dataset, sub, stop = queue.get()
        if stop is None:
            break
        if dataset is None:
            continue
        if len(dataset) == 0:
            continue
        compute_saliency_maps(
            dataset,
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

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    GBP = GuidedBackpropReLUModel(net, device=DEVICE)

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
        X = X.to(DEVICE)
        # Compute predictions of the trained network, and confidence
        preds = torch.nn.Softmax(dim=1)(net(X)).detach().cpu()
        pred = preds.argmax().item()
        confidence = preds.max()
        label = int(label)

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
    logging.info(f"{n_saliencies} saliency maps computed for {sub}")
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
    with open("default_values.toml", "r") as f:
        default_values = toml.load(f)

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

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        log_name = f"saliencies_{args.model_name}_{args.seed}_{args.sensors}"
        log_name += ".log"
        log_file = os.path.join(args.save_path, log_name)
        logging.basicConfig(filename=log_file, filemode="a")
        LOG.info(f"Starting logging in {log_file}")

    ###############################
    ### TRANSLATING PARSER INFO ###
    ###############################

    if args.clf_type == "eventclf":
        labels = [
            "visual",
            "auditory",
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

    my_model = Model(name, args.net_option, input_size, n_outputs, save_path=args.save_path)
    my_model.from_pretrained()

    csv_file = os.path.join(args.save_path, f"participants_info.csv")
    dataframe = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)
    if args.clf_type == "subclf":
        data_path = os.path.join(args.save_path, f"downsampled_{args.sfreq}")
        n_subjects = len(os.listdir(data_path))
        n_outputs = min(n_subjects, args.max_subj)
        lso = False
    else:
        n_outputs = 2
        lso = True

    #######################
    ### PRODUCE CONSUME ###
    #######################

    # Setting start method to spawn for CUDA compatibility.
    multiprocessing.set_start_method("spawn")
    # Define the maximum number of threads that can read from the disk at once
    MAX_DISK_READERS = 1
    # Define the maximum size of the queue
    MAX_QUEUE_SIZE = 12  # Adjust this value based on your memory constraints
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
            [(queue, labels, sal_path, my_model.net, args)] * NUM_CONSUMERS,
        )

    # Wait for all producer threads to finish
    for t in threads:
        t.join()

    # Signal the consumer processes to exit
    for _ in range(NUM_CONSUMERS):
        queue.put(tuple([None] * 3))

    pool.close()
    pool.join()
