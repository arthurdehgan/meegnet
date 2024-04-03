import threading
import multiprocessing
import numpy as np
import pandas as pd
import torch
import os
import logging
from meegnet.params import TIME_TRIAL_LENGTH
from meegnet.parsing import parser, save_config
from meegnet.network import create_net
from meegnet.misc_functions import compute_saliency_based_psd
from meegnet.utils import load_checkpoint
from meegnet.dataloaders import load_data


DEVICE = "cpu"


def produce_data(queue, sub, args, disk_semaphore):
    with disk_semaphore:
        saliencies = {}
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
        selected_trials = [[] for _ in labels]
        selected_targets = [[] for _ in labels]
        for saliency_type in ("pos", "neg"):
            save_path = os.path.join(
                args.save_path,
                "saliency_based_psd",
                name,
                saliency_type + f"_{args.confidence}confidence",
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for trial, target in zip(data, targets):
                X = trial[np.newaxis].type(torch.FloatTensor).to(DEVICE)
                if len(X.shape) < 4:
                    X = X[np.newaxis, :]
                # Compute predictions of the trained network, and confidence
                preds = torch.nn.Softmax(dim=1)(net(X)).detach().cpu()
                pred = preds.argmax().item()
                confidence = preds.max()
                if confidence >= args.confidence and pred == target:
                    # Remove newaxis added on;y for forward pass by using X[0]
                    selected_trials[target].append(X[0])
                    selected_targets[target].append(target)

            for i, label in enumerate(labels):
                lab = "" if args.clf_type == "subclf" else f"_{label}"
                file_name = f"{sub}{lab}_{saliency_type}_sal_{args.confidence}confidence.npy"
                saliency_file = os.path.join(sal_path, file_name)
                if os.path.exists(saliency_file):
                    try:
                        saliencies = np.load(saliency_file)
                    except IOError:
                        logging.warning(f"Error loading {saliency_file}")
                        queue.put(tuple([None] * 3 + [0]))
                        continue
                else:
                    logging.warning(f"{saliency_file} does not exist.")
                    queue.put(tuple([None] * 3 + [0]))
                    continue
                file_name = f"sbpsd_{sub}{lab}.npy"
                file_path = os.path.join(save_path, file_name)
                queue.put((np.array(selected_trials[i]), saliencies, file_path, 1))
        return


def process_data(args):
    queue, fs = args
    while True:
        trials, saliencies, file_path, stop = queue.get()
        if stop is None:
            break
        if trials is None:
            continue
        bands_values = [[] * trials.shape[2]]
        for i in range(len(bands_values)):
            for trial, saliency in zip(trials[i], saliencies[i]):
                sbpsd = compute_saliency_based_psd(saliency, trial, w_size, fs)
                sbpsd = [e for e in sbpsd if e is not None]
                bands_values[i] += sbpsd

        np.save(file_path, np.array(bands_values).squeeze())


if __name__ == "__main__":

    args = parser.parse_args()
    logging.info(parser.format_values())
    save_config(vars(args), args.config)

    ###############################
    ### EXTRACTING PARSER INFO ###
    ###############################

    if args.clf_type == "eventclf":
        labels = ["visual", "auditory"]  # image is label 0 and sound label 1
    elif args.clf_type == "subclf":
        labels = []
    else:
        labels = ["male", "female"]

    if args.feature == "bins":
        trial_length = 241
    if args.feature == "bands":
        trial_length = 5
    elif args.feature == "temporal":
        trial_length = TIME_TRIAL_LENGTH

    if args.sensors == "MAG":
        n_channels = 102
        chan_index = 0
    elif args.sensors == "GRAD":
        n_channels = 204
        chan_index = [1, 2]
    elif args.sensors == "ALL":
        n_channels = 306
        chan_index = [0, 1, 2]

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

    input_size = (n_channels // 102, 102, trial_length)

    if args.clf_type == "subclf":
        # TODO
        raise "not yet implemented for subclf"
    else:
        n_outputs = 2

    #########################
    ### PREPARING FOLDERS ###
    #########################

    sal_path = os.path.join(args.save_path, "saliency_maps", name)
    model_filepath = os.path.join(args.save_path, name + ".pt")
    csv_path = os.path.join(args.save_path, f"participants_info_{args.datatype}.csv")

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        log_name = args.model_name
        if args.fold != -1:
            log_name += f"_fold{args.fold}"
        log_name += "_psd_based_saliency_computations.log"
        log_filepath = os.path.join(args.save_path, log_name)
        logging.basicConfig(
            filename=log_filepath,
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

    #######################
    ### LOADING NETWORK ###
    #######################

    net = create_net(args.net_option, name, input_size, n_outputs, DEVICE, args)
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    #################################
    ### LOADING PARTICIPANTS INFO ###
    #################################

    dataframe = pd.read_csv(csv_path, index_col=0)
    file_list = os.listdir(sal_path)
    subjects = np.unique(
        [
            file.split("_")[0]
            for file in file_list
            if set(file.split("_")).intersection(set(labels)) != set()
        ]
    )[: args.max_subj]

    #############################
    ### HARD-CODED PARAMETERS ###
    #############################

    w_size = int(300 * args.sfreq / 1000)

    #######################
    ### PRODUCE CONSUME ###
    #######################

    # Define the maximum number of threads that can read from the disk at once
    MAX_DISK_READERS = 1
    # Define the maximum size of the queue
    MAX_QUEUE_SIZE = 100  # Adjust this value based on your memory constraints
    NUM_CONSUMERS = 6

    # Create a bounded queue with the maximum size
    queue = multiprocessing.Manager().Queue(maxsize=MAX_QUEUE_SIZE)
    # Create a semaphore with the maximum number of readers
    disk_semaphore = threading.Semaphore(MAX_DISK_READERS)

    # Start the producer threads
    threads = []
    for sub in subjects:
        t = threading.Thread(
            target=produce_data,
            args=(
                queue,
                sub,
                args,
                disk_semaphore,
            ),
        )
        t.start()
        threads.append(t)

    # Start the consumer processes
    with multiprocessing.Pool(processes=NUM_CONSUMERS) as pool:
        pool.map(process_data, [(queue, args.sfreq)] * NUM_CONSUMERS)

    # Wait for all producer threads to finish
    for t in threads:
        t.join()

    # Signal the consumer processes to exit
    for _ in range(NUM_CONSUMERS):
        queue.put(tuple([None] * 4))

    pool.close()
    pool.join()
