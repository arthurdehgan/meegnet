import threading
import multiprocessing
import os
import logging
import torch
import numpy as np
import pandas as pd
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.dataloaders import Dataset, RestDataset
from meegnet.utils import compute_saliency_maps
from meegnet_functions import get_input_size, get_name


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


if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    args = parser.parse_args()
    save_config(vars(args), args.config)

    if args.clf_type == "eventclf":
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run event classification"

    if args.clf_type == "eventclf":
        labels = [
            "visual",
            "auditory",
        ]  # image is label 0 and sound label 1
    elif args.clf_type == "subclf":
        labels = []
    else:
        labels = ["male", "female"]

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
        log_name = f"saliencies_{args.model_name}_{args.seed}_{args.sensors}"
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
