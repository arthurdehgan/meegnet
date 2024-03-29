from itertools import product
import os
import logging
import torch
import numpy as np
from scipy.io import loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loader, load_sets
from torch.utils.data import ConcatDataset
from network import FullNet
from utils import train, load_checkpoint
from parsing import parser

if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    parser.add_argument(
        "--dattype",
        default="rest",
        choices=["rest", "task", "passive"],
        help="the type of data to be loaded",
    )
    args = parser.parse_args()
    data_path = args.path
    if not data_path.endswith("/"):
        data_path += "/"
    save_path = args.save
    if not save_path.endswith("/"):
        save_path += "/"
    data_type = args.feature
    crossval = args.crossval
    maxpool = args.maxpool
    batch_size = args.batch_size
    max_subj = args.max_subj
    ch_type = args.elec
    debug = args.debug
    hlayers = args.hlayers
    filters = args.filters
    nchan = args.nchan
    dropout = args.dropout
    dropout_option = args.dropout_option
    linear = args.linear
    seed = args.seed
    mode = args.mode
    train_size = args.train_size
    num_workers = args.num_workers
    model_name = args.model_name
    patience = args.patience
    learning_rate = args.lr
    log = args.log
    printmem = args.printmem
    permute_labels = args.permute_labels
    samples = args.samples
    dattype = args.dattype
    batchnorm = args.batchnorm
    ages = (args.age_min, args.age_max)

    ##############
    # CUDA CHECK #
    ##############

    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

    ################
    # Starting log #
    ################

    if log:
        logging.basicConfig(
            filename=save_path + model_name + ".log",
            filemode="a",
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

    #######################
    # Torchsummary checks #
    #######################

    torchsum = True
    try:
        from torchsummary import summary
    except:
        logging.warning("Warning: Error loading torchsummary")
        torchsum = False

    #################
    # Parser checks #
    #################

    ##############
    # data types #
    ##############

    if ch_type == "MAG":
        n_channels = 102
    elif ch_type == "GRAD":
        n_channels = 204
    elif ch_type == "ALL":
        n_channels = 306

    if data_type == "bins":
        trial_length = 241
    if data_type == "bands":
        trial_length = 5
    elif data_type == "temporal":
        trial_length = TIME_TRIAL_LENGTH
    elif data_type == "cov":
        # TODO
        pass
    elif data_type == "cosp":
        # TODO
        pass

    #######################
    # learning parameters #
    #######################

    if debug:
        logging.debug("ENTERING DEBUG MODE")
        max_subj = 20
        dropout = 0.5
        dropout_option = "same"
        patience = 1

    #####################
    # preparing network #
    #####################

    input_size = (n_channels // 102, 102, trial_length)

    # We create loaders and datasets (see dataloaders.py)
    n_sub, datasets = load_sets(
        data_path,
        max_subj=max_subj,
        ch_type=ch_type,
        seed=seed,
        printmem=printmem,
        dattype=dattype,
    )

    if mode == "overwrite":
        save = True
        load = False
    elif mode in ("continue", "evaluate"):
        save = True
        load = True
    else:
        save = False
        load = False

    fold = 1
    if crossval:
        fold = 4
        cv = []

    # Actual training (loading nework if existing and load option is True)
    for i in range(fold):
        name = f"{model_name}_{seed}_fold{i}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}_depth{hlayers}"
        if batchnorm:
            name += "_BN"
        if maxpool != 0:
            name += f"_maxpool{maxpool}"
        net = FullNet(
            name,
            input_size,
            hlayers,
            filters,
            nchan,
            linear,
            dropout,
            dropout_option,
            batchnorm,
            maxpool,
            sub=True,
            n_sub=n_sub,
        ).to(device)

        model_filepath = save_path + net.name + ".pt"
        logging.info(net.name)
        if torchsum:
            logging.info(summary(net, input_size))
        else:
            logging.info(net)

        if crossval:
            logging.info(f"Training model for fold {i+1}/4:")
        else:
            logging.info("Training model:")
        train_dataset = ConcatDataset(datasets[:i] + datasets[i + 1 :])
        trainloader = create_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validloader = create_loader(
            datasets[i],
            batch_size=int(len(datasets[i]) / 4),
            num_workers=num_workers,
        )
        # TODO update modes and check if we can add testing to this script or needs another one
        if mode != "evaluate":
            train(
                net,
                trainloader,
                validloader,
                model_filepath,
                save_model=save,
                load_model=load,
                debug=debug,
                p=patience,
                lr=learning_rate,
                mode=mode,
                save_path=save_path,
            )
        else:
            if os.path.exists(model_filepath):
                _, net_state, _ = load_checkpoint(model_filepath)
                net.load_state_dict(net_state)
            else:
                logging.warning(
                    f"Error: Can't evaluate model {model_filepath}, file not found."
                )

        # Evaluating
        if crossval:
            logging.info(f"Evaluating model for fold {i}/4:")
        else:
            logging.info("Evaluating model:")
        results = loadmat(model_filepath[:-2] + "mat")
        acc = results["acc_score"]
        if crossval:
            cv.append(acc)
        logging.info(f"loss: {results['loss_score']} // accuracy: {acc}")
        logging.info(f"best epoch: {results['best_epoch']}/{results['n_epochs']}\n")

    if crossval:
        logging.info(f"\nAverage accuracy: {np.mean(cv)}")
