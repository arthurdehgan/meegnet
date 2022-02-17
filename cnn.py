from itertools import product
import os
import logging
import torch
import numpy as np
from scipy.io import loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loader, create_datasets, load_sets
from torch.utils.data import ConcatDataset, TensorDataset
from network import FullNet, MLP
from utils import train, load_checkpoint
from parsing import parser

if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    parser.add_argument(
        "--notest",
        action="store_true",
        help="Will remove the 20% holdout set by default and usit for cross-val. Using 5-Fold instead of 4-Fold.",
    )
    parser.add_argument(
        "--eventclf",
        action="store_true",
        help="launches event classification instead of gender classification.",
    )
    parser.add_argument(
        "--subclf",
        action="store_true",
        help="launches subject classification instead of gender classification.",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if notest option is true",
    )
    parser.add_argument(
        "--net-option",
        default="cNet",
        choices=["cNet", "MLP"],
        help="cNet is the custom net.",
    )
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
    fold = None if args.fold is None else int(args.fold)
    if args.eventclf:
        assert (
            args.dattype == "passive"
        ), "dattype must be set to passive in order to run eventclf"
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

    if args.log:
        logging.basicConfig(
            filename=save_path + args.model_name + ".log",
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

    if args.elec == "MAG":
        n_channels = 102
    elif args.elec == "GRAD":
        n_channels = 204
    elif args.elec == "ALL":
        n_channels = 306

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

    #######################
    # learning parameters #
    #######################

    if args.debug:
        logging.debug("ENTERING DEBUG MODE")
        args.max_subj = 20
        args.dropout = 0.5
        args.dropout_option = "same"
        args.patience = 1

    #####################
    # preparing network #
    #####################

    input_size = (n_channels // 102, 102, trial_length)

    # We create loaders and datasets (see dataloaders.py)
    if args.subclf:
        n_outputs, datasets = load_sets(
            data_path,
            max_subj=args.max_subj,
            ch_type=args.elec,
            seed=args.seed,
            printmem=args.printmem,
            dattype=args.dattype,
            testing=args.notest,
        )
        # Note: replace testing = notest or testing when we add the option to load test set and use it for a test pass.
    else:
        datasets = create_datasets(
            data_path,
            args.train_size,
            args.max_subj,
            args.elec,
            args.feature,
            seed=args.seed,
            debug=args.debug,
            printmem=args.printmem,
            ages=ages,
            samples=args.samples,
            dattype=args.dattype,
            load_events=args.eventclf,
            testing=args.notest,
        )
        n_outputs = 2
        # Note: replace testing = notest or testing when we add the option to load test set and use it for a test pass.

    if args.mode == "overwrite":
        save = True
        load = False
    elif args.mode in ("continue", "evaluate"):
        save = True
        load = True
    else:
        save = False
        load = False

    folds = 1
    if args.crossval:
        folds = 4
        if args.notest:
            folds = 5
        cv = []

    # Actual training (loading nework if existing and load option is True)
    for i in range(folds):
        if fold is not None:
            i = fold
        name = f"{args.model_name}_{args.seed}_fold{i+1}_{args.elec}_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
        if args.batchnorm:
            name += "_BN"
        if args.maxpool != 0:
            name += f"_maxpool{args.maxpool}"
        if args.net_option == "MLP":
            net = MLP(
                name=name,
                input_size=input_size,
                n_outputs=n_outputs,
                hparams={
                    "mlp_width": args.linear,
                    "mlp_depth": args.hlayers,
                    "mlp_dropout": args.dropout,
                },
            ).to(device)
        else:
            net = FullNet(
                name,
                input_size,
                n_outputs,
                args.hlayers,
                args.filters,
                args.nchan,
                args.linear,
                args.dropout,
                args.dropout_option,
                args.batchnorm,
                args.maxpool,
            ).to(device)

        model_filepath = save_path + net.name + ".pt"
        logging.info(net.name)
        if torchsum:
            logging.info(summary(net, input_size))
        else:
            logging.info(net)

        if args.crossval:
            logging.info(f"Training model for fold {i+1}/{folds}:")
        else:
            logging.info("Training model:")

        n_lab = 612 if args.subclf else 2

        train_dataset = ConcatDataset(datasets[:i] + datasets[i + 1 :])
        trainloader = create_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        validloader = create_loader(
            datasets[i],
            batch_size=int(len(datasets[i]) / folds),
            num_workers=args.num_workers,
            shuffle=True,
        )
        # TODO update modes and check if we can add testing to this script or needs another one
        if args.mode != "evaluate":
            train(
                net,
                trainloader,
                validloader,
                model_filepath,
                save_model=save,
                load_model=load,
                debug=args.debug,
                p=args.patience,
                lr=args.lr,
                mode=args.mode,
                save_path=save_path,
                permute_labels=args.permute_labels,
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
        if args.crossval:
            logging.info(f"Evaluating model for fold {i+1}/{fold}:")
        else:
            logging.info("Evaluating model:")
        results = loadmat(model_filepath[:-2] + "mat")
        acc = results["acc_score"]
        if args.crossval:
            cv.append(acc)
        logging.info(f"loss: {results['loss_score']} // accuracy: {acc}")
        logging.info(f"best epoch: {results['best_epoch']}/{results['n_epochs']}\n")
        if fold is not None:
            break

    if args.crossval:
        logging.info(f"\nAverage accuracy: {np.mean(cv)}")

    # # Final testing
    # if os.path.exists(model_filepath):
    #     _, net_state, _ = load_checkpoint(model_filepath)
    #     net.load_state_dict(net_state)
    # else:
    #     logging.warning(
    #         f"Error: Can't evaluate model {model_filepath}, file not found."
    #     )
    # print("Evaluating on test set:")
    # tloss, tacc = evaluate(net, testloader)
    # print("loss: ", tloss, " // accuracy: ", tacc)
    # if save:
    #     results = loadmat(model_filepath[:-2] + "mat")
    #     print("best epoch: ", f"{results['best_epoch']}/{results['n_epochs']}")
    #     results["test_acc"] = tacc
    #     results["test_loss"] = tloss
    #     savemat(save_path + net.name + ".mat", results)
