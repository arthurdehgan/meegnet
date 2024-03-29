import os
import logging
import toml
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import ConcatDataset
from meegnet.params import TIME_TRIAL_LENGTH
from meegnet.dataloaders import create_loader, create_datasets, load_sets
from meegnet.network import create_net
from meegnet.utils import train, load_checkpoint, cuda_check, evaluate
from meegnet.parsing import parser

####################
# DEBUG PARAMETERS #
####################

DEBUG = {"dropout": 0.5, "dropout_option": "same", "patience": 1}

##############
# CUDA CHECK #
##############

DEVICE = cuda_check()

#######################
# Torchsummary checks #
#######################

# try:
#     from torchinfo import summary as model_summary
# except ImportError:
#     logging.warning("Warning: Error loading torchinfo")


def get_df_status(file_path, args) -> pd.DataFrame:
    """checks if the specific parameters for the network are already logged."""
    df = pd.read_csv(file_path, index_col=0)
    return df.loc[
        (df["f"] == float(args.filters))
        & (df["linear"] == float(args.linear))
        & (df["d"] == float(args.dropout))
        & (df["hlayers"] == float(args.hlayers))
        & (df["nchan"] == float(args.nchan))
        & (df["batchnorm"] == float(args.batchnorm))
        & (df["maxpool"] == float(args.maxpool))
        & (df["bs"] == float(args.batch_size))
        & (df["lr"] == float(args.lr))
    ]


def do_crossval(folds: int, datasets: list, net_option: str, args) -> list:
    """Will call train_evaluate function a folds amount of times and return a list of
    each fold accuracies.

    Parameters
    ----------
    folds : int
        The number of folds for the K-Fold Cross-validatation that will be done.
    datasets : list
        A list of Datasets objects, each one will be used for a different Fold of the cross-val.
    net_option : str
        A network option for the prepare_net function to return a network object. Must be in
        ['MLP', 'custom_net', 'VGG', 'EEGNet', 'vanPutNet']
    args :
        Args should be args = parser.parse_args() when using parser from the parsing file.

    Returns
    -------
    cv : list
        A list containing the best network validation accuracy for each fold.
    """
    cv = []
    for fold in range(folds):
        logging.info(f"Training model for fold {fold+1}/{folds}:")
        results = train_evaluate(fold, datasets, net_option, args=args)
        logging.info(f"Finished training for fold {fold+1}/{folds}:")
        logging.info(f"loss: {results['loss_score']} // accuracy: {results['acc_score']}")
        logging.info(f"best epoch: {results['best_epoch']}/{results['n_epochs']}\n")
        cv.append(results["acc_score"])
    return cv


def train_evaluate(
    fold: int, datasets: list, net_option: str, args, skip_done: bool = False
) -> dict:
    """Trains and evaluate a specified network based on the selected network option.
    This function will also update the randomsearch.py generated csv if one exists with the
    accuracy scores for each fold. It returns a dictionnary containing all relevent information
    for the current training. This function is also capable of picking up where it left off
    in case of interrupted training.

    Parameters
    ----------
    fold : int
        then number of the fold to use if we run only one fold of the datasets.
    datasets : list
        A list of Datasets objects, each one will be used for a different Fold of the cross-val.
    net_option : str
        A network option for the prepare_net function to return a network object. Must be in
        ['MLP', 'custom_net', 'VGG', 'EEGNet', 'vanPutNet']
    args :
        Args should be args = parser.parse_args() when using parser from the parsing file.
    skip_done : bool
        Allows the script to check in the randomsearch-generated csv file if the fold has been
        computed already and skips it. Useful when compute ressources are limited and restarting
        jobs is common.

    Returns
    -------
    results : dict
        A dictionnary containing the information for the training and evaluation of the network.
        results contains the follwing keys and values
            "acc_score": [best_vacc], # The validation accuracy of the best network at early stop.
            "loss_score": [best_vloss], # The validation loss of the best network at early stop.
            "acc": valid_accs, # list of validation accuracies at each epoch
            "train_acc": train_accs, # list of train accuracies at each epoch
            "valid_loss": valid_losses, # list of validation losses at each epoch
            "train_loss": train_losses, # list of train losses at each epoch
            "best_epoch": best_epoch, # the epoch for early stop
            "n_epochs": epoch, # The total number of epochs ran before coming back to early stop.
            "patience": patience, # The patience parameter used.
            "current_patience": patience_state, # The current patience step,
                only useful if the network training was interrupted.
    """
    rs_csv_path = os.path.join(args.save_path, f"tested_params_seed{args.seed}.csv")
    if os.path.exists(rs_csv_path) and skip_done and args.randomsearch:
        check = get_df_status(rs_csv_path, args)
        if next(check[f"fold{fold}"].items())[1] != 0:
            return None

    name = f"{args.model_name}_{args.seed}_fold{fold+1}_{args.sensors}"
    suffixes = ""
    if net_option == "custom_net":
        if args.batchnorm:
            suffixes += "_BN"
        if args.maxpool != 0:
            suffixes += f"_maxpool{args.maxpool}"

        name += f"_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
        name += suffixes

    # TODO maybe use a dictionnary in order to store these values or use switch case
    if args.feature == "bins":
        trial_length = default_values["TRIAL_LENGTH_BINS"]
    elif args.feature == "bands":
        trial_length = default_values["TRIAL_LENGTH_BANDS"]
    elif args.feature == "temporal":
        trial_length = TIME_TRIAL_LENGTH
    elif args.feature == "cov":
        # TODO
        pass
    elif args.feature == "cosp":
        # TODO
        pass
    else:
        pass

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

    if args.mode == "overwrite":
        save = True
        load = False
    elif args.mode in ("continue", "evaluate"):
        save = True
        load = True
    else:
        save = False
        load = False

    net = create_net(args.net_option, name, input_size, n_outputs, DEVICE, args)
    model_filepath = os.path.join(args.save_path, net.name + ".pt")

    logging.info(net.name)
    try:
        logging.info(model_summary(net, input_size))
    except NameError:
        logging.info(net)

    # Create dataset splits for the current fold of the cross-val
    train_dataset = ConcatDataset(datasets[:fold] + datasets[fold + 1 :])
    trainloader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    validloader = create_loader(
        datasets[fold],
        batch_size=args.batch_size,
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
            patience=args.patience,
            lr=args.lr,
            mode=args.mode,
            save_path=args.save_path,
            permute_labels=args.permute_labels,
        )
    else:  # if we are in evaluate mode, we load the model if it exists, return warning if not
        if os.path.exists(model_filepath):
            _, net_state, _ = load_checkpoint(model_filepath)
            net.load_state_dict(net_state)
        else:
            logging.warning(f"Error: Can't evaluate model {model_filepath}, file not found.")

    results = loadmat(model_filepath[:-2] + "mat")
    if os.path.exists(rs_csv_path) and args.randomsearch:
        df = get_df_status(rs_csv_path, args)
        df[f"fold{fold}"] = results["acc_score"]
        df.to_csv(rs_csv_path)
    return results


if __name__ == "__main__":
    ###########
    # PARSING #
    ###########

    args = parser.parse_args()
    args_dict = vars(args)
    toml_string = toml.dumps(args_dict)
    with open(args.config, "w") as toml_file:
        toml.dump(args_dict, toml_file)
    with open("default_values.toml") as toml_file:
        default_values = toml.load(toml_file)

    fold = None if args.fold is None else int(args.fold)
    assert not (args.eventclf and args.subclf), "Please choose only one type of classification"
    if args.eventclf:
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run eventclf"
    ages = (args.age_min, args.age_max)

    ################
    # Starting log #
    ################

    logging.root.setLevel(logging.INFO)
    if args.log:
        log_name = args.model_name
        if fold is not None:
            log_name += f"_fold{args.fold}"
        log_name += ".log"
        log_file = os.path.join(args.save_path, log_name)
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        logging.info(f"Logging in {log_file}")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

    #######################
    # learning parameters #
    #######################

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("ENTERING DEBUG MODE")
        # args.max_subj = 20
        args.dropout = DEBUG["dropout"]
        args.dropout_option = DEBUG["droupt_option"]
        args.patience = DEBUG["patience"]

    if args.sensors == "MAG":
        n_channels = default_values["N_CHANNELS_MAG"]
        chan_index = [0]
    elif args.sensors == "GRAD":
        n_channels = default_values["N_CHANNELS_GRAD"]
        chan_index = [1, 2]
    else:
        n_channels = default_values["N_CHANNELS_OTHER"]
        chan_index = [0, 1, 2]

    ################
    # Loading data #
    ################

    # We create loaders and datasets (see dataloaders.py)
    if args.subclf:
        n_outputs, datasets = load_sets(
            args.save_path,
            n_samples=args.n_samples,
            max_subj=args.max_subj,
            chan_index=chan_index,
            seed=args.seed,
            printmem=args.printmem,
            epoched=args.epoched,
            datatype=args.datatype,
            testing=args.testsplit,
            sfreq=args.sfreq,
        )
        # Note: replace testing = testsplit or testing when we add the option to load test set and
        # use it for a test pass.
    else:
        datasets = create_datasets(
            args.save_path,
            args.train_size,
            args.max_subj,
            chan_index,
            seed=args.seed,
            debug=args.debug,
            printmem=args.printmem,
            ages=ages,
            n_samples=args.n_samples,
            datatype=args.datatype,
            eventclf=args.eventclf,
            epoched=args.epoched,
            testing=args.testsplit,
            sfreq=args.sfreq,
        )
        n_outputs = 2
        # Note: replace testing = testsplit or testing when we add the option to load test set and
        # use it for a test pass.

    ############
    # Training #
    ############

    DEFAULT_FOLDS = 4
    if args.randomsearch:
        if args.testsplit is None:
            for outer_fold in range(5):
                args.testsplit = outer_fold
                cv = do_crossval(
                    default_values["DEFAULT_FOLDS"], datasets, args.net_option, args=args
                )
                logging.info(f"\nAverage accuracy: {np.mean(cv)}")
        else:
            cv = do_crossval(
                default_values["DEFAULT_FOLDS"], datasets, args.net_option, args=args
            )
            cv = [score for score in cv if score is not None]
            logging.info(f"\nAverage accuracy: {np.mean(cv)}")

    elif args.crossval:
        folds = 5 if args.testsplit is None else DEFAULT_FOLDS
        cv = do_crossval(folds, datasets, args.net_option, args)
        logging.info(f"\nAverage accuracy: {np.mean(cv)}")

    else:
        fold = 0 if fold is None else fold
        logging.info("Training model:")
        train_evaluate(fold, datasets, args.net_option, args=args)
        # logging.info("Evaluating model:")
        # evaluate(fold, datasets, args.net_option, args=args)
