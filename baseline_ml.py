from itertools import product
import os
import logging
import torch
import sklearn as sk
from mlneurotools.ml import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from scipy.io import loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loader, create_datasets
from torch.utils.data import ConcatDataset, DataLoader
from network import FullNet
from utils import train, load_checkpoint
from parsing import parser

if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    parser.add_argument(
        "--cv",
        default=4,
        help="number of Fold in a K-FOLD",
    )
    parser.add_argument(
        "--hypop",
        default=0,
        help="number of parameters to test in randomsearch if hyperparameter optimization. set to 0 for no hyperparammeter optimization",
    )
    parser.add_argument(
        "--dattype",
        default="rest",
        choices=["rest", "task", "passive"],
        help="the type of data to be loaded.",
    )
    parser.add_argument(
        "--space",
        default="euclidian",
        choices=["euclidian", "riemannian"],
        help="the space in which to classify.",
    )
    parser.add_argument(
        "--classifier",
        default="LDA",
        choices=["LDA", "SVM", "LR"],
        help="The classifier to use.",
    )

    args = parser.parse_args()
    data_path = args.path
    if not data_path.endswith("/"):
        data_path += "/"
    save_path = args.save
    if not save_path.endswith("/"):
        save_path += "/"
    data_type = args.feature
    max_subj = args.max_subj
    hypop = args.hypop
    ch_type = args.elec
    debug = args.debug
    seed = args.seed
    train_size = args.train_size
    num_workers = args.num_workers
    model_name = args.model_name
    log = args.log
    printmem = args.printmem
    permute_labels = args.permute_labels
    samples = args.samples
    dattype = args.dattype
    space = args.space
    classifier = args.classifier
    ages = (args.age_min, args.age_max)

    if space == "riemannian":
        from pyriemann.classification import TSclassifier

    ###############
    # Classifiers #
    ###############

    if classifier == "LDA":
        clf = sk.discriminant_analysis()
        params = {}
    elif classifier == "SVM":
        clf = sk.svm.SVC()
        params = {
            "C": scipy.stats.expon(scale=100),
            "gamma": scipy.stats.expon(scale=0.1),
            "kernel": ["rbf"],
            "class_weight": ["balanced", None],
        }
    elif classifier == "LR":
        clf = sk.linear.LogisticRegression()
        params = {"penality": ["l1", "l2"], "C": uniform(loc=0, scale=4)}

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

    #########################
    # debug mode definition #
    #########################

    if debug:
        logging.debug("ENTERING DEBUG MODE")
        max_subj = 20

    #####################
    # preparing network #
    #####################

    # We create loaders and datasets (see dataloaders.py)
    datasets = create_datasets(
        data_path,
        train_size,
        max_subj,
        ch_type,
        data_type,
        seed=seed,
        num_workers=num_workers,
        debug=debug,
        printmem=printmem,
        ages=ages,
        permute_labels=permute_labels,
        samples=samples,
        dattype=dattype,
        load_groups=True,
    )
    crossval = StratifiedGroupKFold(n_splits=cv, random_state=seed)

    if space == "riemannian":
        classifier = f"riemannian{classifier}"
        clf = TSClassifier(clf=clf)
    name = f"{model_name}_{seed}_{classifier}_{ch_type}_{data_type}"

    logging.info(f"{clf}")
    logging.info(f"Training...")
    train_dataset = ConcatDataset(datasets[:4])
    X, y, groups = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))

    if hypop != 0:
        name += "_opti"
        clf = RandomizedSearchCV(
            estimator=clf,
            param_distributions=params,
            random_state=seed,
            n_iter=hypop,
            cv=crossval,
            verbose=0,
            n_jobs=-1,
        )
        clf.fit(X=X, y=y, groups=groups)
        scores = clf.fit()
    else:
        scores = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            groups=groups,
            cv=crossval,
            n_jobs=-1,
        )