import os
import argparse
import scipy as sp
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    StratifiedShuffleSplit as SSS,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit as SSS
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from mlneurotools.ml import StratifiedShuffleGroupSplit as SSGS
from params import DATA_PATH, SAVE_PATH, CHAN_DF, SUB_DF, LABELS


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--cores", default=-1, type=int, help="The number of cores to use"
)
parser.add_argument("-v", "--verbose", action="store_true", help="display more info")
parser.add_argument(
    "--clf",
    choices=["SVM", "LDA", "QDA", "RF", "perceptron"],
    default="LDA",
    help="The classifier that will be used for the classification",
)
parser.add_argument(
    "-p", "--permutations", type=int, default=None, help="The number of permutations"
)
parser.add_argument(
    "-d",
    "--data_type",
    choices=["task", "rest", "passive"],
    default="rest",
    help="The type of data to use for classification",
)
parser.add_argument(
    "--clean_type",
    choices=["mf", "transdef_mf", "raw"],
    default="mf",
    help="The type of data to use for classification",
)
parser.add_argument(
    "-l",
    "--label",
    choices=["gender", "age", "subject"],
    default="gender",
    help="The type of classification to run",
)
parser.add_argument(
    "-e",
    "--elec",
    choices=["MAG", "GRAD", "all"],
    default="MAG",
    help="The type of electrodes to keep",
)
parser.add_argument(
    "-f",
    "--feature",
    choices=["bands", "bins"],
    default="bands",
    help="The type of features to use",
)
parser.add_argument(
    "--n_crossval", type=int, default=1000, help="The number of cross-validations to do"
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.5,
    help="The percentage of the dataset to use as test set",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=100,
    help="The number of iterations to do for random search hyperparameter optimization",
)
parser.add_argument(
    "-t",
    "--test",
    action="store_true",
    help="Launch the pipeline in test mode : will not save and will only do 2 iteration for each loop",
)
parser.add_argument(
    "-o",
    "--out_path",
    default=SAVE_PATH + "results/",
    help="Where to save the result matrices",
)
parser.add_argument(
    "-i", "--in_path", default=DATA_PATH, help="Where is the data to load"
)
args = parser.parse_args()


def extract_bands(data):
    f = np.asarray([float(i / 3) for i in range(data.shape[-1])])
    # data = data[:, :, (f >= 8) * (f <= 12)].mean(axis=2)
    data = [
        data[:, :, (f >= 0.5) * (f <= 4)].mean(axis=-1)[..., None],
        data[:, :, (f >= 4) * (f <= 8)].mean(axis=-1)[..., None],
        data[:, :, (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[:, :, (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[:, :, (f >= 30) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=2)
    return data


def load_freq_data(dataframe, data_type, clean_type, get_features, labels, args, path):
    X = None
    y, groups = [], []
    for i, row in enumerate(dataframe[: int(len(dataframe))].iterrows()):
        sub = row[1]["participant_id"]
        try:
            data = np.load(path + f"{sub}_{data_type}_{clean_type}_psd.npy")
        except FileNotFoundError:
            print(sub, "could not be loaded with datatype", data_type, clean_type)
        sub_data = get_features(data)
        if NORM:
            sub_data = sp.stats.zscore(sub_data)
        y += [labels[i]] * len(sub_data)
        groups += [i] * len(sub_data)
        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        i += 1
        if args.test and i >= 40:
            break
    if args.label == "subject":
        y = groups
    return np.array(X), np.array(y), np.array(groups)


def load_data(label, data_type, clean_type, feature, args):
    data_path = args.in_path + f"{args.clean_type}/"
    og_labels = np.array(LABELS[label])
    stratify = og_labels
    if label == "gender":
        col = args.label
    else:
        col = "age"

    if label in ["subject"]:
        stratify = None

    if feature == "bands":
        get_features = extract_bands
    else:
        get_features = lambda x: x

    if args.verbose:
        print("Loading data...", sep="")
    data_df = SUB_DF[["participant_id", col]]
    train_index, test_index = train_test_split(
        list(range(len(data_df))),
        test_size=int(args.test_size * len(data_df)),
        shuffle=True,
        stratify=stratify,
        random_state=42,
    )

    train_df = data_df.iloc[train_index]
    train_labels = og_labels[train_index]
    test_df = data_df.iloc[test_index]
    test_labels = og_labels[train_index]

    train_set = load_freq_data(
        train_df, data_type, clean_type, get_features, train_labels, args, data_path
    )
    test_set = load_freq_data(
        test_df, data_type, clean_type, get_features, test_labels, args, data_path
    )
    if args.verbose:
        print("Done")
    return train_set, test_set


def classif(
    train_set, test_set, clf, data_type, clean_type, elec, label, feature, args
):
    elec_name = CHAN_DF.iloc[elec]["ch_name"]
    X_og, y, groups = train_set
    X_test_og, y_test, groups_test = test_set
    if label != "subject":
        cv = SSGS(len(np.unique(y)) * 1, args.n_crossval)
    else:
        cv = SSS(10)
    savepath = (
        args.out_path
        + f"{args.label}/{args.data_type}_{args.clean_type}/{args.feature}/{args.clf}/"
    )
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    savename = savepath + f"test_scores_elec{elec_name}.npy"
    # if args.verbose:
    # print(CHAN_DF.iloc[elec])

    if not os.path.exists(savename) or args.test:
        with open(savename, "w") as f:
            f.write("")

        if args.verbose:
            print(savename)
        X = X_og[:, elec].squeeze()
        X_test = X_test_og[:, elec].squeeze()
        pattern = [0, 1, 180, 181, 500, 501, 360, 361]
        if args.test:
            print(f"data shapes train: {X.shape}, test: {X_test.shape}")
            print(f"example labels train: {y[pattern]}, test: {y_test[pattern]}")
            print(
                f"example groups train: {groups[pattern]}, test: {groups_test[pattern]}"
            )
        # if args.clf in ["LDA"] and len(np.unique(y)) > 2:
        #     return

        if args.clf == "RF":
            n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
            max_features = ["auto", "sqrt"]
            max_depth = [int(x) for x in np.linspace(3, 110, num=11)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]

            random_grid = {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "bootstrap": bootstrap,
            }
            randsearch = RandomizedSearchCV(
                estimator=RF(),
                param_distributions=random_grid,
                n_iter=args.iterations,
                cv=cv,
                random_state=42,
                n_jobs=args.cores,
            )
            clf = RF(**randsearch.best_params_)

        elif args.clf == "perceptron":
            random_grid = {
                "penality": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.00001, 0.0005],
            }
            randsearch = RandomizedSearchCV(
                estimator=Perceptron(),
                param_distributions=random_grid,
                n_iter=args.iterations,
                cv=cv,
                random_state=42,
                n_jobs=args.cores,
            )
            print("BOOM")
            clf = Perceptron(**randsearch.best_params_)

        elif args.clf == "SVM":
            param_distributions = {
                "C": sp.stats.expon(scale=10),
                "gamma": sp.stats.expon(scale=0.1),
            }
            if args.verbose:
                print("Random Search...", sep="")
            randsearch = RandomizedSearchCV(
                SVC(),
                param_distributions=param_distributions,
                n_iter=args.iterations,
                cv=cv,
                random_state=42,
                n_jobs=args.cores,
            )
            randsearch.fit(X, y, groups)
            param = randsearch.best_params_
            train = randsearch.best_score_
            clf = SVC(**randsearch.best_params_)
            if args.verbose:
                print("Done")
        elif args.clf == "LDA":
            clf = LDA()
        elif args.clf == "QDA":
            clf = QDA()

        if args.verbose:
            print("Evaluating...", sep="")
        if label != "subject":
            cv = SSGS(len(np.unique(y_test)) * 1, args.n_crossval)
        else:
            cv = SSS(10)

        test = np.mean(
            cross_val_score(
                clf, X_test, y_test, groups=groups_test, cv=cv, n_jobs=args.cores
            )
        )
        if args.verbose:
            print("Done")

        if not args.test:
            if args.clf in ["perceptron", "RF", "SVM"]:
                np.save(savepath + f"train_scores_elec{elec_name}", train)
                np.save(savepath + f"params_elec{elec_name}", param)
            np.save(savename, test)


if __name__ == "__main__":

    if args.elec == "MAG":
        elec_index = range(2, 306, 3)
    elif args.elec == "GRAD":
        elec_index = list(range(0, 306, 3))
        elec_index += list(range(1, 306, 3))
    elif args.elec == "all":
        elec_index = range(306)

    NORM = False
    if args.clf == "SVM":
        NORM = True

    if args.test:
        print("Testing")
        args.n_crossval = 2
        args.iterations = 2
        elec = np.random.choice(list(range(len(CHAN_DF["ch_name"]))), 1)
        # for clean_type in ["mf", "transdef_mf", "raw"]:
        for clean_type in ["mf"]:
            for data_type in ["rest", "task", "passive"]:
                for clf in ["SVM", "LDA", "QDA", "RF", "perceptron"]:
                    for label in ["subject", "age", "gender"]:
                        for feature in ["bins", "bands"]:
                            print("\n", clf, label, feature)
                            train_set, test_set = load_data(
                                label, data_type, clean_type, feature, args
                            )
                            classif(
                                train_set,
                                test_set,
                                clf,
                                data_type,
                                clean_type,
                                elec,
                                label,
                                feature,
                                args,
                            )
    else:
        train_set, test_set = load_data(
            args.label, args.data_type, args.clean_type, args.feature, args
        )
        for elec in elec_index:
            classif(
                train_set,
                test_set,
                args.clf,
                args.data_type,
                args.clean_type,
                elec,
                args.label,
                args.feature,
                args,
            )
