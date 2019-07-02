import os
import argparse
import scipy as sp
from sklearn.model_selection import (
    cross_val_score,
    StratifiedShuffleSplit as SSS,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from mlneurotools.ml import StratifiedShuffleGroupSplit
import numpy as np
from params import DATA_PATH, SAVE_PATH, CHAN_DF, SUB_DF, LABELS


def extract_bands(data):
    f = np.asarray([float(i / 3) for i in range(data.shape[-1])])
    # data = data[:, :, (f >= 8) * (f <= 12)].mean(axis=2)
    data = [
        data[:, :, (f >= .5) * (f <= 4)].mean(axis=-1)[..., None],
        data[:, :, (f >= 4) * (f <= 8)].mean(axis=-1)[..., None],
        data[:, :, (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[:, :, (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[:, :, (f >= 30) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=2)
    return data


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
    "-l",
    "--label",
    choices=["gender", "age", "age_all", "subject"],
    default="gender",
    help="The type of classification to run",
)
parser.add_argument(
    "-e",
    "--elec",
    choices=["MAG", "GRAD", "all"],
    default="all",
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
    "--n_subjout",
    type=int,
    default=4,
    help="The number of subjects to leave out for cross_validations",
)
parser.add_argument(
    "--n_crossval", type=int, default=1000, help="The number of cross-validations to do"
)
parser.add_argument(
    "--test_size",
    type=float,
    default=.5,
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

args = parser.parse_args()

if args.feature == "bands":
    get_features = extract_bands
else:
    get_features = lambda x: x

labels = LABELS[args.label]
if args.label == "gender":
    label = args.label
else:
    label = "age"

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


def load_freq_data(dataframe, get_features, labels, path=DATA_PATH):
    X = None
    y, groups = [], []
    for i, row in enumerate(dataframe[: int(len(dataframe))].iterrows()):
        sub = row[1]["Observations"]
        lab = labels[i]
        sub_data = get_features(np.load(path + "{}_rest_psd.npy".format(sub)))
        if NORM:
            sub_data = sp.stats.zscore(
                get_features(np.load(path + "{}_rest_psd.npy".format(sub)))
            )
        y += [lab] * len(sub_data)
        groups += [i] * len(sub_data)
        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        i += 1
    if args.label == "subject":
        y = groups
    return np.array(X), np.array(y), groups


if __name__ == "__main__":

    if args.verbose:
        print("Loading data...", sep="")
    data_df = SUB_DF[["Observations", label]]
    train_index, test_index = train_test_split(
        list(range(len(data_df))),
        test_size=int(args.test_size * len(data_df)),
        shuffle=True,
        stratify=labels,
        random_state=0,
    )

    train_df = data_df.iloc[train_index]
    test_df = data_df.iloc[test_index]

    all_scores = []
    X_og, y, groups = load_freq_data(train_df, get_features, labels)
    X_test_og, y_test, test_groups = load_freq_data(test_df, get_features, labels)
    if args.verbose:
        print("Done")

    for elec in elec_index:
        cv = StratifiedShuffleGroupSplit(args.n_subjout, args.n_crossval)
        savename = (
            SAVE_PATH
            + f"{args.clf}_{args.label}_test_scores_elec{CHAN_DF.iloc[elec]['ch_name']}.npy"
        )
        if args.verbose:
            print(CHAN_DF.iloc[elec])

        if not os.path.exists(savename) and not args.test:
            print(savename)
            X = X_og[:, elec]
            X_test = X_test_og[:, elec]

            try:
                if args.clf == "RF":
                    n_estimators = [
                        int(x) for x in np.linspace(start=200, stop=2000, num=10)
                    ]
                    max_features = ["auto", "sqrt"]
                    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
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
                        n_iter=args.iter,
                        cv=cv,
                        random_state=42,
                        n_jobs=args.cores,
                    )
                    clf = RF(**randsearch.best_params_)
                    cv = StratifiedShuffleGroupSplit(args.n_subjout, args.n_crossval)

                elif args.clf == "perceptron":
                    random_grid = {
                        "penality": ["l2", "l1", "elasticnet"],
                        "alpha": [0.0001, 0.001, 0.00001, 0.0005],
                    }
                    randsearch = RandomizedSearchCV(
                        estimator=Perceptron(),
                        param_distributions=random_grid,
                        n_iter=args.iter,
                        cv=cv,
                        random_state=42,
                        n_jobs=args.cores,
                    )
                    clf = Perceptron(**randsearch.best_params_)
                    cv = StratifiedShuffleGroupSplit(args.n_subjout, args.n_crossval)

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
                    cv = StratifiedShuffleGroupSplit(args.n_subjout, args.n_crossval)
                    if args.verbose:
                        print("Done")
                elif args.clf == "LDA":
                    clf = LDA()
                elif args.clf == "QDA":
                    clf = QDA()
            except:
                print("\rerror in algo optimization")

            try:
                if args.verbose:
                    print("Testing...", sep="")
                test = np.mean(
                    cross_val_score(
                        clf,
                        X_test,
                        y_test,
                        groups=test_groups,
                        cv=cv,
                        n_jobs=args.cores,
                    )
                )
                if args.verbose:
                    print("Done")
            except:
                print("\rproblem while testing")

            if not args.test:
                try:
                    if args.clf in ["perceptron", "RF", "SVM"]:
                        np.save(
                            SAVE_PATH
                            + f"{args.clf}_gender_train_scores_elec{CHAN_DF[elec]}",
                            train,
                        )
                        np.save(
                            SAVE_PATH + f"{args.clf}_gender_params_elec{CHAN_DF[elec]}",
                            param,
                        )
                    np.save(savename, test)
                except:
                    print("problem while saving")

            if args.test:
                break
