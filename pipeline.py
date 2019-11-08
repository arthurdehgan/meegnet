import os
import time
from parser import args
import xgboost as xgb
import scipy as sp
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    StratifiedShuffleSplit as SSS,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from mlneurotools.ml import StratifiedShuffleGroupSplit as SSGS
from utils import elapsed_time
from params import CHAN_DF, SUB_DF, LABELS


def check_classif_done(elec_index, args):
    elec_list = []
    savepath = args.out_path + f"{args.label}/{args.feature}/{args.clf}/"
    for elec in elec_index:
        elec_name = CHAN_DF.iloc[elec]["ch_name"]
        savename = savepath + f"test_scores_elec{elec_name}.npy"
        if not os.path.exists(savename):
            elec_list.append(elec_name)
    return elec_list


def print_info_classif(args):
    try:
        elec = CHAN_DF.iloc[int(args.elec)]["ch_name"]
    except:
        elec = args.elec
    print(f"\nClassification of {args.label}s on individual electrodes.")
    print(f"Classifier: {args.clf}")
    if args.label != "subject":
        print("Cross-Validation: Stratified Leave Groups Out.")
    else:
        print("Cross-Validation: Stratified Shuffle Split.")
    print(f"Number of cross-validation steps: {args.n_crossval}.")
    print(f"Features used: frequency {args.feature}.")
    print(f"Sensor used: {elec}")
    if args.clf in ["perceptron", "SVM", "RF", "XGBoost"]:
        print(f"Random search will be used to fine tune hyperparameters.")
        print(f"n_iters={args.iterations}")
        print(f"{(1-args.test_size) * 100}% of the data will be used for fine tuning.")
    print(f"Results will be saved in: {args.out_path}")
    if args.permutations is not None:
        print(f"Permutation test will be done with {args.permutations} permutations.")


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


def load_freq_data(dataframe, elec_index, get_features, labels, args, path):
    X = None
    y, groups = [], []
    for i, row in enumerate(dataframe[: int(len(dataframe))].iterrows()):
        sub = row[1]["participant_id"]
        try:
            data = np.load(path + f"{sub}_psd.npy")
            data = np.take(data, elec_index, args.elec_axis)
        except FileNotFoundError:
            print(sub, "could not be loaded in path:")
            print(f"{path}")
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


def load_data(elec_index, args):
    data_path = args.in_path
    og_labels = np.array(LABELS[args.label])
    stratify = og_labels
    if args.label == "gender":
        col = args.label
    else:
        col = "age"

    if args.label in ["subject"]:
        stratify = None

    if args.feature == "bands":
        get_features = extract_bands
    else:
        get_features = lambda x: x

    if args.verbose > 0:
        print("Loading data...")
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

    if args.verbose > 0 and args.time:
        start = time.time()
    train_set = load_freq_data(
        train_df, elec_index, get_features, train_labels, args, data_path
    )
    test_set = load_freq_data(
        test_df, elec_index, get_features, test_labels, args, data_path
    )
    if args.verbose > 0 and args.time:
        print("Time spent loading data:", elapsed_time(time.time(), start))
    if args.verbose > 0:
        print("Done")
        print(f"train_size: {train_set[0].shape} (Used for hyperparameter tuning)")
        print(f"test_size: {test_set[0].shape} (used to evaluate the model)")
    return train_set, test_set


def random_search(args, cv, X, y, groups=None):
    if args.clf == "XGBoost":
        param_grid = {
            "silent": [False],
            "max_depth": [6, 10, 15, 20],
            "learning_rate": [0.001, 0.01, 0.1, 0.2, 0, 3],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            "gamma": [0, 0.25, 0.5, 1.0],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            "n_estimators": [100],
        }
        randsearch = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(),
            param_distributions=param_grid,
            n_iter=args.iterations,
            cv=cv,
            random_state=42,
            n_jobs=args.cores,
        )
        randsearch.fit(X, y, groups)
        param = randsearch.best_params_
        train = randsearch.best_score_
        clf = xgb.XGBClassifier(**randsearch.best_params_)

    if args.clf == "RF":
        n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
        max_features = ["auto", "sqrt"]
        max_depth = [int(x) for x in np.linspace(3, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }
        randsearch = RandomizedSearchCV(
            estimator=RF(),
            param_distributions=param_grid,
            n_iter=args.iterations,
            cv=cv,
            random_state=42,
            n_jobs=args.cores,
        )
        randsearch.fit(X, y, groups)
        param = randsearch.best_params_
        train = randsearch.best_score_
        clf = RF(**randsearch.best_params_)

    elif args.clf == "perceptron":
        param_grid = {
            "penality": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.00001, 0.0005],
        }
        randsearch = RandomizedSearchCV(
            estimator=Perceptron(),
            param_distributions=param_grid,
            n_iter=args.iterations,
            cv=cv,
            random_state=42,
            n_jobs=args.cores,
        )
        randsearch.fit(X, y, groups)
        param = randsearch.best_params_
        train = randsearch.best_score_
        clf = Perceptron(**randsearch.best_params_)

    elif args.clf == "SVM":
        param_distributions = {
            "C": sp.stats.expon(scale=10),
            "gamma": sp.stats.expon(scale=0.1),
        }
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
    elif args.clf == "LDA":
        clf = LDA()
        param = None
        train = None
    elif args.clf == "QDA":
        clf = QDA()
        param = None
        train = None
    return clf, param, train


def create_crossval(label, y):
    if label != "subject":
        return SSGS(len(np.unique(y)) * 1, args.n_crossval)
    return SSS(10)


def classif(train_set, test_set, args):
    X, y, groups = train_set
    X_test, y_test, groups_test = test_set

    cv = create_crossval(args.label, y)
    if args.verbose > 0 and args.time:
        start = time.time()
    clf, param, train = random_search(args, cv, X, y, groups)
    if args.verbose > 0 and args.time and args.clf not in ["LDA", "QDA"]:
        print("Time spend in Random Search:", elapsed_time(time.time(), start))
    cv = create_crossval(args.label, y_test)
    if args.verbose > 0 and args.time:
        start = time.time()
    test = np.mean(
        cross_val_score(
            clf, X_test, y_test, groups=groups_test, cv=cv, n_jobs=args.cores
        )
    )
    if args.verbose > 0 and args.time:
        print("Time spent evaluating the model:", elapsed_time(time.time(), start))
    return param, train, test


def classif_all_elecs(train_set, test_set, elec_list, args):
    assert args.elec_axis < len(train_set[0].shape), "Error, elec axis out of bounds."
    X_og, y, groups = train_set
    X_test_og, y_test, groups_test = test_set

    for elec, elec_name in enumerate(elec_list):
        if args.feature == "subject":
            labelname = f"{args.label}_{len(set(test_set[1]))}"
        else:
            labelname = args.label
        savepath = args.out_path + f"{labelname}/{args.feature}/{args.clf}/"
        if not os.path.isdir(savepath):
            try:
                os.makedirs(savepath)
            except:
                print("couldnt create the directory for some reason")
        savename = savepath + f"test_scores_elec{elec_name}.npy"

        if not os.path.exists(savename) or args.test:
            if args.verbose > 1:
                print(f"Save path: {savename}")
            if not args.test:
                with open(savename, "w") as f:
                    f.write("")

            X = np.take(X_og, elec, args.elec_axis).squeeze()
            X_test = np.take(X_test_og, elec, args.elec_axis).squeeze()

            if args.test:
                pattern = [0, 1, 180, 181, 500, 501, 360, 361]
                print(f"data shapes train: {X.shape}, test: {X_test.shape}")
                print(f"example labels train: {y[pattern]}, test: {y_test[pattern]}")
                print(
                    f"example groups train: {groups[pattern]}, test: {groups_test[pattern]}"
                )

            train_set = X, y, groups
            test_set = X_test, y_test, groups_test
            param, train, test = classif(train_set, test_set, args)

            if not args.test:
                if args.clf in ["perceptron", "RF", "SVM", "XGBoost"]:
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
    elif args.elec.startswith("MEG"):
        elec_index = [CHAN_DF.index[CHAN_DF["ch_name"] == args.elec]]
    elif int(args.elec) in list(range(306)):
        elec_index = [int(args.elec)]

    NORM = False
    if args.clf == "SVM":
        NORM = True

    if args.test:
        print("Testing")
        args.n_crossval = 2
        args.iterations = 2
        args.time = True
        elec = np.random.choice(list(range(len(CHAN_DF["ch_name"]))), 1)
        elec_list = check_classif_done([elec], args)
        for clf in ["XGBoost", "SVM", "LDA", "QDA", "RF"]:
            for label in ["subject", "age", "gender"]:
                for feature in ["bins", "bands"]:
                    args.clf = clf
                    args.label = label
                    args.feature = feature
                    if args.verbose > 0:
                        print_info_classif(args)
                    train_set, test_set = load_data(elec_index, args)
                    classif_all_elecs(train_set, test_set, elec_list=[elec], args=args)
    else:
        if args.verbose > 0:
            print_info_classif(args)
        if args.time:
            start = time.time()
        elec_list = check_classif_done(elec_index, args)
        if elec_list != []:
            train_set, test_set = load_data(elec_index, args)
            classif_all_elecs(train_set, test_set, elec_list=elec_list, args=args)
        else:
            print("This classification has already been done")
        if args.verbose > 0 and args.time:
            print("Total time:", elapsed_time(start, time.time()))
