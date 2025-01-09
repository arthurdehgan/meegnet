import os
import logging
import configparser
import numpy as np
from meegnet.dataloaders import EpochedDataset, ContinuousDataset
from meegnet.parsing import parser, save_config
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from meegnet.utils import compute_psd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

LOG = logging.getLogger("meegnet")


def process_sensor(sensor_type, sensor, train_index, test_index, data, labels, fs):
    LOG.info(f"Processing sensor_type {sensor_type}, sensor {sensor}")

    # Compute PSD for the specific sensor
    psd_data = compute_psd(data[:, sensor_type, sensor], fs=fs)

    # Split data into train and test sets
    X_train, y_train = psd_data[train_index], labels[train_index]
    X_test, y_test = psd_data[test_index], labels[test_index]

    param_distributions = {
        "C": np.logspace(-2, 2, 10),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [100, 200, 500, 1000],
        "tol": [1e-4, 1e-3, 1e-2, 1e-1],
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"],
    }

    # Logistic Regression model
    model = LogisticRegression()

    # Randomized Search
    random_search = RandomizedSearchCV(
        model, param_distributions, n_iter=100, cv=5, scoring="accuracy", random_state=42
    )
    random_search.fit(X_train, y_train)

    # Best model parameters and validation accuracy
    best_params = random_search.best_params_
    val_accuracy = random_search.best_score_

    # Train the best model on the full training set
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Evaluate on the training set
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    results = {
        "sensor_type": sensor_type,
        "sensor": sensor,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_parameters": best_params,
    }
    LOG.info(f"Finished processing sensor_type {sensor_type}, sensor {sensor}")
    return results


if __name__ == "__main__":

    ###############
    ### PARSING ###
    ###############

    args = parser.parse_args()
    save_config(vars(args), args.config)

    script_path = os.getcwd()
    config_path = os.path.join(script_path, "../default_values.ini")
    default_values = configparser.ConfigParser()
    assert os.path.exists(config_path), "default_values.ini not found"
    default_values.read(config_path)
    default_values = default_values["config"]

    fold = None if args.fold == -1 else int(args.fold)
    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)

    ####################
    ### LOADING DATA ###
    ####################

    if args.epoched:
        dataset = EpochedDataset(
            sfreq=args.sfreq,
            n_subjects=args.max_subj,
            n_samples=n_samples,
            sensortype=args.sensors,
            lso=args.lso,
        )
    else:
        dataset = ContinuousDataset(
            window=args.segment_length,
            overlap=args.overlap,
            sfreq=args.sfreq,
            n_subjects=args.max_subj,
            n_samples=n_samples,
            sensortype=args.sensors,
            lso=args.lso,
        )

    dataset.load(args.save_path)
    data, labels = dataset.data, dataset.labels

    # Split data into train and test sets
    train_index, test_index, _ = dataset.split_data(0.9, 0.1, 0)

    ########################
    ### START PROCESSING ###
    ########################

    LOG.info("Starting parallel processing...")
    all_results = Parallel(n_jobs=-1)(
        delayed(process_sensor)(
            sensor_type, sensor, train_index, test_index, data, labels, args.sfreq
        )
        for sensor_type in [0, 1, 2]
        for sensor in range(102)
    )

    #######################
    ### FIND BEST RESULT ###
    #######################

    best_result = max(all_results, key=lambda x: x["validation_accuracy"])
    LOG.info("Best sensor combination:")
    LOG.info(best_result)

    #######################
    ### SAVING RESULTS ###
    #######################

    output_file = os.path.join(args.save_path, f"baseline_performance.npy")
    np.save(output_file, all_results)

    LOG.info("Performance metrics saved.")
