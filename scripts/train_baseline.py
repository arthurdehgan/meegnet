import logging
from meegnet.dataloaders import EpochedDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import os
from meegnet.utils import compute_psd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

data_path = "/home/arthur/data/camcan/eventclf"
fs = 500

# Define function to process a single sensor


def process_sensor(sensor_type, sensor, data_slice, labels, fs):
    logging.info(f"Processing sensor_type {sensor_type}, sensor {sensor}")

    # Compute PSD for the specific sensor
    psd_data = compute_psd(data_slice, fs=fs)

    # Split data into train and test sets
    n_samples = len(labels)
    train_size = int(0.9 * n_samples)
    train_index = np.arange(train_size)
    test_index = np.arange(train_size, n_samples)

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
    logging.info(f"Finished processing sensor_type {sensor_type}, sensor {sensor}")
    return results


if __name__ == "__main__":
    # Initialize dataset
    dataset = EpochedDataset(
        sfreq=fs,
        n_subjects=50,
        n_samples=None,
        sensortype="ALL",
        lso=True,
    )

    dataset.load(data_path)

    # Extract data and labels once
    data, labels = dataset.data, dataset.labels

    # Run parallel processing using joblib
    logging.info("Starting parallel processing...")
    all_results = Parallel(n_jobs=-1)(
        delayed(process_sensor)(
            sensor_type, sensor, data[:, sensor_type, sensor].clone(), labels, fs
        )
        for sensor_type in [0, 1, 2]
        for sensor in range(102)
    )

    # Save all results
    output_file = "model_performance.npy"
    np.save(output_file, all_results)

    logging.info("Performance metrics saved:")
    logging.info(all_results)
