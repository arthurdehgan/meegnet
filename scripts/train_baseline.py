import logging
from meegnet.dataloaders import EpochedDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import os
from meegnet.utils import compute_psd
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

data_path = "/home/arthur/data/camcan/eventclf"
fs = 500


def producer(sensor_data_queue, dataset, fs):
    for sensor_type in [0, 1, 2]:
        for sensor in range(102):
            logging.info(f"Processing sensor_type {sensor_type}, sensor {sensor}")
            dataset.data = compute_psd(dataset.data[:, sensor_type, sensor], fs=fs)
            train_index, test_index, _ = dataset.split_data(0.9, 0.1, 0)
            X_train, y_train = dataset.data[train_index], dataset.labels[train_index]
            X_test, y_test = dataset.data[test_index], dataset.labels[test_index]

            sensor_data_queue.put((X_train, y_train, X_test, y_test))
    sensor_data_queue.put(None)  # Signal completion


def consumer(sensor_data_queue, results_queue):
    param_distributions = {
        "C": np.logspace(-2, 2, 10),
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [100, 200, 500, 1000],
        "tol": [1e-4, 1e-3, 1e-2, 1e-1],
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"],
    }

    while True:
        data = sensor_data_queue.get()
        if data is None:
            break

        X_train, y_train, X_test, y_test = data

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
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "best_parameters": best_params,
        }
        results_queue.put(results)


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

    # Prepare multiprocessing queues
    sensor_data_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()

    # Start producer and consumer processes
    producer_process = multiprocessing.Process(
        target=producer, args=(sensor_data_queue, dataset, fs)
    )
    consumer_process = multiprocessing.Process(
        target=consumer, args=(sensor_data_queue, results_queue)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    # Collect results
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())

    # Save all results
    output_file = "model_performance.npy"
    np.save(output_file, all_results)

    logging.info("Performance metrics saved:")
    logging.info(all_results)
