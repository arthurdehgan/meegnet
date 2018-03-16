from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as data_split
from itertools import combinations
from mne.io import read_raw_fif
from params import DATA_PATH, SUBJECT_LIST, SAVE_PATH, CHAN_DF, LABELS, MODEL_PATH
tf.logging.set_verbosity(tf.logging.INFO)


N_SUB_PER_BATCH = 10
SAVE_PATH = SAVE_PATH


def load_subject(sub, data=None, timepoints=2000, ch_type='all'):
    df = pd.read_csv('{}/clean_camcan_participant_data.csv'.format(SAVE_PATH))
    df = df.set_index('Observations')
    gender = (df['gender_code'] - 1)[sub]
    # subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, sub)
    subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, sub)
    trial = read_raw_fif(subject_file, preload=True).pick_types(meg=True)[:][0]
    if ch_type == 'all':
        mask = [True for _ in range(len(trial))]
        n_channels = 306
    elif ch_type == 'mag':
        mask = CHAN_DF['mag_mask']
        n_channels = 102
    elif ch_type == 'grad':
        mask = CHAN_DF['grad_mask']
        n_channels = 204
    else:
        raise('Error : bad channel type selected')
    trial = trial[mask]
    n_trials = trial.shape[-1] // timepoints
    for i in range(1, n_trials - 1):
        curr = trial[:, i*timepoints:(i+1)*timepoints]
        curr = curr.reshape(1, n_channels, timepoints)
        data = curr if data is None else np.concatenate((data, curr))
    labels = [gender] * (n_trials - 2)
    data = data.astype(np.float32, copy=False)
    return data, labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    a, b, c = features['x'].shape
    input_layer = tf.reshape(features["x"], [a, b, c, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    # Dropout layer #1
    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2 and Pooling Layer #2 and Dropout layer #2
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=16,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #3 and Pooling Layer #3 and Dropout layer #3
    conv3 = tf.layers.conv2d(
        inputs=dropout2,
        filters=32,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    dropout3 = tf.layers.dropout(
        inputs=pool3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Dense Layer
    dropout3_flat = tf.reshape(dropout3, [-1, pool3.shape[1]
                                    * pool3.shape[2]
                                    * pool3.shape[3]])
    layer_size = pool3_flat.shape[-1]
    dense = tf.layers.dense(inputs=dropout3_flat, units=layer_size,
                            activation=tf.nn.relu)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT
        # and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def eval(eval_sub_list):
    eval_data = None
    eval_labels = []
    for sub in test_sub_list:
        if eval_data is not None:
            eval_data, temp_labels = load_subject(sub, eval_data)
            eval_labels += temp_labels
        else:
            eval_data, eval_labels = load_subject(sub)
    eval_labels = np.array(eval_labels)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = camcan_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def train(train_sub_list):
    train_labels = []
    train_data = None
    for sub in train_sub_list:
        if train_data is not None:
            train_data, temp_labels = load_subject(sub,
                                                   train_data,
                                                   ch_type='mag')
            train_labels += temp_labels
        else:
            train_data, train_labels = load_subject(sub,
                                                    ch_type='mag')
    train_labels = np.array(train_labels)
    camcan_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=MODEL_PATH)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True)
    camcan_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])


def main(unused_argv):
    X_train, X_test = data_split(SUBJECT_LIST,
                                                  shuffle=True,
                                                  stratify=LABELS,
                                                  random_state=0,
                                                  test_size=.1)
    train_sub_list = combinations(X_train, N_SUB_PER_BATCH)
    test_sub_list = combinations(X_test, N_SUB_PER_BATCH)

    for sub_list in train_sub_list:
        train(sub_list)

    for sub_list in test_sub_list:
        eval(sub_list)


if __name__ == "__main__":
    tf.app.run()
