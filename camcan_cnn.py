from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from mne.io import read_raw_fif
from params import DATA_PATH, SUBJECT_LIST, SAVE_PATH, CHAN_DF
tf.logging.set_verbosity(tf.logging.INFO)


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
    elif ch_type == 'mag':
        mask = CHAN_DF['mask_mag']
    elif ch_type == 'grad':
        mask = CHAN_DF['grad_mask']
    else:
        raise('Error : bad channel type selected')
    trial = trial[mask]
    n_trials = trial.shape[-1] // timepoints
    for i in range(1, n_trials - 1):
        curr = trial[:, i*timepoints:(i+1)*timepoints]
        curr = curr.reshape(1, 306, timepoints)
        data = curr if data is None else np.concatenate((data, curr))
    labels = [gender] * (n_trials - 2)
    data = data.astype(np.float32, copy=False)
    return data, labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 306, 5000, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=5,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    print(pool1.shape)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    print(pool2.shape)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[1, 5],
                                    strides=(1, 5))
    print(pool3.shape)
    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, pool3.shape[1]
                                    * pool3.shape[2]
                                    * pool3.shape[3]])
    print(pool3_flat.shape)
    dense = tf.layers.dense(inputs=pool3_flat, units=306000,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

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


def main(unused_argv):
    train_sub_list = SUBJECT_LIST[:200]
    test_sub_list = SUBJECT_LIST[500:]
    eval_data = None
    train_data = None
    train_labels = []
    for sub in train_sub_list:
        if train_data is not None:
            train_data, temp_labels = load_subject(sub,
                                                   train_data,
                                                   ch_type='mag')
            train_labels += temp_labels
        else:
            train_data, train_labels = load_subject(sub)
    train_labels = np.array(train_labels)
    camcan_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/home/kikuko/")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True)
    camcan_classifier.train(
        input_fn=train_input_fn,
        steps=100000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_labels = []
    for sub in test_sub_list:
        if eval_data is not None:
            eval_data, temp_labels = load_subject(sub, eval_data)
            eval_labels += temp_labels
        else:
            eval_data, eval_labels = load_subject(sub)
        print(eval_data.shape, len(eval_labels))
    eval_labels = np.array(eval_labels)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = camcan_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
