"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from pyriemann.estimation import CospCovariances
from pyriemann.estimation import Covariances
import argparse


def split_data(dat, df, sf, offset=10):
    """By default we skip the first 10s of the data.
    Change to 0 if you want to use all data

    This function is adapted for ELEKTA systems with 2 planar gradiometers and 1 magnetometer
    at each channel location. We split the channels for cov and cosp computations.

    input data must be of shape n_elec x n_samples
    Output is of shape 3 x n_trials x n_elec x n_samples for covariance computations.

    """
    data = []
    for begin, end in zip(df["begin"], df["end"]):
        # the dataframe countains the segment timestamps for the ds200 data. Here we plan on using 1000Hz data so we multiply the timespamps by 5.
        begin *= 5
        end *= 5
        seg = dat[:306, begin:end]
        if (
            seg.shape[-1] == end - begin
            and begin >= offset * sf
            and not np.isnan(seg).any()
        ):
            try:
                data.append(seg)
            except:
                continue
    data = np.array(data)
    # Splitting channels
    data = (
        data[:, np.arange(0, 306, 3), :],
        data[:, np.arange(1, 306, 3), :],
        data[:, np.arange(2, 306, 3), :],
    )

    return np.array(data)


def load_and_compute(
    data_path, save_path, sub_inf, bands, window, overlap, sf, dattype, feature
):
    sub_df = pd.read_csv(sub_inf, index_col=0)
    subs = list(sub_df["subs"])
    for some_file in os.listdir(data_path):
        sub = some_file.split("_")[0]
        if dattype in some_file and sub in subs:
            savename = save_path + f"{sub}_{dattype}_{feature}.npy"
            if not os.path.exists(savename):
                data = np.load(data_path + some_file)
                data = split_data(data.T, sub_df[sub_df["subs"] == sub], sf, 10)
                channels = []
                for channel in data:
                    if feature == "cosp":
                        cov = CospCovariances(
                            window=window,
                            overlap=overlap,
                            fmin=1,
                            fmax=50,
                            fs=sf,
                        )
                        mat = cov.fit_transform(channel)
                        mat = np.array(
                            [
                                mat[:, :, :, :4].mean(axis=-1),
                                mat[:, :, :, 4:8].mean(axis=-1),
                                mat[:, :, :, 8:12].mean(axis=-1),
                                mat[:, :, :, 12:30].mean(axis=-1),
                                mat[:, :, :, 30:].mean(axis=-1),
                            ]
                        )
                        mat = np.swapaxes(mat, 0, 1)
                    elif feature == "cov":
                        mat = Covariances(estimator="lwf").fit_transform(channel)
                    channels.append(mat)
                final = np.array(channels)
                np.save(savename, final)


if __name__ == "__main__":
    bands = {
        "delta": (0, 4),
        "theta": (4, 8),
        "alpha": (8, 15),
        "beta": (15, 25),
        "gamma": (25, 50),
    }

    ###########
    # PARSING #
    ###########

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path where the data will be saved.",
    )
    parser.add_argument(
        "--sub_info_path",
        type=str,
        help="The path to the csv containing the info on the timestamps for the data segments.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path where the data samples can be found.",
    )
    parser.add_argument(
        "--overlap",
        default=0.5,
        type=float,
        help="Overlap parameter for ferquency domain transformation",
    )
    parser.add_argument(
        "--window",
        default=1000,
        type=int,
        help="Window parameter for ferquency domain transformation",
    )
    parser.add_argument(
        "--sf",
        default=1000,
        type=int,
        help="Sampling frequency of the data, necessary for frequency domain transformation",
    )
    parser.add_argument(
        "--feature_type",
        default="cosp",
        choices=["cosp", "cov"],
        help="the type of features thwt you want to compute",
    )
    parser.add_argument(
        "--dattype",
        default="rest",
        choices=["rest", "task", "passive"],
        help="the type of data to be loaded",
    )
    args = parser.parse_args()
    data_path = args.data_path
    if not data_path.endswith("/"):
        data_path += "/"
    save_path = args.save_path
    if not save_path.endswith("/"):
        save_path += "/"
    sub_inf = args.sub_info_path
    feature = args.feature_type
    overlap = args.overlap
    sf = args.sf
    window = args.window
    dattype = args.dattype

    load_and_compute(
        data_path, save_path, sub_inf, bands, window, overlap, sf, dattype, feature
    )
