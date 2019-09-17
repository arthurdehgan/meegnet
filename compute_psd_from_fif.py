"""Computes PSD vectors and save them.

Computes PSD for each frequency from meg fif files and saves.
Outputs one PSD file per subject.

Author: Arthur Dehgan
"""
import os
from itertools import product
import numpy as np
from scipy.io import savemat
from scipy.signal import welch
from joblib import Parallel, delayed
from mne.io import read_raw_fif
from params import SUBJECT_LIST


SF = 1000  # Sampling Frequency
WINDOW = 3000  # windows for computation of PSD
OVERLAP = 0.75  # overlap for computation of PSD (0 = no overlap)
SAVE_PATH = "/home/arthur/data/camcan/spectral"
DATA_PATH = "/home/arthur/data/camcan/data"
# DATA_TYPES = ["rest", "passive", "smt"]
# CLEAN_TYPES = ["mf", "transdef_mf", "raw"]
DATA_TYPES = ["rest"]
CLEAN_TYPES = ["mf"]


def load_subject(subject_file):
    return read_raw_fif(subject_file, preload=True).pick_types(meg=True)[:][0]


def compute_psd(signal, segment_size, sf, window, overlap):
    signal_length = signal.shape[1]
    n_segs = int(signal_length / segment_size)
    padding = int((signal_length - n_segs * segment_size) / 2)

    data = []
    for i in range(padding, signal_length, segment_size):
        segment = signal[:, i : i + segment_size]
        f, psd = welch(segment, fs=sf, window="hamming", nperseg=window, nfft=None)
        data.append(psd[:, (f >= 0) * (f <= 120)])
        f = f[(f >= 0) * (f <= 120)]
    if data[-1].shape[1] != data[0].shape[1]:
        return f, np.asarray(data[:-1])
    else:
        return f, np.asarray(data)


def compute_save_psd(
    save_path, data_type, clean_type, segment_size, subject, sf, window, overlap
):
    fname = f"sub-{subject}_ses-{data_type}_task-{data_type}"
    if clean_type != "raw":
        fname += "_proc-sss"
    subject_path = f"{DATA_PATH}/meg_{data_type}_{clean_type}/sub-{subject}"
    subject_file = f"{subject_path}/ses-{data_type}/meg/{fname}.fif"
    out_path = f"{save_path}/{clean_type}"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_file = f"{out_path}/{subject}_{data_type}_{clean_type}_psd.npy"

    if not os.path.exists(out_file):
        try:
            data = load_subject(subject_file)
            f, psd = compute_psd(data, segment_size, sf, window, overlap)
            np.save(out_file, psd)
        except:
            print(subject_file, "doesnt exist")


if __name__ == "__main__":
    """Main function."""
    Parallel(n_jobs=3)(
        delayed(compute_save_psd)(
            SAVE_PATH,
            data_type,
            clean_type,
            30000,
            subject,
            sf=SF,
            window=WINDOW,
            overlap=OVERLAP,
        )
        for subject, data_type, clean_type in product(
            SUBJECT_LIST, DATA_TYPES, CLEAN_TYPES
        )
    )
