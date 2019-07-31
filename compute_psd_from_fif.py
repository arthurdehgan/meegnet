"""Computes PSD vectors and save them.

Computes PSD for each frequency from meg fif files and saves.
Outputs one PSD file per subject.

Author: Arthur Dehgan
"""
import numpy as np
from scipy.io import savemat
from scipy.signal import welch
from joblib import Parallel, delayed
from mne.io import read_raw_fif
from params import SUBJECT_LIST


SF = 1000  # Sampling Frequency
WINDOW = 3000  # windows for computation of PSD
OVERLAP = 0.25  # overlap for computation of PSD (0 = no overlap)
SAVE_PATH = "/home/arthur/data/camcan/spectral"
DATA_PATH = "/home/arthur/data/raw_camcan/data/data"


def load_subject(subject_file):
    return read_raw_fif(subject_file, preload=True).pick_types(meg=True)[:][0]


def compute_psd(signal, sf, window, overlap):
    f, psd = welch(
        signal, fs=sf, window="hamming", nperseg=window, noverlap=overlap, nfft=None
    )
    psd = psd[(f >= 0) * (f <= 150)]
    f = f[(f >= 0) * (f <= 150)]
    return f, psd


def compute_save_psd(save_path, datatype, subject, sf, window, overlap):
    subject_file = f"{DATA_PATH}/{subject}/{datatype}/{datatype}_raw.fif"
    out_file = f"{save_path}/{subject}_{datatype}_psd.npy"
    data = load_subject(subject_file)
    f, psd = compute_psd(data, sf, window, overlap)
    np.save(out_file, psd)


if __name__ == "__main__":
    """Main function."""
    Parallel(n_jobs=-1)(
        delayed(compute_save_psd)(
            SAVE_PATH, "passive", subject, sf=SF, window=WINDOW, overlap=OVERLAP
        )
        for subject in SUBJECT_LIST
    )
