"""Computes PSD vectors and save them.

Computes PSD for each frequency from meg fif files and saves.
Outputs one PSD file per subject.

Author: Arthur Dehgan
"""
from scipy.io import savemat
from scipy.signal import welch
from joblib import Parallel, delayed
from mne.io import read_raw_fif
from params import DATA_PATH, SAVE_PATH, CAMCAM_SUBJECT_LIST


SF = 1000  # Sampling Frequency
WINDOW = 3000  # windows for computation of PSD
OVERLAP = 0.25  # overlap for computation of PSD (0 = no overlap)
SUBJECT_LIST = CAMCAM_SUBJECT_LIST


def load_subject(subject_file):
    return read_raw_fif(subject_file, preload=True).pick_types(
        meg=True)[:][0]


def compute_psd(signal, sf, window, overlap):
    f, psd = welch(signal, fs=sf, window='hamming', nperseg=window,
                   noverlap=overlap, nfft=None)
    psd = psd[(f >= 0)*(f <= 200)]
    f = f[(f >= 0)*(f <= 200)]
    return f, psd


def compute_save_psd(save_path, subject, sf, window, overlap):
    out_file = '{}/PSD_{}_{}_{:.2f}.mat'.format(
        save_path, subject, window, overlap)
    subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, subject)
    data = load_subject(subject_file)
    f, psd = compute_psd(data, sf, window, overlap)
    savemat(out_file, {'data': psd})
    return 0


if __name__ == '__main__':
    """Main function."""
    Parallel(n_jobs=-1)(delayed(compute_save_psd)(SAVE_PATH,
                                                  subject,
                                                  sf=SF,
                                                  window=WINDOW,
                                                  overlap=OVERLAP)
                        for subject in SUBJECT_LIST)
