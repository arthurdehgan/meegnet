"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from path import Path as path
from joblib import Parallel, delayed
import numpy as np
# from pyriemann.estimationmod import CospCovariances
from pyriemann.estimation import CospCovariances
from scipy.io import savemat, loadmat
from mne.io import read_raw_fif
from cc_params import SAVE_PATH, SUBJECT_LIST, \
                      FREQ_DICT, SF, WINDOW, OVERLAP

SAVE_PATH += 'crosspectre/'


def reshape_data(data):
    data = data.reshape(0, data.shape[0], data.shape[1])
    temp = np.zeros((17, 306, 30000))
    for i in range(1, 18):
        temp[i] = data[1, :, i*30000:(i+1)*30000]
    return temp


def load_subject(subject_file):
    return read_raw_fif(subject_file, preload=True).pick_types(
           meg=True)[:][0]


def combine_subjects(state, freq, window, overlap):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    for sub in SUBJECT_LIST:
        # file_path = path(SAVE_PATH / 'im_cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
        file_path = path(SAVE_PATH / 'cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
            sub, state, freq, window, overlap))
        try:
            data = loadmat(file_path)['data']
            dat.append(data)
            load_list.append(str(file_path))
        except IOError:
            print(file_path, "not found")
        path_len = len(SAVE_PATH)
    # savemat(file_path[:path_len + 7] + file_path[path_len + 11:],
    savemat(file_path[:path_len + 4] + file_path[path_len + 8:],
            {'data': np.asarray(dat)})
    for f in load_list:
        os.remove(f)


def compute_cosp(sub, freq, window, overlap):
    """Computes the crosspectrum matrices per subjects."""
    print(sub, freq)
    freqs = FREQ_DICT[freq]
    # file_path = path(SAVE_PATH / 'im_cosp_{}_{}_{}_{:.2f}.mat'.format(
    file_path = path(SAVE_PATH / 'cosp_{}_{}_{}_{:.2f}.mat'.format(
        sub, freq, window, overlap))

    if not file_path.isfile():
        # data must be of shape n_trials x n_elec x n_samples
        subject_file = path('{}_rest_raw.fif'.format(sub))
        if subject_file.isfile():
            data = reshape_data(load_subject(subject_file))
            print(data)
            cov = CospCovariances(window=window, overlap=overlap,
                                  fmin=freqs[0], fmax=freqs[1], fs=SF)
            mat = cov.fit_transform(data)
            if len(mat.shape) > 3:
                mat = np.mean(mat, axis=-1)

            savemat(file_path, {'data': mat})
        else:
            print(subject_file, 'Not Found')


if __name__ == '__main__':
    Parallel(n_jobs=1)(delayed(compute_cosp)(
        sub, freq, WINDOW, OVERLAP)
                        for sub in SUBJECT_LIST
                        for freq in FREQ_DICT)
    """
    Parallel(n_jobs=-1)(delayed(combine_subjects)(
        state, freq, WINDOW, OVERLAP)
                            for freq in FREQ_DICT)
                            """
