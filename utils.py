"""Functions used to compute and analyse EEG/MEG data with pyriemann."""
from scipy.io import loadmat
from time import time
import numpy as np
import pandas as pd
from path import Path as path


def load_psd_cc_subjects(PSD_PATH, sub_info_path, window, overlap):
    df = pd.read_csv(sub_info_path)
    sub_list = df["Observations"].tolist()
    labels = list(df["gender_code"] - 1)
    psd = []
    for sub in sub_list:
        file_path = path(PSD_PATH) / "PSD_{}_{}_{}".format(sub, window, overlap)
        try:
            psd.append(loadmat(file_path)["data"].ravel())
        except IOError:
            print(sub, "Not Found")
    return np.array(psd), np.array(labels)


def elapsed_time(t0, t1):
    """Time lapsed between t0 and t1.

    Returns the time (from time.time()) between t0 and t1 in a
    more readable fashion.

    Parameters
    ----------
    t0: float,
        time.time() initial measure of time
        (eg. at the begining of the script)
    t1: float,
        time.time() time at the end of the script
        or the execution of a function.

    """
    lapsed = abs(t1 - t0)
    m, h, j = 60, 3600, 24 * 3600
    nbj = lapsed // j
    nbh = (lapsed - j * nbj) // h
    nbm = (lapsed - j * nbj - h * nbh) // m
    nbs = lapsed - j * nbj - h * nbh - m * nbm
    if lapsed > m:
        if lapsed > h:
            if lapsed > j:
                Time = "%ij, %ih:%im:%is" % (nbj, nbh, nbm, nbs)
            else:
                Time = "%ih:%im:%is" % (nbh, nbm, nbs)
        else:
            Time = "%im:%is" % (nbm, nbs)
    elif lapsed < 1:
        Time = "<1s"
    else:
        Time = "%is" % nbs
    return Time
