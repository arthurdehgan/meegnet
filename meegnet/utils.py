import os
import logging
import torch
import mne
import numpy as np
import pandas as pd
from mne.time_frequency.multitaper import psd_array_multitaper
from scipy.io import loadmat
from scipy.signal import welch

LOG = logging.getLogger("meegnet")
mne.set_log_level(False)


def extract_bands(data: np.array, f: list = None) -> np.array:
    """extract_bands.

    Parameters
    ----------
    data : np.array
        the data after it has been transformed to frequency space. Of shape n_samples x n_channels x n_bins
        or n_channels x n_bins
    f : list
        the list of bins. id set to None, a list of all bins every .5 from 0 to n_bins will be generated.

    Returns
    -------
    data : np.array
        the data after averaging for frequency bands of shape n_samples x n_channels x 7 or n_channels x 7 depending
        on input shape.
        bands are defined as follow:
            delta: .5 to 4 Hz
            theta: 4 to 8 Hz
            alpha: 8 to 12 Hz
            beta1: 12 to 30 Hz
            beta2: 30 to 60 Hz
            gamma1: 60 to 90 Hz
            gamma2: 90 to 120 Hz

    """
    if f is None:
        f = np.asarray([float(i / 2) for i in range(data.shape[-1])])
    data = [
        data[..., (f >= 0.5) * (f <= 4)].mean(axis=-1)[..., None],
        data[..., (f >= 4) * (f <= 8)].mean(axis=-1)[..., None],
        data[..., (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[..., (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[..., (f >= 30) * (f <= 60)].mean(axis=-1)[..., None],
        data[..., (f >= 60) * (f <= 90)].mean(axis=-1)[..., None],
        data[..., (f >= 90) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=-1)
    return data


def compute_psd(data: np.array, fs: int, option: str = "multitaper"):
    """compute_psd.

    Parameters
    ----------
    data : np.array
        The data to compute psd on. Must be of shape n_channels x n_samples.
    fs : int
        The sampling frequency.
    option : str
        method option for the psd computation. Can be welch or multitaper.
    """
    """"""
    mne.set_log_level(verbose=False)
    if option == "multitaper":
        psd, f = psd_array_multitaper(data, sfreq=fs, fmax=150)
    elif option == "welch":
        f, psd = welch(data, fs=fs)
    else:
        raise "Error: invalid option for psd computation."
    return extract_bands(psd, f)


def stratified_sampling(data, targets, n_samples, subject=None, groups=None):
    """
    Performs stratified sampling for a single subject.

    Parameters
    ----------
    data : torch.Tensor
        The dataset containing all subjects' data.
    targets : torch.Tensor
        The target labels corresponding to the data.
    n_samples : int
        The number of samples to draw. If None, all samples are used in a stratified manner.
    groups : torch.Tensor, optional
        The group labels indicating the subject for each sample.
    subject : int or str, optional
        The subject ID for which to perform stratified sampling.

    Returns
    -------
    sampled_indices : torch.Tensor
        The indices of the stratified sampled data for the subject.
    """
    # Ensure that both `subject` and `groups` are either set or both are None
    assert (subject is None and groups is None) or (
        subject is not None and groups is not None
    ), "Both `subject` and `groups` must be set, or both must be None."

    # Filter data, targets, and groups for the given subject
    if groups is not None:
        subject_index = np.where(groups == subject)[0]
    else:
        subject_index = np.arange(len(data))

    if len(subject_index) == 0:
        LOG.warning(f"No data found for subject {subject}.")
        return torch.empty(0, dtype=torch.long)

    # Get unique classes and their counts
    unique_labels = torch.unique(targets[subject_index])
    if len(unique_labels) == 1:
        return subject_index

    # Determine the number of samples per class
    samples_per_class = int(n_samples / len(unique_labels))

    # Perform stratified sampling
    sampled_indices = []
    for label in unique_labels:
        class_indices = subject_index[targets[subject_index] == label]
        n_cls_samples = min(len(class_indices), samples_per_class)  # Avoid oversampling
        sampled_indices.extend(
            class_indices[torch.randperm(len(class_indices))[:n_cls_samples]].tolist()
        )

    return sampled_indices


def cuda_check():
    """
    Checks if a CUDA device is available and returns it.

    Returns
    -------
    torch.device
        A CUDA device if one is available, otherwise a CPU device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        LOG.info("CUDA is available! Using GPU.")
    else:
        LOG.info("CUDA is not available. Using CPU.")
    return device


def check_PD(mat):
    if len(mat.shape) > 2:
        out = []
        for submat in mat:
            out.append(check_PD(submat))
        return np.array(out)
    else:
        return np.all(np.linalg.eigvals(mat) > 0)


def strip_string(string: str) -> str:
    return string.translate({ord(i): None for i in "!@#$[]'"})


def load_psd_cc_subjects(PSD_PATH, sub_info_path, window, overlap):
    df = pd.read_csv(sub_info_path)
    sub_list = df["Observations"].tolist()
    labels = list(df["gender_code"] - 1)
    psd = []
    for sub in sub_list:
        file_path = os.path.join(PSD_PATH, "PSD_{}_{}_{}".format(sub, window, overlap))
        try:
            psd.append(loadmat(file_path)["data"].ravel())
        except IOError:
            LOG.info(sub, "Not Found")
    return np.array(psd), np.array(labels)


def string_to_int(array, target_labels=None):
    """
    Converts an array of strings to integers based on unique labels.

    Parameters
    ----------
    array : array-like
        The input array of strings to be converted.
    target_labels : list, optional
        A predefined list of unique labels. If None, unique labels will be inferred from the input array.

    Returns
    -------
    tuple
        A tuple containing:
        - An array of integers corresponding to the input strings.
        - A list of unique labels used for the mapping.

    Raises
    ------
    ValueError
        If the input array is empty or if `target_labels` contains duplicates.
    """
    # Ensure the input is a numpy array
    array = np.asarray(array)

    if array.size == 0:
        raise ValueError("Input array is empty.")

    # Infer unique labels if not provided
    if target_labels is None:
        target_labels = np.unique(array)
    else:
        # Ensure target_labels is unique
        if len(set(target_labels)) != len(target_labels):
            raise ValueError("`target_labels` must contain unique values.")

    # Create a mapping from labels to integers
    label_to_int = {label: idx for idx, label in enumerate(target_labels)}

    # Map the input array to integers
    int_array = np.vectorize(label_to_int.get)(array)

    return int_array.astype(int), target_labels


def nice_time(time):
    """Returns time in a humanly readable format."""
    m, h, j = 60, 3600, 24 * 3600
    nbj = time // j
    nbh = (time - j * nbj) // h
    nbm = (time - j * nbj - h * nbh) // m
    nbs = time - j * nbj - h * nbh - m * nbm
    if time > m:
        if time > h:
            if time > j:
                nt = "%ij, %ih:%im:%is" % (nbj, nbh, nbm, nbs)
            else:
                nt = "%ih:%im:%is" % (nbh, nbm, nbs)
        else:
            nt = "%im:%is" % (nbm, nbs)
    elif time < 1:
        nt = "<1s"
    else:
        nt = "%is" % nbs
    return nt


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
    return nice_time(lapsed)
