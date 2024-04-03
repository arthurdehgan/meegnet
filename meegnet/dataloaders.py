from __future__ import print_function, division
import os
import random
import logging
from ast import literal_eval
import torch
import psutil
import pandas as pd
import numpy as np
from scipy.stats import zscore
from torch.utils.data import DataLoader, random_split, TensorDataset

BANDS = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"]


# From Domainbed, modified for my use case
class _InfiniteSampler(torch.utils.data.Sampler):
    """
    Wraps another Sampler to yield an infinite stream.

    Parameters
    ----------
    sampler : torch.utils.data.Sampler
        The sampler to be wrapped.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    """
    Creates an infinite data loader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be loaded.
    batch_size : int
        The size of the batches to be loaded.
    num_workers : int, optional
        The number of workers to use for loading the data, by default 0.
    pin_memory : bool, optional
        If True, the data loader will copy tensors into CUDA pinned memory before returning them. This can make data
        transfer faster, but requires more memory.
    weights : torch.Tensor, optional
        A 1D tensor assigning a weight to each sample in the dataset. If not provided, all samples are assumed to have
        the same weight.

    Returns
    -------
    InfiniteDataLoader
        An infinite data loader.
    """

    def __init__(self, dataset, batch_size, num_workers=0, pin_memory=False, weights=None):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        if weights is None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                pin_memory=pin_memory,
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


# End of domainBed code


def assert_params(band, datatype):
    if datatype not in ["rest", "passive", "task"]:
        logging.error(f"Incorrect data type: {datatype}. Must be in (rest, passive, task)")
    return


def create_datasets(
    data_path,
    train_size,
    max_subj=1000,
    chan_index=[0, 1, 2],
    sfreq=200,
    seg=0.8,
    debug=False,
    seed=0,
    printmem=False,
    ages=(0, 100),
    datatype="rest",
    band="",
    n_samples=None,
    clf_type="",
    epoched=False,
    testing=None,
    psd=False,
):
    """
    Create dataloader iterators.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the data.
    train_size : float
        Fraction of the total dataset to be used for training.
    max_subj : int, optional
        Maximum number of subjects to be included in the dataset. Default is 1000.
    chan_index : list, optional
        List of channel indices to be considered. Default is [0, 1, 2].
        0 being MAG channel, 1 the first GRAD channel and 2 the second GRAD channel.
    sfreq : int, optional
        Sampling frequency. Default is 200.
    seg : int, optional
        Segment size in seconds. Default is .8.
    debug : bool, optional
        Whether to enable debugging mode. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    printmem : bool, optional
        Whether to print memory usage. Default is False.
    ages : tuple, optional
        Age range of the participants to be included in the dataset. Default is (0, 100).
    datatype : str, optional
        Type of data to be loaded. Default is "rest".
        other options are "passive" and "smt"
    band : str, optional
        Frequency band to be considered. Default is "".
        Frequency band must be one of ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"]
    n_samples : int, optional
        Number of samples to load. Default is None.
    clf_type : str, optional
        if set to "eventclf", perform event-wise classification.
    epoched : bool, optional
        Whether to epoch the data. Default is False.
    testing : int, optional
        If set to an integer between 0 and 4, it will leave out 20% of the dataset. Useful for random search. Default is None.
    psd : bool, optional
        Whether to calculate power spectral density. Default is False.

    Returns
    -------
    list
        A list of TensorDatasets.

    Raises
    ------
    ValueError
        If clf_type == "eventclf" and `epoched` is set to False. We need to use epoched data around stimuli
        in order to perform event classification

    Notes
    -----
    The function creates dataloader iterators for a given dataset. The dataset is divided into training, validation, and test sets.
    The division ratio is determined by the `train_size` parameter. The function also handles the removal of subjects that cause issues
    during data loading.
    """
    if clf_type == "eventclf" and not epoched:
        logging.warning(
            "Event classification can only be performed with epoched data. And epoched has bot been set to True. Setting epoched to True."
        )
        epoched = True
    torch.manual_seed(seed)
    # Using trials_df ensures we use the correct subjects that do not give errors since
    # it is created by reading the data. It is therefore better than SUB_DF previously used
    # We now use trials_df_clean that contains one less subjects that contained nans
    folder = "psd" if psd else f"downsampled_{sfreq}"
    csv_file = os.path.join(data_path, f"participants_info_{datatype}.csv")
    participants_df = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    participants_df["age"] = pd.to_numeric(participants_df["age"])

    subs = np.array(
        participants_df[participants_df["age"].between(*ages)].drop(["age"], axis=1)["sub"]
    )

    subs = subs[:max_subj]

    # We noticed that specific subjects were the reason why we couldn't
    # learn anything from the data: TODO we might remove those now that we updated dataset
    if datatype == "passive":
        # forbidden_subs = ["CC620526", "CC220335", "CC320478", "CC410113", "CC620785"]
        forbidden_subs = []
        if len(forbidden_subs) > 0:
            logging.info(f"removed subjects {forbidden_subs}, they were causing problems...")
        for sub in forbidden_subs:
            if sub in subs:
                subs = np.delete(subs, np.where(subs == sub)[0])
    N = len(subs)
    train_size = int(N * train_size)

    # We use train_size of the data for model selection and use test_size data for evaluating best model.
    # train_size data will be split in 4 folds and used to choose the model through 4-Fold CV.
    fold_size = int(train_size / 4)
    test_size = N - fold_size * 4
    indexes = random_split(np.arange(N), [*[fold_size] * 4, test_size])
    logging.info(
        f"Using {N} subjects: {fold_size*3} for train, {fold_size} for validation, and {test_size} for test"
    )

    dataframes = [
        participants_df.loc[participants_df["sub"].isin(subs[index])]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
        for i, index in enumerate(indexes)
    ]

    if testing is not None:
        dataframes = dataframes[:testing] + dataframes[testing + 1 :]

    logging.info("Loading Train Set")
    datasets = [
        TensorDataset(
            *load_data(
                df,
                data_path=data_path,
                chan_index=chan_index,
                printmem=printmem,
                datatype=datatype,
                n_samples=n_samples,
                seed=seed,
                epoched=epoched,
                clf_type=clf_type,
                band=band,
                sfreq=sfreq,
                seg=seg,
                psd=psd,
                group_id=i * len(df),
            )
        )
        for i, df in enumerate(dataframes)
    ]
    return datasets


def load_sets(
    data_path,
    max_subj=1000,
    n_splits=5,
    offset=0,
    seg=0.8,
    sfreq=500,
    n_samples=None,
    chan_index=[0, 1, 2],
    printmem=False,
    datatype="rest",
    epoched=False,
    seed=0,
    band="",
    testing=None,
    psd=False,
):
    """
    Load data subject per subject.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the data.
    max_subj : int, optional
        Maximum number of subjects to be included in the dataset. Default is 1000.
    n_splits : int, optional
        Number of splits to make for the data. Default is 5.
    offset : int, optional
        Offset for the data. Default is 0.
    seg : int, optional
        Segment size in seconds. Default is .8.
    sfreq : int, optional
        Sampling frequency. Default is 200.
    n_samples : int, optional
        Number of samples to load. Default is None.
    chan_index : list, optional
        List of channel indices to be considered. Default is [0, 1, 2].
        0 being MAG channel, 1 the first GRAD channel and 2 the second GRAD channel.
    printmem : bool, optional
        Whether to print memory usage. Default is False.
    datatype : str, optional
        Type of data to be loaded. Default is "rest".
        Other options are "passive" and "smt".
    epoched : bool, optional
        Whether to epoch the data. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    band : str, optional
        Frequency band to be considered. Default is "".
        Frequency band must be one of ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"].
    testing : int, optional
        If set to an integer between 0 and 4, it will leave out 20% of the dataset. Useful for random search. Default is None.
    psd : bool, optional
        Whether to calculate power spectral density. Default is False.

    Returns
    -------
    int
        Number of subjects successfully loaded.
    list
        A list of TensorDatasets.

    Notes
    -----
    This function loads data subject by subject. It divides the data into a specified number of splits. Each subject's data is loaded into a separate split.
    The function also handles the removal of subjects that cause issues during data loading.
    """
    assert_params(band, datatype)

    folder = "psd" if psd else f"downsampled_{sfreq}"
    csv_file = os.path.join(data_path, f"participants_info_{datatype}.csv")
    dataframe = pd.read_csv(csv_file, index_col=0)
    # For some reason this subject makes un unable to learn #TODO might remove those since we changed dataset
    # forbidden_subs = ["CC220901"]
    forbidden_subs = []
    if len(forbidden_subs) > 0:
        logging.info(f"removed subjects {forbidden_subs}, they were causing problems...")

    for sub in forbidden_subs:
        dataframe = dataframe.loc[dataframe["sub"] != sub]

    dataframe = dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)[:max_subj]

    n_sub = len(dataframe)
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        totmem = psutil.virtual_memory().total / 10**9
        logging.info(f"Total Available memory: {totmem:.3f} Go")

    final_n_splits = n_splits - 1 if testing is not None else n_splits
    X_sets = [[] for _ in range(final_n_splits)]
    y_sets = [[] for _ in range(final_n_splits)]
    logging.debug(list(dataframe["sub"]))
    for i, row in enumerate(dataframe.iterrows()):
        random.seed(seed)
        torch.manual_seed(seed)
        if printmem:
            usedmem = psutil.virtual_memory().used / 10**9
            memstate = f"Used memory: {usedmem:.3f} / {totmem:.3f} Go."
            if n_sub > 10:
                if i % (n_sub // 10) == 0:
                    logging.debug(memstate)
            else:
                logging.debug(memstate)

        sub = row[1]["sub"]
        data = load_sub(
            data_path,
            sub,
            n_samples=n_samples,
            band=band,
            chan_index=chan_index,
            datatype=datatype,
            epoched=epoched,
            offset=offset,
            sfreq=sfreq,
            seg=seg,
            psd=psd,
        )
        if data is None:
            n_sub -= 1
            continue

        N = len(data)
        fold_size = int(N / n_splits)
        test_size = N - fold_size * (n_splits - 1)
        indexes = random_split(
            np.arange(N),
            [*[fold_size] * (n_splits - 1), test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        if testing is not None:
            indexes.pop(testing)
        random.shuffle(data)
        labels = np.array([i for _ in range(N)])
        for j, index in enumerate(indexes):
            X_sets[j].append(torch.Tensor(np.array(data))[index])
            y_sets[j].append(torch.Tensor(labels)[index])

    logging.info(
        f"Loaded {final_n_splits} subsets of {fold_size} trials succesfully for {n_sub} subjects\n"
    )

    datasets = []
    for i in range(final_n_splits):
        datasets.append(TensorDataset(torch.cat(X_sets[i], 0), torch.cat(y_sets[i], 0)))

    return n_sub, datasets


def create_loader(
    dataset,
    infinite=False,
    **kwargs,
):
    """
    Creates a DataLoader or InfiniteDataLoader based on the specified conditions.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be loaded.
    infinite : bool, optional
        If set to True, creates an InfiniteDataLoader; otherwise, creates a regular DataLoader. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the DataLoader or InfiniteDataLoader constructor.

    Returns
    -------
    DataLoader or InfiniteDataLoader
        The created DataLoader or InfiniteDataLoader.

    Notes
    -----
    This function is useful when you want to create a DataLoader that can loop over the dataset indefinitely.
    """
    if infinite:
        loader = InfiniteDataLoader
    else:
        loader = DataLoader

    return loader(dataset, **kwargs)


def create_loaders(**kwargs):
    datasets = create_datasets(**kwargs)
    loaders = []
    for dataset in datasets:
        loaders.append(create_loader(dataset, **kwargs))
    return loaders


def load_data(
    dataframe,
    data_path,
    offset=30,
    seg=0.8,
    sfreq=500,
    chan_index=[0, 1, 2],
    printmem=False,
    datatype="rest",
    n_samples=None,
    seed=0,
    epoched=False,
    clf_type="",
    band="",
    psd=False,
    group_id=None,
):
    """
    Loads data subject per subject and returns labels according to the task.

    Parameters
    ----------
    dataframe : DataFrame
        DataFrame containing information about the data.
    data_path : str
        Path to the folder containing the data.
    offset : int, optional
        Offset for the data. Default is 30.
    seg : int, optional
        Segment size in seconds. Default is .8.
    sfreq : int, optional
        Sampling frequency. Default is 200.
    chan_index : list, optional
        List of channel indices to be considered. Default is [0, 1, 2].
        0 being MAG channel, 1 the first GRAD channel and 2 the second GRAD channel.
    printmem : bool, optional
        Whether to print memory usage. Default is False.
    datatype : str, optional
        Type of data to be loaded. Default is "rest".
        Other options are "passive" and "smt".
    n_samples : int, optional
        Number of samples to load. Default is None.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    epoched : bool, optional
        Whether to epoch the data. Default is False.
    clf_type : str, optional
        if set to "eventclf", perform event-wise classification.
    band : str, optional
        Frequency band to be considered. Default is "".
        Frequency band must be one of ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"].
    psd : bool, optional
        Whether to calculate power spectral density. Default is False.
    group_id : int, optional
        Group ID for grouping the data. Default is None.

    Returns
    -------
    tuple
        A tuple containing the loaded data, labels, and group IDs (if applicable).

    Notes
    -----
    The function loads data subject by subject and assigns labels according to the task.
    If clf_type == "eventclf", use labels from the events. Otherwise we use whatever is in the dataframe as default label.
    """

    assert_params(band, datatype)
    if clf_type == "eventclf":
        assert (
            datatype != "rest"
        ), "We can not perform event classification on resting state data as it contains no events and therefore can not be epoched."

    n_sub = len(dataframe)
    X, y, groups = [], [], []
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        # subj_sizes = [] # assigned but never used ?
        totmem = psutil.virtual_memory().total / 10**9
        logging.info(f"Total Available memory: {totmem:.3f} Go")
    for i, row in enumerate(dataframe.iterrows()):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if printmem:
            usedmem = psutil.virtual_memory().used / 10**9
            memstate = f"Used memory: {usedmem:.3f} / {totmem:.3f} Go."
            if n_sub > 10:
                if i % (n_sub // 10) == 0:
                    logging.debug(memstate)
            else:
                logging.debug(memstate)

        sub, lab = row[1]["sub"], row[1]["label"]
        data = load_sub(
            data_path,
            sub,
            n_samples=n_samples,
            band=band,
            chan_index=chan_index,
            datatype=datatype,
            epoched=epoched,
            offset=offset,
            sfreq=sfreq,
            seg=seg,
            psd=psd,
        )
        if data is None:
            n_sub -= 1
            continue

        if n_samples is not None:
            random_samples = np.random.choice(np.arange(len(data)), n_samples, replace=False)
            data = torch.Tensor(data)[random_samples]

        if clf_type == "eventclf":
            events = np.array(
                literal_eval(dataframe.loc[dataframe["sub"] == sub]["event_labels"].item())
            )
            if len(events) != len(data):
                n_sub -= 1
                continue
            y += [0 if e == "visual" else int(e[-1]) for e in events]
        else:
            y += [1 if lab == "FEMALE" else 0] * len(data)
        if group_id is not None:
            groups += [group_id + i] * len(data)

        X.append(torch.as_tensor(np.array(data)))
    logging.info(f"Loaded {n_sub} subjects succesfully\n")

    if len(X) > 0:
        X = torch.cat(X, 0)
    else:
        return None, None
    if len(X.shape) != 4:  # Will happen if only one sensor channel is selected (eg MAG)
        X = X[:, np.newaxis]
    y = torch.as_tensor(y)
    if group_id is not None:
        groups = torch.as_tensor(groups)
        return X, y, groups
    return X, y


def load_epoched_sub(data_path, sub, chan_index, datatype="passive", sfreq=500, psd=False):
    """
    Loads epoched data for a particular subject.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the data.
    sub : str
        Subject identifier.
    chan_index : list
        List of channel indices to be considered.
    datatype : str, optional
        Type of data to be loaded. Default is "passive".
        Other options is "smt".
    sfreq : int, optional
        Sampling frequency. Default is 500.
    psd : bool, optional
        Whether to load power spectral density data. Default is False.

    Returns
    -------
    ndarray or None
        The loaded epoched data for the subject, or None if an error occurred.

    Raises
    ------
    AssertionError
        If `datatype` is not "passive" or "smt".

    Notes
    -----
    This function loads epoched data for a particular subject. The data is loaded from a `.npy` file located in a specific directory depending on whether power spectral density data is requested. The function also performs zero mean normalization on the data if it's not power spectral density data.
    """
    assert datatype in ("passive", "smt"), "cannot load epoched data for resting state"
    folder = "psd" if psd else f"downsampled_{sfreq}"
    sub_path = os.path.join(data_path, folder, f"{datatype}_{sub}_epoched.npy")
    try:
        sub_data = np.load(sub_path)[:, chan_index]
        if not psd:
            sub_data = np.array(list(map(lambda x: zscore(x, axis=-1), sub_data)))
    except IOError:
        logging.warning(f"There was a problem loading subject {sub_path}")
        return None
    except:
        logging.warning(f"An error occured while loading {sub_path}")
        return None

    return sub_data


def load_sub(
    data_path,
    sub,
    n_samples=None,
    band="",
    chan_index=[0, 1, 2],
    datatype="rest",
    epoched=False,
    offset=30,
    sfreq=500,
    seg=0.8,
    psd=False,
):
    """
    Loads data for a particular subject.

    Parameters
    ----------
    data_path : stl
        Path to the folder containing the data.
    sub : str
        Subject identifier.
    n_samples : int, optional
        Number of samples to load. Default is None.
    band : str, optional
        Frequency band to be considered. Default is "".
        Frequency band must be one of ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"].
    chan_index : list, optional
        List of channel indices to be considered. Default is [0, 1, 2].
        0 being MAG channel, 1 the first GRAD channel and 2 the second GRAD channel.
    datatype : str, optional
        Type of data to be loaded. Default is "rest".
        Other options are "passive" and "smt".
    epoched : bool, optional
        Whether to load epoched data. Default is False.
    offset : int, optional
        Offset for the data. Default is 30.
    sfreq : int, optional
        Sampling frequency. Default is 200.
    seg : int, optional
        Segment size in seconds. Default is .8.
    psd : bool, optional
        Whether to load power spectral density data. Default is False.

    Returns
    -------
    ndarray or None
        The loaded data for the subject, or None if an error occurred.

    Notes
    -----
    This function loads data for a particular subject.
    The data is loaded from a `.npy` file located in a specific directory.
    The function also performs zero mean normalization on the data if it's not power spectral density data.
    """
    if epoched:
        data = load_epoched_sub(
            data_path, sub, chan_index, datatype=datatype, sfreq=sfreq, psd=psd
        )
    else:
        try:
            # if domain in ("cov", "cosp"):  # TODO Deprecated
            #     file_path = os.path.join(
            #         data_path, "covariances", f"{sub}_{datatype}_{domain}.npy"
            #     )
            #     data = np.load(file_path)[chan_index]
            #     data = np.swapaxes(data, 0, 1)
            #     if domain == "cosp":
            #         data = data[:, :, BANDS.index(band)]
            # else:
            folder = "psd" if psd else f"downsampled_{sfreq}"
            file_path = os.path.join(data_path, folder, f"{datatype}_{sub}.npy")
            sub_data = np.load(file_path)[chan_index]
            if len(sub_data.shape) < 3:
                sub_data = sub_data[np.newaxis, :]
            if not psd:
                step = int(seg * sfreq)
                start = int(offset * sfreq)
                data = []
                for i in range(start, sub_data.shape[-1], step):
                    trial = sub_data[:, :, i : i + step]
                    if trial.shape[-1] == step:
                        if not np.isnan(trial).any():
                            data.append(zscore(trial, axis=-1))
                if len(data) < 50:
                    return None
                if n_samples is not None and n_samples <= len(data):
                    random_samples = np.random.choice(
                        np.arange(len(data)), n_samples, replace=False
                    )
                    data = torch.Tensor(np.array(data))[random_samples]
            else:
                data = sub_data

        except IOError:
            logging.warning(f"There was a problem loading subject file for {sub}")
            return None
    return data
