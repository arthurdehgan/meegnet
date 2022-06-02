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
from scipy.signal import welch
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split, TensorDataset
from camcan.utils import extract_bands

BANDS = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2", "gamma3"]


# From Domainbed, modified for my use case
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(
        self, dataset, batch_size, num_workers, pin_memory=False, weights=None
    ):
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


def assert_params(band, domain, dattype):
    if domain == "cosp":
        if band == "":
            logging.error(
                "A frequency band must be specified when using co-spectrum matrices"
            )
        elif band not in BANDS:
            logging.error(
                f"{band} is not a correct band option. band must be in {BANDS}"
            )
    if dattype not in ["rest", "passive", "task"]:
        logging.error(
            f"Incorrect data type: {dattype}. Must be in (rest, passive, task)"
        )
    return


def create_datasets(
    data_folder,
    train_size,
    max_subj,
    chan_index,
    domain,
    s_freq=200,
    seg=2,
    debug=False,
    seed=0,
    printmem=False,
    ages=(0, 100),
    dattype="rest",
    band="",
    n_samples=None,
    eventclf=False,
    epoched=False,
    testing=None,
):
    """create dataloaders iterators.

    testing: if set to an integer between 0 and 4 will leave out a part of the dataset.
             Useful for random search.
    """
    if eventclf and not epoched:
        logging.warning(
            "Event classification can only be performed with epoched data. And epoched has bot been set to True. Setting epoched to True."
        )
        epoched = True
    torch.manual_seed(seed)
    # Using trials_df ensures we use the correct subjects that do not give errors since
    # it is created by reading the data. It is therefore better than SUB_DF previously used
    # We now use trials_df_clean that contains one less subjects that contained nans
    csv_filepath = os.path.join(data_folder, f"participants_info_{dattype}.csv")
    participants_df = (
        pd.read_csv(csv_filepath, index_col=0)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    participants_df["age"] = pd.to_numeric(participants_df["age"])

    subs = np.array(
        participants_df[participants_df["age"].between(*ages)].drop(["age"], axis=1)[
            "sub"
        ]
    )

    subs = subs[:max_subj]

    # We noticed that specific subjects were the reason why we couldn't
    # learn anything from the data: TODO we might remove those now that we updated dataset
    if dattype == "passive":
        # forbidden_subs = ["CC620526", "CC220335", "CC320478", "CC410113", "CC620785"]
        forbidden_subs = []
        if len(forbidden_subs) > 0:
            logging.info(
                f"removed subjects {forbidden_subs}, they were causing problems..."
            )
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
                data_path=data_folder,
                chan_index=chan_index,
                domain=domain,
                printmem=printmem,
                dattype=dattype,
                n_samples=n_samples,
                seed=seed,
                epoched=epoched,
                eventclf=eventclf,
                band=band,
                s_freq=s_freq,
                seg=seg,
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
    seg=2,
    s_freq=200,
    n_samples=None,
    chan_index=[0, 1, 2],
    domain="temporal",
    printmem=False,
    dattype="rest",
    epoched=False,
    seed=0,
    band="",
    testing=None,
):
    """Loading data subject per subject."""
    assert_params(band, domain, dattype)

    csv_file = os.path.join(data_path, f"participants_info_{dattype}.csv")
    dataframe = pd.read_csv(csv_file, index_col=0)
    # For some reason this subject makes un unable to learn #TODO might remove those since we changed dataset
    # forbidden_subs = ["CC220901"]
    forbidden_subs = []
    if len(forbidden_subs) > 0:
        logging.info(
            f"removed subjects {forbidden_subs}, they were causing problems..."
        )

    for sub in forbidden_subs:
        dataframe = dataframe.loc[dataframe["sub"] != sub]

    dataframe = dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)[
        :max_subj
    ]

    n_sub = len(dataframe)
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        totmem = psutil.virtual_memory().total / 10 ** 9
        logging.info(f"Total Available memory: {totmem:.3f} Go")

    final_n_splits = n_splits - 1 if testing is not None else n_splits
    X_sets = [[] for _ in range(final_n_splits)]
    y_sets = [[] for _ in range(final_n_splits)]
    logging.debug(list(dataframe["sub"]))
    for i, row in enumerate(dataframe.iterrows()):
        random.seed(seed)
        torch.manual_seed(seed)
        if printmem:
            usedmem = psutil.virtual_memory().used / 10 ** 9
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
            dattype=dattype,
            epoched=epoched,
            domain=domain,
            offset=offset,
            s_freq=s_freq,
            seg=seg,
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
    seg=2,
    s_freq=200,
    chan_index=[0, 1, 2],
    domain="temporal",
    printmem=False,
    dattype="rest",
    n_samples=None,
    seed=0,
    epoched=False,
    eventclf=False,
    band="",
):
    """Loading data subject per subject and returns labels according to the task.
    Currently if epoched is set to True, we will load the event labels from epoched data.
    id eventclf is set to True, we will used labels from the events, if not,
    sex is used as the default label but can be easily changed to hand or age for example.
    """
    assert_params(band, domain, dattype)
    if eventclf:
        assert (
            dattype != "rest"
        ), "We can not perform event classification on resting state data as it contains no events and therefore can not be epoched."

    n_sub = len(dataframe)
    X, y = [], []
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        # subj_sizes = [] # assigned but never used ?
        totmem = psutil.virtual_memory().total / 10 ** 9
        logging.info(f"Total Available memory: {totmem:.3f} Go")
    for i, row in enumerate(dataframe.iterrows()):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if printmem:
            usedmem = psutil.virtual_memory().used / 10 ** 9
            memstate = f"Used memory: {usedmem:.3f} / {totmem:.3f} Go."
            if n_sub > 10:
                if i % (n_sub // 10) == 0:
                    logging.debug(memstate)
            else:
                logging.debug(memstate)

        # TODO Add option to change the label from sex to some other column of the dataframe
        sub, lab = row[1]["sub"], row[1]["sex"]
        data = load_sub(
            data_path,
            sub,
            n_samples=n_samples,
            band=band,
            chan_index=chan_index,
            dattype=dattype,
            epoched=epoched,
            domain=domain,
            offset=offset,
            s_freq=s_freq,
            seg=seg,
        )
        if data is None:
            n_sub -= 1
            continue

        if n_samples is not None:
            random_samples = np.random.choice(
                np.arange(len(data)), n_samples, replace=False
            )
            data = torch.Tensor(data)[random_samples]

        if eventclf:
            events = np.array(
                literal_eval(
                    dataframe.loc[dataframe["sub"] == sub]["event_labels"].item()
                )
            )
            if len(events) != len(data):
                n_sub -= 1
                continue
            y += [0 if e == "visual" else 1 for e in events]
        else:
            y += [1 if lab == "FEMALE" else 0] * len(data)

        X.append(torch.as_tensor(np.array(data)))
    logging.info(f"Loaded {n_sub} subjects succesfully\n")

    X = torch.cat(X, 0)
    y = torch.as_tensor(y)

    return X, y


def load_epoched_sub(data_path, sub, chan_index, dattype="passive", s_freq=500):
    assert dattype in ("passive", "smt"), "cannot load epoched data for resting state"
    sub_path = os.path.join(
        data_path, f"downsampled_{s_freq}", f"{dattype}_{sub}_epoched.npy"
    )
    try:
        sub_data = np.array(
            list(map(lambda x: zscore(x, axis=-1), np.load(sub_path)[:, chan_index]))
        )
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
    dattype="rest",
    epoched=False,
    domain="temporal",
    offset=30,
    s_freq=200,
    seg=2,
):
    # TODO doc
    """seg is the size of the segments in seconds"""
    if epoched:
        data = load_epoched_sub(
            data_path, sub, chan_index, dattype=dattype, s_freq=s_freq
        )
    else:
        try:
            if domain in ("cov", "cosp"):  # TODO Deprecated
                file_path = os.path.join(
                    data_path, "covariances", f"{sub}_{dattype}_{domain}.npy"
                )
                data = np.load(file_path)[chan_index]
                data = np.swapaxes(data, 0, 1)
                if domain == "cosp":
                    data = data[:, :, BANDS.index(band)]
            else:
                file_path = os.path.join(
                    data_path, f"downsampled_{s_freq}", f"{dattype}_{sub}.npy"
                )
                sub_data = np.load(file_path)[chan_index]
                step = int(seg * s_freq)
                start = int(offset * s_freq)
                if domain == "both":
                    # TODO the welch code might be wrong, check if the trans_freqormation is actually done correctly. It is supposed to give data = n x n_channels x time + bins so probably 3 x 102 x 200 + n_bins
                    # Deprecated
                    data = [
                        np.append(
                            zscore(sub_data[:, :, i : i + step], axis=-1),
                            welch(sub_data[:, :, i : i + step], fs=s_freq)[1],
                        )
                        for i in range(start, sub_data.shape[-1], step)
                    ]
                elif domain == "bands":
                    # TODO for now does the same thing as bins, but should be averaged to get the bands value instead of the bins directly.
                    # We can use a function that should be in utils also, start using multitaper instead of welch
                    data = [
                        np.append(welch(sub_data[:, :, i : i + step], fs=s_freq)[1])
                        for i in range(start, sub_data.shape[-1], step)
                    ]
                elif domain == "bins":
                    data = [
                        np.append(welch(sub_data[:, :, i : i + step], fs=s_freq)[1])
                        for i in range(start, sub_data.shape[-1], step)
                    ]
                elif domain == "temporal":
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
                        data = torch.Tensor(data)[random_samples]

        except IOError:
            logging.warning(f"There was a problem loading subject file for {sub}")
            return None
    return data
