from __future__ import print_function, division
import os
import random
import logging
import torch
import psutil
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split, TensorDataset

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

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


def extract_bands(data):
    if len(data.shape) < 3:
        data = data[np.newaxis, :, :]
        add_axis = True
    f = np.asarray([float(i / 2) for i in range(data.shape[-1])])
    # data = data[:, :, (f >= 8) * (f <= 12)].mean(axis=2)
    data = [
        data[:, :, (f >= 0.5) * (f <= 4)].mean(axis=-1)[..., None],
        data[:, :, (f >= 4) * (f <= 8)].mean(axis=-1)[..., None],
        data[:, :, (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[:, :, (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[:, :, (f >= 30) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=2)
    if add_axis:
        return data[0]
    return data


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
    else:
        logging.info(f"Loading data from the {dattype} data set")
    return


def create_datasets(
    data_folder,
    train_size,
    max_subj,
    ch_type,
    domain,
    debug=False,
    seed=0,
    printmem=False,
    ages=(0, 100),
    dattype="rest",
    band="",
    n_samples=None,
    load_groups=False,
    load_events=False,
    testing=None,
):
    """create dataloaders iterators.

    testing: if set to an integer between 0 and 4 will leave out a part of the dataset.
             Useful for random search.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    # Using trials_df ensures we use the correct subjects that do not give errors since
    # it is created by reading the data. It is therefore better than SUB_DF previously used
    # We now use trials_df_clean that contains one less subjects that contained nans
    samples_df = pd.read_csv(f"{data_folder}trials_df_clean.csv", index_col=0)
    ages_df = (
        pd.read_csv(f"{data_folder}clean_participant_new.csv", index_col=0)
        .rename(columns={"participant_id": "subs"})
        .drop(["hand", "sex_text", "sex"], axis=1)
    )
    subs = (
        samples_df.drop(["begin", "end", "sex"], axis=1)
        .drop_duplicates(subset=["subs"])
        .reset_index(drop=True)
    )

    subs = subs.merge(ages_df[ages_df["subs"].isin(subs["subs"])].dropna(), "left")
    subs = np.array(subs[subs["age"].between(*ages)].drop(["age"], axis=1).subs)
    idx = rng.permutation(range(len(subs)))
    subs = subs[idx]
    subs = subs[:max_subj]
    # We noticed that specific subjects were the reason why we couldn't
    # learn anything from the data:
    if dattype == "passive":
        forbidden_subs = ["CC620526", "CC220335", "CC320478", "CC410113", "CC620785"]
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
        samples_df.loc[samples_df["subs"].isin(subs[index])]
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
                dpath=data_folder,
                ch_type=ch_type,
                domain=domain,
                printmem=printmem,
                dattype=dattype,
                n_samples=n_samples,
                seed=seed,
                load_groups=load_groups,
                load_events=load_events,
                band=band,
            )
        )
        for i, df in enumerate(dataframes)
    ]
    return datasets


def load_sets(
    dpath,
    max_subj=1000,
    n_splits=5,
    offset=30,
    n_samples=None,
    ch_type="ALL",
    domain="temporal",
    printmem=False,
    dattype="rest",
    seed=0,
    band="",
    testing=None,
):
    """Loading data subject per subject."""
    assert_params(band, domain, dattype)

    dataframe = pd.read_csv(f"{dpath}trials_df_clean.csv", index_col=0)
    subs_df = (
        dataframe.drop(["begin", "end"], axis=1)
        .drop_duplicates(subset=["subs"])
        .reset_index(drop=True)
    )[:max_subj]

    n_sub = len(subs_df)
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        totmem = psutil.virtual_memory().total / 10 ** 9
        logging.info(f"Total Available memory: {totmem:.3f} Go")

    final_n_splits = n_splits - 1 if testing is not None else n_splits
    X_sets = [[] for _ in range(final_n_splits)]
    y_sets = [[] for _ in range(final_n_splits)]
    for i, row in enumerate(subs_df.iterrows()):
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

        sub = row[1]["subs"]
        data = load_sub(
            dpath,
            sub,
            n_samples=n_samples,
            band=band,
            ch_type=ch_type,
            dattype=dattype,
            domain=domain,
            offset=offset,
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
            X_sets[j].append(torch.Tensor(data)[index])
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
    dpath,
    offset=30,
    ch_type="MAG",
    domain="temporal",
    printmem=False,
    dattype="rest",
    n_samples=None,
    seed=0,
    load_groups=False,
    load_events=False,
    band="",
):
    """Loading data subject per subject."""
    assert_params(band, domain, dattype)

    subs_df = (
        dataframe.drop(["begin", "end"], axis=1)
        .drop_duplicates(subset=["subs"])
        .reset_index(drop=True)
    )

    n_sub = len(subs_df)
    X, y, e_targets = [], [], []
    if load_groups:
        groups = []
    logging.debug(f"Loading {n_sub} subjects data")
    if printmem:
        # subj_sizes = [] # assigned but never used ?
        totmem = psutil.virtual_memory().total / 10 ** 9
        logging.info(f"Total Available memory: {totmem:.3f} Go")
    for i, row in enumerate(subs_df.iterrows()):
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

        sub, lab = row[1]["subs"], row[1]["sex"]
        if dattype != "passive" and not load_events:
            data = load_sub(
                dpath,
                sub,
                n_samples=n_samples,
                band=band,
                ch_type=ch_type,
                dattype=dattype,
                domain=domain,
                offset=offset,
            )
        else:
            data, targets = load_passive_sub_events(dpath, sub, ch_type=ch_type)
        if data is None:
            n_sub -= 1
            continue

        if n_samples is not None:
            random_samples = np.random.choice(
                np.arange(len(data)), n_samples, replace=False
            )
            data = torch.Tensor(data)[random_samples]

        X.append(torch.as_tensor(data))
        y += [lab] * len(data)
        if dattype == "passive":
            e_targets += targets
        if load_groups:
            groups += [int(sub[2:])] * len(data)
    logging.info(f"Loaded {n_sub} subjects succesfully\n")

    X = torch.cat(X, 0)
    y = torch.as_tensor(y)
    if dattype == "passive":
        e_targets = torch.as_tensor(e_targets)

    if load_groups:
        groups = torch.as_tensor(groups)
        return X, y, groups
    elif dattype == "passive" and load_events:
        return X, e_targets
    else:
        return X, y


def load_passive_sub_events(dpath, sub, ch_type="ALL"):
    s_freq = 500
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]

    data, targets = [], []
    try:
        sub_data = np.load(dpath + f"{sub}_passive_ICA_transdef_mfds500.npy")[
            chan_index
        ]
        events = loadmat(dpath + f"{sub}_passive_events_timestamps.mat")["times"]
    except:
        logging.warning(f"Warning: There was a problem loading subject {sub}")
        return None, None

    for e_type, e_time in events:
        stim_timing = int(float(e_time) * s_freq)
        # 75 and 325 to get 150ms before stim and 650ms after stim. total is the
        # same size as previous examples for architecture: 400 time samples
        # but the s_freq is different. reminder: we did this in order to get a full stim
        # in the example but stims arrive every 1s
        seg = sub_data[:, :, stim_timing - 75 : stim_timing + 325]
        if seg.shape[-1] == 400 and not np.isnan(seg).any():
            try:
                data.append(zscore(seg, axis=1))
                target = 0 if e_type.strip() == "image" else 1
                targets.append(target)
            except:
                continue

    if len(targets) < 30:
        return None, None

    assert len(data) == len(
        targets
    ), f"{sub} has a number of target and trials missmatch"

    return data, targets


def load_sub(
    dpath,
    sub,
    n_samples=None,
    band="",
    ch_type="ALL",
    dattype="rest",
    domain="temporal",
    offset=30,
):
    SAMPLING_FREQ = 200
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]
    dataframe = pd.read_csv(f"{dpath}trials_df_clean.csv", index_col=0)

    try:
        if domain in ("cov", "cosp"):
            data = np.load(dpath + f"{sub}_{dattype}_{domain}.npy")[chan_index]
            data = np.swapaxes(data, 0, 1)
            if domain == "cosp":
                data = data[:, :, BANDS.index(band)]
        else:
            sub_data = np.load(dpath + f"{sub}_{dattype}_ICA_transdef_mfds200.npy")[
                chan_index
            ]
            sub_segments = dataframe.loc[dataframe["subs"] == sub].drop(["sex"], axis=1)
            if domain == "both":
                # TODO the welch code might be wrong, check if the transformation is actually done correctly. It is supposed to give data = n x n_channels x time + bins so probably 3 x 102 x 200 + n_bins
                data = [
                    np.append(
                        zscore(sub_data[:, :, begin:end], axis=1),
                        welch(sub_data[:, :, begin:end], fs=SAMPLING_FREQ)[1],
                    )
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            elif domain == "bands":
                # TODO for now does the same thing as bins, but should be averaged to get the bands value instead of the bins directly.
                data = [
                    np.append(welch(sub_data[:, :, begin:end], fs=SAMPLING_FREQ)[1])
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            elif domain == "bins":
                data = [
                    np.append(welch(sub_data, fs=SAMPLING_FREQ)[1])
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            elif domain == "temporal":
                data = []
                for begin, end in zip(sub_segments["begin"], sub_segments["end"]):
                    seg = sub_data[:, :, begin:end]
                    if seg.shape[-1] == end - begin:
                        if begin >= offset * SAMPLING_FREQ:
                            if not np.isnan(seg).any():
                                data.append(zscore(seg, axis=1))
                if len(data) < 50:
                    return None
                if n_samples is not None and n_samples <= len(data):
                    random_samples = np.random.choice(
                        np.arange(len(data)), n_samples, replace=False
                    )
                    data = torch.Tensor(data)[random_samples]

    except IOError as e:
        logging.warning(f"Warning: There was a problem loading subject file for {sub}")
        return None

    return data
