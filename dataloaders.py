from __future__ import print_function, division
import os
import logging
import torch
import psutil
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
from torch.utils.data import DataLoader, random_split, TensorDataset

# From Domainbed, modified
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
    samples=None,
    permute_labels=False,
    load_groups=False,
    testing=False,
):
    """create dataloaders iterators."""
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

    if not testing:
        dataframes = dataframes[:-1]

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
                samples=samples,
                seed=seed,
                permute_labels=permute_labels,
                load_groups=load_groups,
            )
        )
        for i, df in enumerate(dataframes)
    ]
    return datasets


def create_loader(
    dataset,
    batch_size,
    infinite=False,
    num_workers=0,
):
    if infinite:
        loader = InfiniteDataLoader
    else:
        loader = DataLoader

    return loader(dataset, batch_size=batch_size, num_workers=num_workers)


def load_data(
    dataframe,
    dpath,
    offset=30,
    ch_type="MAG",
    domain="temporal",
    printmem=False,
    dattype="rest",
    samples=None,
    seed=0,
    permute_labels=False,
    load_groups=False,
):
    """Loading data subject per subject."""
    SAMPLING_FREQ = 200
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]

    if dattype not in ["rest", "passive", "task"]:
        logging.error(
            f"Incorrect data type: {dattype}. Must be in (rest, passive, task)"
        )
    else:
        logging.info(f"Loading data from the {dattype} data set")

    subs_df = (
        dataframe.drop(["begin", "end"], axis=1)
        .drop_duplicates(subset=["subs"])
        .reset_index(drop=True)
    )

    n_subj = len(subs_df)
    X = []
    y = []
    if load_groups:
        groups = []
    logging.debug(f"Loading {n_subj} subjects data")
    if printmem:
        # subj_sizes = [] # assigned but never used ?
        totmem = psutil.virtual_memory().total / 10 ** 9
        logging.info(f"Total Available memory: {totmem:.3f} Go")
    for i, row in enumerate(subs_df.iterrows()):
        np.random.seed(seed)
        if printmem:
            usedmem = psutil.virtual_memory().used / 10 ** 9
            memstate = f"Used memory: {usedmem:.3f} / {totmem:.3f} Go."
            if n_subj > 10:
                if i % (n_subj // 10) == 0:
                    logging.debug(memstate)
            else:
                logging.debug(memstate)

        sub, lab = row[1]["subs"], row[1]["sex"]
        try:
            # TODO MUST COMPUTE COV AND COSP MATRICES, and this will work (Will have to use sub_segments to compute cov and cosp)
            if domain == "cov":
                sub_data = np.load(
                    dpath + f"{sub}_{dattype}_ICA_transdef_mfds200_cov.npy"
                )[chan_index]
            elif domain == "cosp":
                sub_data = np.load(
                    dpath + f"{sub}_{dattype}_ICA_transdef_mfds200_cosp.npy"
                )[chan_index]
            else:
                sub_data = np.load(dpath + f"{sub}_{dattype}_ICA_transdef_mfds200.npy")[
                    chan_index
                ]
        except:
            logging.warning(f"Warning: There was a problem loading subject {sub}")
            n_subj -= 1
            continue

        # TODO there must be a better way to code this.
        sub_segments = dataframe.loc[dataframe["subs"] == sub].drop(["sex"], axis=1)
        if domain == "both":
            # TODO the welch code might be wrong, check if the transformation is actually done correctly. It is supposed to give data = n x n_channels x time + bins so probably 3 x 102 x 200 + n_bins
            try:
                data = [
                    np.append(
                        zscore(sub_data[:, :, begin:end], axis=1),
                        welch(sub_data[:, :, begin:end], fs=SAMPLING_FREQ)[1],
                    )
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            except:
                logging.warning(f"Warning: There was a problem loading subject {sub}")
                n_subj -= 1
                continue

        elif domain == "bands":
            # TODO for now does the same thing as bins, but should be averaged to get the bands value instead of the bins directly.
            try:
                data = [
                    np.append(welch(sub_data[:, :, begin:end], fs=SAMPLING_FREQ)[1])
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            except:
                logging.warning(f"Warning: There was a problem loading subject {sub}")
                n_subj -= 1

        elif domain == "bins":
            try:
                data = [
                    np.append(welch(sub_data, fs=SAMPLING_FREQ)[1])
                    for begin, end in zip(sub_segments["begin"], sub_segments["end"])
                    if begin >= offset * SAMPLING_FREQ
                ]
            except:
                logging.warning(f"Warning: There was a problem loading subject {sub}")
                n_subj -= 1

        elif domain == "temporal":
            data = []
            for begin, end in zip(sub_segments["begin"], sub_segments["end"]):
                seg = sub_data[:, :, begin:end]
                if (
                    seg.shape[-1] == end - begin
                    and begin >= offset * SAMPLING_FREQ
                    and not np.isnan(seg).any()
                ):
                    try:
                        data.append(zscore(seg, axis=1))
                    except:
                        continue
            if len(data) < 50:
                logging.warning(f"Warning: There was a problem loading subject {sub}")
                n_subj -= 1
                continue

        # TODO Here add something to only load the cov and cosp matrices from offset to end
        if samples is not None:
            random_samples = np.random.choice(
                np.arange(len(data)), samples, replace=False
            )
            data = torch.Tensor(data)[random_samples]
        else:
            data = torch.Tensor(data)
        X.append(data)
        y += [lab] * len(data)
        if load_groups:
            groups += [i] * len(data)
    logging.info(f"Loaded {n_subj} subjects succesfully\n")

    y = torch.Tensor(y)
    if permute_labels:
        y = y[np.random.permutation(list(range(len(y))))]
        logging.info("Labels shuffled for permutation test!")

    if load_groups:
        return torch.cat(X, 0), y, groups
    else:
        return torch.cat(X, 0), y
