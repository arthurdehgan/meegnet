from __future__ import print_function, division
from time import time
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from params import NBINS


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


def create_dataset(data_df, data_path, ch_type, dtype="temporal", debug=False):
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]

    if debug:
        data_df = data_df[:200]

    meg_dataset = chunkedMegDataset(
        data_df=data_df,
        root_dir=data_path,
        chan_index=chan_index,
        dtype=dtype,
    )
    # else:
    #     sexlist = []
    #     data = None
    #     print("Loading data...")
    #     for row in data_df.iterrows():
    #         sub, sex, begin, end = row[1]
    #         f = f"{sub}_{sex}_{begin}_{end}_ICA_ds200.npy"
    #         file_path = os.path.join(data_path, f)
    #         trial = zscore(np.load(file_path)[chan_index], axis=1)
    #         data = (
    #             trial[np.newaxis, ...]
    #             if data is None
    #             else np.concatenate((trial[np.newaxis, ...], data))
    #         )
    #         sex = int(f.split("_")[1])
    #         sexlist.append(sex)

    #     if np.isnan(np.sum(trial)):
    #         print(file_path, "becomes nan")
    #     print("Data sucessfully loaded")

    #     meg_dataset = TensorDataset(torch.Tensor(data), torch.Tensor(sexlist))

    return meg_dataset


def create_loaders(
    data_folder,
    train_size,
    batch_size,
    max_subj,
    ch_type,
    dtype,
    debug=False,
    seed=0,
    num_workers=0,
    chunkload=True,
):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    # Using trials_df ensures we use the correct subjects that do not give errors since
    # it is created by reading the data. It is therefore better than SUB_DF previously used
    # We now use trials_df_clean that contains one less subjects that contained nans
    samples_df = pd.read_csv(f"{data_folder}trials_df_clean.csv", index_col=0)
    subs = np.array(list(set(samples_df["subs"])))
    idx = rng.permutation(range(len(subs)))
    subs = subs[idx]
    subs = subs[:max_subj]
    N = len(subs)
    train_size = int(N * train_size)
    remaining_size = N - train_size
    valid_size = int(remaining_size / 2)
    test_size = remaining_size - valid_size
    train_index, test_index, valid_index = random_split(
        np.arange(N), [train_size, test_size, valid_size]
    )

    bands = False
    load_fn = load_freq_data
    if dtype == "temporal":
        load_fn = load_data
    elif dtype == "bands":
        bands = True
    elif dtype == "both":
        load_fn = load_data

    train_df = (
        samples_df.loc[samples_df["subs"].isin(subs[train_index])]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    valid_df = (
        samples_df.loc[samples_df["subs"].isin(subs[valid_index])]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    test_df = (
        samples_df.loc[samples_df["subs"].isin(subs[test_index])]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    if chunkload:
        train_set = create_dataset(
            train_df, data_folder, ch_type, dtype=dtype, debug=debug
        )
        valid_set = create_dataset(
            valid_df, data_folder, ch_type, dtype=dtype, debug=debug
        )
        test_set = create_dataset(
            test_df, data_folder, ch_type, dtype=dtype, debug=debug
        )

    else:
        X_test, y_test = load_fn(
            test_df,
            dpath=data_folder,
            ch_type=ch_type,
            bands=bands,
            dtype=dtype,
            debug=debug,
        )
        X_valid, y_valid = load_fn(
            valid_df,
            dpath=data_folder,
            ch_type=ch_type,
            bands=bands,
            dtype=dtype,
            debug=debug,
        )
        X_train, y_train = load_fn(
            train_df,
            dpath=data_folder,
            ch_type=ch_type,
            bands=bands,
            dtype=dtype,
            debug=debug,
        )
        train_set = TensorDataset(X_train, y_train)
        valid_set = TensorDataset(X_valid, y_valid)
        test_set = TensorDataset(X_test, y_test)

    # loading data with num_workers=0 is faster that using more because of IO read speeds on my machine.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


class chunkedMegDataset(Dataset):
    """MEG dataset, from examples of the pytorch website: FaceLandmarks"""

    def __init__(self, data_df, root_dir, chan_index, dtype="temporal"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the cut samples of MEG trials.
            chan_index (list): The index of electrodes to keep.
        """
        self.data_df = data_df
        self.root_dir = root_dir
        self.chan_index = chan_index
        self.dtype = dtype

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sub = self.data_df["subs"].iloc[idx]
        sex = self.data_df["sex"].iloc[idx]
        begin = self.data_df["begin"].iloc[idx]
        end = self.data_df["end"].iloc[idx]

        if self.dtype == "temporal":
            data_path = os.path.join(
                self.root_dir, f"{sub}_{sex}_{begin}_{end}_ICA_ds200.npy"
            )
            trial = np.load(data_path)[self.chan_index]
            trial = zscore(trial, axis=1)
            if np.isnan(np.sum(trial)):
                print(data_path, "becomes nan")
        elif self.dtype.startswith("freq"):
            data_path = os.path.join(self.root_dir, f"{sub}_psd.npy")
            trial = np.load(data_path)[:, self.chan_index]
            if self.dtype == "freqbands":
                trial = extract_bands(trial)
        elif self.dtype == "both":
            data_path = os.path.join(
                self.root_dir, f"{sub}_{sex}_{begin}_{end}_ICA_ds200.npy"
            )
            signal = np.load(data_path)[self.chan_index]
            time = zscore(signal, axis=1)
            freq = np.zeros(list(signal.shape[:-1]) + [NBINS])
            for i, mat in enumerate(signal):
                for j, seg in enumerate(mat):
                    freq[i, j] = welch(seg, fs=200)[1]
            trial = np.concatenate((time, freq), axis=-1)

        sample = (trial, sex)

        return sample


def load_freq_data(dataframe, dpath, ch_type="MAG", bands=True, debug=False):
    """Loading psd values, subject by subject. Still viable, takes some time
    but data is small, so not too much. Might need repairing as code has changed
    a lot since last time this function was used.
    """
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]

    if debug:
        # Not currently working
        print("ENTERING DEBUG MODE")
        nb = 5 if bands else 241
        dummy = np.zeros((25000, len(elec_index), nb))
        return torch.Tensor(dummy).float(), torch.Tensor(dummy).float()

    X = None
    y = []
    i = 0
    for row in dataframe.iterrows():
        print(f"loading subject {i+1}...")
        sub, lab = row[1]["participant_id"], row[1]["sex"]
        try:
            sub_data = np.array(np.load(dpath + f"{sub}_psd.npy"))[:, elec_index]
        except:
            print("There was a problem loading subject ", sub)

        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        y += [lab] * len(sub_data)
        i += 1
    if bands:
        X = extract_bands(X)
    return torch.Tensor(X).float(), torch.Tensor(y).long()


def load_data(
    dataframe,
    dpath,
    offset=2000,
    ch_type="MAG",
    bands=True,
    dtype="temporal",
    debug=False,
):
    """Loading data subject per subject.

    bands is here only for compatibility with load_freq_data"""
    if ch_type == "MAG":
        chan_index = [2]
    elif ch_type == "GRAD":
        chan_index = [0, 1]
    elif ch_type == "ALL":
        chan_index = [0, 1, 2]

    if debug:
        dataframe = dataframe[:20]

    subs_df = (
        dataframe.drop(["begin", "end"], axis=1)
        .drop_duplicates(subset=["subs"])
        .reset_index(drop=True)
    )

    X = None
    y = []
    # i = 0
    print("Loading the whole dataset...")
    for row in dataframe.iterrows():
        sub, lab = row[1]["subs"], row[1]["sex"]
        try:
            sub_data = np.load(dpath + f"{sub}_ICA_transdef_mfds200.npy")[chan_index]
        except:
            print("There was a problem loading subject ", sub)
            continue

        sub_segments = dataframe.loc[dataframe["subs"] == sub].drop(["sex"], axis=1)
        if dtype == "both":
            sub_data = [
                np.append(
                    zscore(sub_data[:, :, begin:end], axis=1),
                    welch(sub_data, fs=200)[1],
                )
                for begin, end in zip(sub_segments["begin"], sub_segments["end"])
            ]
        elif dtype == "temporal":
            sub_data = [
                zscore(sub_data[:, :, begin:end], axis=1)
                for begin, end in zip(sub_segments["begin"], sub_segments["end"])
            ]

        sub_data = np.array(sub_data)
        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        y += [lab] * len(sub_data)
        # y += [i] * len(sub_data)
        # i += 1
    print("Loading successfull")
    return torch.Tensor(X).float(), torch.Tensor(y).long()


def load_subject(sub, data_path, data=None, timepoints=500, ch_type="all"):
    """Loads a single subject from info found in the csv file. Deprecated, takes too much time.
    path needs to be updated as CHAN_DF is no longer defined and paths have changed.
    """
    df = pd.read_csv("{}/cleansub_data_camcan_participant_data.csv".format(data_path))
    df = df.set_index("participant_id")
    sex = (df["sex"])[sub]
    subject_file = "{}_rest.mat".format(data_path + sub)
    # trial = read_raw_fif(subject_file,
    #                      preload=True).pick_types(meg=True)[:][0]
    trial = np.load(subject_file)
    if ch_type == "all":
        mask = [True for _ in range(len(trial))]
        n_channels = 306
    elif ch_type == "mag":
        mask = CHAN_DF["mag_mask"]
        n_channels = 102
    elif ch_type == "grad":
        mask = CHAN_DF["grad_mask"]
        n_channels = 204
    else:
        raise ("Error : bad channel type selected")
    trial = trial[mask]

    n_trials = trial.shape[-1] // timepoints
    for i in range(1, n_trials - 1):
        curr = trial[:, i * timepoints : (i + 1) * timepoints]
        curr = curr.reshape(1, n_channels, timepoints)
        data = curr if data is None else np.concatenate((data, curr))
    labels = [sex] * (n_trials - 2)
    data = data.astype(np.float32, copy=False)
    return data, labels
