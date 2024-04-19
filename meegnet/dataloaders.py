import os
import random
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from scipy.stats import zscore
from meegnet.utils import strip_string

LOG = logging.getLogger("meegnet")


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


def string_to_int(array):
    array = np.array(array)
    for i, element in enumerate(np.unique(array)):
        array[array == element] = i
    return array.astype(int)


class Dataset:
    def __init__(
        self,
        sfreq=500,
        n_subjects=None,
        zscore=True,
        n_samples=None,
        sensortype=None,
        lso=False,
        random_state=0,
    ):
        self.sfreq = sfreq
        self.n_subjects = n_subjects
        self.zscore = zscore
        self.n_samples = n_samples
        self.lso = lso
        self.random_state = random_state
        self.sensors = self._select_sensors(sensortype)

        self.data = []
        self.labels = []
        self.groups = []

    def _load_csv(self, data_path, csv_name="participants_info.csv"):
        csv_file = os.path.join(data_path, csv_name)
        participants_df = (
            pd.read_csv(csv_file, index_col=0)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)[: self.n_subjects]
        )
        return participants_df

    def load(self, data_path, csv_path=None):
        if csv_path is not None and os.path.exists(csv_path):
            dataframe = pd.read_csv(csv_path, index_col=0)
        else:
            dataframe = self._load_csv(data_path)
        LOG.info(f"Logging subjects and labels from {data_path}...")
        subject_list = list(dataframe["sub"])
        LOG.info(f"Found {len(subject_list)} subjects to load.")
        data_folder = f"downsampled_{self.sfreq}"
        numpy_filepath = os.path.join(data_path, data_folder)
        for file in os.listdir(numpy_filepath):
            np.random.seed(self.random_state)
            random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            sub = file.split("_")[0]
            if sub in subject_list:
                row = dataframe.loc[dataframe["sub"] == sub]
                sub_data = self._load_sub(os.path.join(numpy_filepath, file))
                if self.sensors is not None:
                    sub_data = sub_data[:, self.sensors, :, :]
                if sub_data is None:
                    continue
                labels = list(map(strip_string, row["label"].item().split(", ")))
                if len(labels) == 1:
                    labels = [labels[0]] * len(sub_data)
                if len(labels) != len(sub_data):
                    LOG.warning(
                        "Length of label vector different from number "
                        f"of data samples for subject {sub}. Skipping."
                    )
                    continue
                if self.n_samples is not None:
                    if self.n_samples > len(sub_data):
                        LOG.warning(
                            f"Number of available samples for {file} "
                            f"below the requested amount ({self.n_samples})",
                        )
                    random_samples = np.random.choice(
                        np.arange(len(sub_data)),
                        self.n_samples,
                        replace=False,
                    )
                    sub_data = sub_data[random_samples]
                    labels = np.array(labels)[random_samples]

                self.data.append(torch.Tensor(sub_data))
                self.labels.append(np.array(labels))
                self.groups += [sub] * len(labels)

        self.data = torch.cat(self.data, 0)
        if len(self.data.shape) != 4:
            self.data = self.data[:, np.newaxis]

        self.labels = np.concatenate(self.labels, axis=0)
        if type(self.labels[0]) != int:
            self.labels = string_to_int(self.labels)
        self.labels = torch.as_tensor(self.labels)

        if type(self.groups[0]) != int:
            self.groups = string_to_int(self.groups)
        self.groups = torch.as_tensor(self.groups)

        self.n_subjects = len(np.unique(self.groups))

    def __len__(self):
        return len(self.data)

    def _load_sub(self, filepath):
        try:
            data = np.load(filepath)
        except IOError:
            LOG.warning(f"There was a problem loading subject {filepath}")
            return None
        if self.zscore:
            data = np.array(list(map(lambda x: zscore(x, axis=-1), data)))
        return torch.Tensor(np.array(data))

    def _select_sensors(self, sensortype):
        if sensortype == "MAG":
            return [0]
        elif sensortype in ["GRAD", "plannar"]:
            return [1, 2]
        elif sensortype in ["GRAD1", "plannar1"]:
            return [1]
        elif sensortype in ["GRAD2", "plannar2"]:
            return [2]
        else:
            return None

    def _assert_sizes(self, train_size, valid_size, test_size=None):
        if test_size is None:
            test_size = 0
        assert (
            sum((train_size, valid_size, test_size)) == 1
        ), "sum of data ratios must be equal to 1"

    def _within_subject_split(self, sizes, generator):
        indexes = []
        index_groups = [[] for _ in range(self.n_subjects)]
        for index, group in enumerate(self.groups):
            index_groups[group].append(index)
        for group in index_groups:
            indexes.append(random_split(group, sizes, generator))
        # logging.info(
        #     f"Using {self.n_subjects} subjects: {train_size} for train, "
        #     f"{valid_size} for validation, and {test_size} for test"
        # )
        return (sum([list(index[i]) for index in indexes], []) for i in range(3))

    def data_split(self, train_size, valid_size, test_size=None):
        # TODO add stratification for the data splits
        self._assert_sizes(train_size, valid_size, test_size)
        generator = torch.Generator().manual_seed(self.random_state)
        if self.lso:
            return self._within_subject_split((train_size, valid_size, test_size), generator)
        else:
            return random_split(
                np.arange(len(self)), [train_size, valid_size, test_size], generator
            )

    def torchDataset(self, index):
        return torch.utils.data.TensorDataset(self.data[index], self.labels[index])


class RestDataset(Dataset):
    def __init__(
        self,
        window=2,
        overlap=0,
        offset=10,
        sfreq=500,
        n_subjects=None,
        zscore=True,
        n_samples=None,
        sensortype=None,
        lso=False,
        random_state=0,
    ):
        Dataset.__init__(
            self, sfreq, n_subjects, zscore, n_samples, sensortype, lso, random_state
        )

        assert 0 <= overlap < 1, "Overlap must be between 0 and 1."
        self.window = window
        self.overlap = overlap
        self.offset = offset

    def _load_sub(self, filepath):
        try:
            step = int(self.window * self.sfreq * (1 - self.overlap))
            start = int(self.offset * self.sfreq)
            sub_data = np.load(filepath)
            if len(sub_data.shape) < 3:
                sub_data = sub_data[np.newaxis, :]
            data = []
            for i in range(start, sub_data.shape[-1], step):
                trial = sub_data[:, :, i : i + step]
                if trial.shape[-1] == step:
                    if not np.isnan(trial).any():
                        if self.zscore:
                            trial = zscore(trial, axis=-1)
                        data.append(trial)
        except IOError:
            LOG.warning(f"There was a problem loading subject {filepath}")
            return None
        except ValueError:
            LOG.warning(f"There was a problem loading subject {filepath}")
            return None
        return torch.Tensor(np.array(data))
