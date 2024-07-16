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


def _string_to_int(array):
    array = np.array(array)
    for i, element in enumerate(np.unique(array)):
        array[array == element] = i
    return array.astype(int)


class Dataset:
    """Creates a dataset

    Parameters
    ----------
    sfreq : int, optional
        The sampling frequency, by default 500.
    n_subjects : int, optional
        The number of subjects. Default value is None, which means all subjects are processed.
    zscore : bool, optional
        If True, z-scoring is applied to the data, by default True.
    n_samples : int, optional
        The number of samples to include, by default None.
    sensortype : str, optional
        The type of sensor to use, by default None.
    lso : bool, optional
        Leave subjects out. If False, within-subject splitting is used, by default False.
    random_state : int, optional
        The random state for reproducibility, by default 0.

    Attributes
    ----------
    sfreq : int
        The sampling frequency.
    n_subjects : int
        The number of subjects.
    n_samples : int
        The number of samples for each subject.
    data : torch.Tensor
        The data.
    labels : torch.Tensor
        The labels.
    groups : list
        The groups.
    subject_list : list
        The subject list.
    """

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
        self.sensors = self._select_sensors(sensortype)
        self.random_state = random_state
        self._reset_seed()

        self.data = []
        self.labels = []
        self.groups = []
        self.subject_list = []
        self.targets = self.labels
        self.data_path = None

    def _reset_seed(self):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _load_csv(self, data_path, csv_name="participants_info.csv"):
        csv_file = os.path.join(data_path, csv_name)
        participants_df = (
            pd.read_csv(csv_file, index_col=0)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)[: self.n_subjects]
        )
        return participants_df

    def preload(self, data_path, csv_path=None):
        """loads the subject list from a csv file (participants_info.csv in the data_folder by default).

        Parameters
        ----------
        data_path :
            _description_
        csv_path : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        if csv_path is not None and os.path.exists(csv_path):
            dataframe = pd.read_csv(csv_path, index_col=0)
        else:
            dataframe = self._load_csv(data_path)
        LOG.info(f"Logging subjects and labels from {data_path}...")
        self.data_path = data_path
        self.subject_list = list(dataframe["sub"])
        return dataframe

    def load(self, data_path=None, csv_path=None, one_sub=None):
        """Loads the data from the "downsamples_[sfreq]" folder in the data_path.

        Parameters
        ----------
        data_path : _type_, optional
            _description_, by default None
        csv_path : _type_, optional
            _description_, by default None
        one_sub : _type_, optional
            _description_, by default None
        """

        assert self.data_path is not None or data_path is not None, "data_path must be set."
        if self.data_path is None:
            self.data_path = data_path
        dataframe = self.preload(self.data_path, csv_path)

        if one_sub is None:
            LOG.info(f"Found {len(self.subject_list)} subjects to load.")
        else:
            if one_sub == "random":
                one_sub = self.random_sub()
            elif one_sub in self.subject_list:
                LOG.info(f"Loading subject {one_sub}")
                self.subject_list = [one_sub]
            else:
                raise AttributeError(f"{one_sub} not a valid subject.")

        data_folder = f"downsampled_{self.sfreq}"
        numpy_filepath = os.path.join(self.data_path, data_folder)
        for file in os.listdir(numpy_filepath):
            self._reset_seed()
            sub = file.split("_")[1]  # The subject ID is placed second !
            if one_sub is not None:
                if sub != one_sub:
                    continue
            if sub in self.subject_list:
                row = dataframe.loc[dataframe["sub"] == sub]
                sub_data = self._load_sub(os.path.join(numpy_filepath, file))
                if sub_data is None:
                    continue
                if self.sensors is not None:
                    sub_data = sub_data[:, self.sensors, :, :]
                labels = list(map(strip_string, row["label"].item().split(", ")))
                if len(labels) == 1:
                    if labels[0] in self.subject_list:
                        labels = [self.subject_list.index(labels[0])] * len(sub_data)
                    else:
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

        if len(self) == 0:
            return
        elif len(self) == 1:
            self.data = self.data[0]
            self.labels = self.labels[0]
        else:
            self.data = torch.cat(self.data, 0)
            self.labels = np.concatenate(self.labels, axis=0)

        if len(self.data.shape) != 4:
            self.data = self.data[:, np.newaxis]

        self.set_labels(self.labels)

        if type(self.groups[0]) != int:
            self.groups = _string_to_int(self.groups)
        self.groups = torch.Tensor(self.groups)

        self.n_subjects = len(np.unique(self.groups))

    def random_sub(self):
        return np.random.choice(self.subject_list)

    def __len__(self):
        """Returns the length of the dataset (total number of data examples).

        Returns
        -------
        _type_
            _description_
        """
        return len(self.data)

    def _load_sub(self, filepath):
        """Loads a subject's data.

        Parameters
        ----------
        filepath : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        try:
            data = np.load(filepath)
        except IOError:
            LOG.warning(f"There was a problem loading subject {filepath}")
            return None
        if self.zscore:
            data = np.array(list(map(lambda x: zscore(x, axis=-1), data)))
        return torch.Tensor(np.array(data))

    def _select_sensors(self, sensortype):
        """For MEG data only. Selects the sensors slices from the data.

        Parameters
        ----------
        sensortype : str
            type of sensor. Must be either "MAG", "plannar1", "plannar2" or "plannar".

        Returns
        -------
        _type_
            _description_
        """
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
        """Asserts that the sum of the data ratios is equal to 1.

        Parameters
        ----------
        train_size : _type_
            _description_
        valid_size : _type_
            _description_
        test_size : _type_, optional
            _description_, by default None
        """
        if test_size is None:
            test_size = 0
        assert (
            sum((train_size, valid_size, test_size)) == 1
        ), "sum of data ratios must be equal to 1"

    def _within_subject_split(self, sizes, generator):
        """Splits the data within each subject using the specified sizes and generator.

        Parameters
        ----------
        sizes : _type_
            _description_
        generator : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
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
        """Splits the data into training, validation, and test sets based on the specified sizes and stratification (if desired).

        Parameters
        ----------
        train_size : _type_
            _description_
        valid_size : _type_
            _description_
        test_size : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
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
        """Returns a Torch dataset instance of the torch Dataset class for the given index.

        Parameters
        ----------
        index : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return torch.utils.data.TensorDataset(self.data[index], self.labels[index])

    def set_labels(self, labels):
        if type(labels[0]) in (str, np.str_):
            labels = _string_to_int(labels)
        self.labels = torch.Tensor(labels)


class RestDataset(Dataset):
    """
    Creates a dataset for deep learning models from REST data with windowing.

    Parameters
    ----------
    window : int, optional
        The window size in seconds, by default 2.
    overlap : float, optional
        The overlap between windows, by default 0.
    offset : int, optional
        The offset in seconds, by default 10.
    sfreq : int, optional
        The sampling frequency, by default 500.
    n_subjects : int, optional
        The number of subjects. Default value is None, which means all subjects are processed.
    zscore : bool, optional
        If True, z-scoring is applied to the data, by default True.
    n_samples : int, optional
        The number of samples to include, by default None.
    sensortype : str, optional
        The type of sensor to use, by default None.
    lso : bool, optional
        Leave subjects out. If False, within-subject splitting is used, by default False.
    random_state : int, optional
        The random state for reproducibility, by default 0.

    Attributes
    ----------
    window : int
        The window size in seconds.
    overlap : float
        The overlap between windows.
    offset : int
        The offset in seconds.
    sfreq : int
        The sampling frequency.
    n_subjects : int
        The number of subjects.
    n_samples : int
        The number of samples for each subject.
    data : torch.Tensor
        The data.
    labels : torch.Tensor
        The labels.
    groups : list
        The groups.
    subject_list : list
        The subject list.

    Methods
    -------
    _load_sub(filepath)
        Loads a subject's data with windowing and overlap.
    """

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
