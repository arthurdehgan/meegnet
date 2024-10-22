import os
import random
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from scipy.stats import zscore
from sklearn.model_selection import GroupShuffleSplit
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


class EpochedDataset:
    """
    Creates a dataset for epoch-based M/EEG data.

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
    targets : torch.Tensor
        The targets (aliases for labels).
    data_path : str
        The path to the data.
    """

    def __init__(
        self,
        sfreq: float = 500,
        n_subjects: int = None,
        zscore: bool = True,
        n_samples: int = None,
        sensortype: str = None,
        lso: bool = False,
        random_state: int = 0,
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

    def _load_csv(self, csv_file: str) -> pd.DataFrame:
        """Loads a CSV file, handling index column."""
        with open(csv_file) as f:
            first_line = f.readline()
        return pd.read_csv(csv_file, index_col=0 if first_line.startswith(",") else None)

    def preload(self, data_path: str, csv_path: str = None) -> pd.DataFrame:
        """
        Loads the subject list from a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the folder containing the dataset.
        csv_path : str, optional
            Path to the CSV file. Defaults to "participants_info.csv" in the data_path.

        Returns
        -------
        pd.DataFrame
            DataFrame containing participant information.
        """

        # Determine CSV file path
        csv_file = csv_path or os.path.join(data_path, "participants_info.csv")
        assert os.path.exists(csv_file), f"CSV file not found: {csv_file}"

        # Load CSV file
        dataframe = self._load_csv(csv_file)

        # Sample participants (if needed)
        if self.n_subjects < len(dataframe):
            dataframe = dataframe.sample(frac=1, random_state=self.random_state).reset_index(
                drop=True
            )[: self.n_subjects]

        # Update instance attributes
        LOG.info(f"Logging subjects and labels from {data_path}...")
        self.data_path = data_path
        self.subject_list = dataframe["sub"].tolist()

        return dataframe

    def _check_index_and_load_csv(self, csv_file) -> pd.DataFrame:
        with open(csv_file) as f:
            first_line = f.readline()
        if first_line.startswith(","):
            df = pd.read_csv(csv_file, index_col=0)
        else:
            df = pd.read_csv(csv_file)
        return df

    def _load_csv(self, data_path, csv_name="participants_info.csv") -> pd.DataFrame:
        csv_file = os.path.join(data_path, csv_name)
        df = self._check_index_and_load_csv(csv_file)
        participants_df = df.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )[: self.n_subjects]
        return participants_df

    def preload(self, data_path: str, csv_path: str = None) -> pd.DataFrame:
        """
        Loads the subject list from a CSV file.

        Parameters
        ----------
        data_path : str
            Path to the folder containing the dataset.
        csv_path : str, optional
            Path to the CSV file. Defaults to "participants_info.csv" in the data_path.

        Returns
        -------
        pd.DataFrame
            DataFrame containing participant information.
        """

        # Determine CSV file path
        if csv_path is not None and os.path.exists(csv_path):
            dataframe = self._check_index_and_load_csv(csv_path)
        else:
            dataframe = self._load_csv(data_path)

        # Update instance attributes
        LOG.info(f"Logging subjects and labels from {data_path}...")
        self.data_path = data_path
        self.subject_list = dataframe["sub"].tolist()

        return dataframe

    def load(
        self,
        data_path: str = None,
        csv_path: str = None,
        one_sub: str = None,
        verbose: int = 2,
        label_col: str = "label",
    ) -> None:
        """
        Loads data from the "downsampled_[sfreq]" folder in the data_path.

        Parameters
        ----------
        data_path : str, optional
            Path to the data. Defaults to self.data_path if None.
        csv_path : str, optional
            Path to the CSV file. Defaults to "participants_info.csv" in data_path.
        one_sub : str, optional
            Subject ID or "random" to select a random subject.
        verbose : int, optional
            Logging verbosity level (0-2).
        label_col : str, optional
            Column name for labels in the CSV file.
        """

        # Ensure data_path is set
        if data_path is None:
            assert self.data_path is not None, "data_path must be set"
            data_path = self.data_path

        # Load participants info
        dataframe = self.preload(data_path, csv_path)

        # Set logging verbosity
        verbosity_levels = {0: logging.NOTSET, 1: logging.WARNING, 2: logging.INFO}
        LOG.setLevel(verbosity_levels.get(verbose, logging.INFO))

        # Select subject(s) to load
        if one_sub == "random":
            one_sub = self.random_sub()
        elif one_sub is not None and one_sub not in self.subject_list:
            raise AttributeError(f"{one_sub} not a valid subject")
        elif one_sub is not None:
            self.subject_list = [one_sub]

        # Load data for selected subject(s)
        data_folder = f"downsampled_{self.sfreq}"
        numpy_filepath = os.path.join(self.data_path, data_folder)
        for file in os.listdir(numpy_filepath):
            self._reset_seed()
            sub = file.split("_")[0]  # The subject ID is placed first in the filename
            if one_sub is not None and sub != one_sub:
                continue
            if sub not in self.subject_list:
                continue

            row = dataframe.loc[dataframe["sub"] == sub]
            sub_data = self._load_sub(os.path.join(numpy_filepath, file))
            if sub_data is None:
                continue  # skip subject if there are no data in the loaded file

            # Process data and labels
            if self.sensors is not None:
                sub_data = sub_data[:, self.sensors, :, :]
            label = row[label_col].item()
            labels = self._process_labels(label, len(sub_data))

            # Handle sampling
            if self.n_samples is not None:
                random_samples = np.random.choice(
                    np.arange(len(sub_data)), self.n_samples, replace=False
                )
                sub_data = sub_data[random_samples]
                labels = labels[random_samples]

            self.data.append(torch.Tensor(sub_data))
            self.labels.append(np.array(labels))
            self.groups += [sub] * len(labels)

        # Format data and labels
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
        self.groups = torch.tensor(self.groups, dtype=int)

        self.n_subjects = len(np.unique(self.groups))

    def _process_labels(self, label: str, n_samples: int) -> list:
        """Process label(s) for a subject."""
        if isinstance(label, str):
            labels = label.split(", ")
            labels = list(map(strip_string, labels))
        else:
            labels = [label]
        if len(labels) == 1:
            if labels[0] in self.subject_list:
                labels = [self.subject_list.index(labels[0])] * n_samples
            else:
                labels = [labels[0]] * n_samples
        return np.array(labels)

    def random_sub(self):
        return np.random.choice(self.subject_list)

    def __len__(self):
        """Returns the length of the dataset (total number of data examples)."""
        return len(self.data)

    def _load_sub(self, filepath: str):
        """Loads a single subject's data."""
        try:
            data = np.load(filepath)
        except IOError:
            LOG.warning(f"There was a problem loading subject {filepath}")
            return None
        if self.zscore:
            data = np.array(list(map(lambda x: zscore(x, axis=-1), data)))
        return torch.Tensor(np.array(data))

    def _select_sensors(self, sensortype: str) -> list:
        """
        Selects MEG sensor slices based on the sensor type.

        Parameters
        ----------
        sensortype : str
            Type of sensor ("MAG", "GRAD", "plannar", or variants).

        Returns
        -------
        list or None
            List of sensor indices if sensortype is valid, otherwise None.
        """
        sensor_mapping = {
            "MAG": [0],
            "GRAD": [1, 2],
            "plannar": [1, 2],
            "GRAD1": [1],
            "plannar1": [1],
            "GRAD2": [2],
            "plannar2": [2],
            "ALL": [0, 1, 2],
        }
        return sensor_mapping.get(sensortype)

    def _assert_sizes(self, train_size, valid_size, test_size=None):
        """Asserts that the sum of the data ratios is equal to 1."""
        if test_size is None:
            test_size = 0
        assert (
            sum((train_size, valid_size, test_size)) == 1
        ), "sum of data ratios must be equal to 1"

    def _leave_subjects_out_split(self, sizes, generator):
        """Leaves subjects out split."""
        indexes = [[], [], []]
        for i, split in enumerate(random_split(np.arange(self.n_subjects), sizes, generator)):
            indexes[i] = [
                idx for sub in split for idx in np.where(self.groups == sub)[0].tolist()
            ]
        return tuple(indexes)

    def _within_subject_split(self, sizes, generator):
        """Splits data within each subject."""
        indexes = []
        index_groups = [[] for _ in range(self.n_subjects)]
        for index, group in enumerate(self.groups):
            index_groups[group].append(index)
        for group in index_groups:
            indexes.append(random_split(group, sizes, generator))
        indexes = zip(*[random_split(group, sizes, generator) for group in index_groups])
        return tuple(sum(map(list, index), []) for index in indexes)

    def data_split(self, train_size: float, valid_size: float, test_size: float = None):
        """
        Splits data into training, validation, and test sets.

        Parameters
        ----------
        train_size : float
            Train set size (%).
        valid_size : float
            Validation set size (%).
        test_size : float, optional
            Test set size (%). Defaults to None.

        Returns
        -------
        tuple
            Indices for the splits.
        """
        self._assert_sizes(train_size, valid_size, test_size)
        generator = torch.Generator().manual_seed(self.random_state)

        sizes = (train_size, valid_size, test_size)
        if self.lso:
            return self._leave_subjects_out_split(sizes, generator)
        elif self.groups is not None:
            return self._within_subject_split(sizes, generator)
        else:
            return random_split(np.arange(len(self)), sizes, generator)

    def torchDataset(self, index):
        """Returns a Torch dataset instance of the torch Dataset class for the given index."""
        return torch.utils.data.TensorDataset(self.data[index], self.labels[index])

    def set_labels(self, labels):
        if type(labels[0]) in (str, np.str_):
            labels = _string_to_int(labels)
        self.labels = torch.Tensor(labels)


class ContinuousDataset(EpochedDataset):
    """
    Creates a dataset for deep learning models from REST data with windowing.

    Parameters
    ----------
    window : int, optional
        Window size in seconds. Defaults to 2.
    overlap : float, optional
        Overlap between windows (0-1). Defaults to 0.
    offset : int, optional
        Offset in seconds. Defaults to 10.
    sfreq : int, optional
        Sampling frequency. Defaults to 500.
    n_subjects : int, optional
        Number of subjects. Defaults to None (all subjects).
    zscore : bool, optional
        Apply z-scoring. Defaults to True.
    n_samples : int, optional
        Number of samples per subject. Defaults to None.
    sensortype : str, optional
        Sensor type. Defaults to None.
    lso : bool, optional
        Leave subjects out. Defaults to False.
    random_state : int, optional
        Random state for reproducibility. Defaults to 0.

    Attributes
    ----------
    window : int
        Window size in seconds.
    overlap : float
        Overlap between windows.
    offset : int
        Offset in seconds.
    sfreq : int
        Sampling frequency.
    n_subjects : int
        Number of subjects.
    n_samples : int
        Number of samples per subject.
    data : torch.Tensor
        Data.
    labels : torch.Tensor
        Labels.
    groups : list
        Groups.
    subject_list : list
        Subject list.

    Methods
    -------
    _load_sub(filepath)
        Loads a subject's data with windowing and overlap.
    """

    def __init__(
        self,
        window: int = 2,
        overlap: float = 0,
        offset: int = 10,
        sfreq: int = 500,
        n_subjects: int = None,
        zscore: bool = True,
        n_samples: int = None,
        sensortype: str = None,
        lso: bool = False,
        random_state: int = 0,
    ) -> None:
        """
        Initializes the ContinuousDataset.

        Args:
        window (int): Window size in seconds.
        overlap (float): Overlap between windows.
        offset (int): Offset in seconds.
        sfreq (int): Sampling frequency.
        n_subjects (int): Number of subjects.
        zscore (bool): Apply z-scoring.
        n_samples (int): Number of samples per subject.
        sensortype (str): Sensor type.
        lso (bool): Leave subjects out.
        random_state (int): Random state for reproducibility.
        """

        super().__init__(sfreq, n_subjects, zscore, n_samples, sensortype, lso, random_state)

        assert 0 <= overlap < 1, "Overlap must be between 0 and 1."
        self.window = window
        self.overlap = overlap
        self.offset = offset

    def _load_sub(self, filepath: str) -> torch.Tensor:
        """
        Loads a subject's data with windowing and overlap.

        Parameters
        ----------
        filepath : str
            Path to the subject's data file.

        Returns
        -------
        torch.Tensor
            Loaded data.
        """
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


# For backwards compatibility purposes
RestDataset = ContinuousDataset
Dataset = EpochedDataset
