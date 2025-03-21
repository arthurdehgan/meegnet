import os
import random
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from scipy.stats import zscore
from meegnet.utils import strip_string, stratified_sampling

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


def _string_to_int(array, target_labels=None):
    array = np.array(array)
    if target_labels is None:
        target_labels = np.unique(array)
    for i, element in enumerate(target_labels):
        array[array == element] = i
    return array.astype(int), target_labels


class EpochedDataset:
    """
    Creates a dataset for epoch-based M/EEG data.

    Parameters
    ----------
    sfreq : int, optional
        The sampling frequency, by default 500.
    n_subjects : int, optional
        The number of subjects. Default value is None, which means all subjects are processed.
    scaling : str, optional
        The scaling method. available options are "zscore" and "minmax". By default, minmax.
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
    targets : torch.Tensor
        The targets.
    groups : list
        The groups.
    subject_list : list
        The subject list.
    data_path : str
        The path to the data.
    """

    def __init__(
        self,
        sfreq: float = 500,
        n_subjects: int = None,
        scaling: str = "minmax",
        n_samples: int = None,
        split_sizes: tuple = (0.8, 0.1, 0.1),
        sensortype: str = None,
        lso: bool = False,
        random_state: int = 0,
        target_labels: list = None,
    ):
        if isinstance(split_sizes, float):
            split_sizes = split_sizes, (1 - split_sizes) / 2, (1 - split_sizes) / 2

        self._assert_sizes(*split_sizes)
        self.split_sizes = split_sizes

        self.sfreq = sfreq
        self.n_subjects = n_subjects
        if scaling == "zscore":
            self.scaler = lambda x: zscore(x, axis=-1)
        elif scaling == "minmax":
            self.scaler = lambda x: (x - x.min()) / (x.max() - x.min())
        else:
            raise ValueError(f"{scaling} is an invalid scaling option.")
        self.n_samples = n_samples
        self.lso = lso
        self.sensors = self._select_sensors(sensortype)
        self.random_state = random_state
        self.target_labels = target_labels
        self._reset_seed()

        self.data = []
        self.targets = []
        self.groups = []
        self.subject_list = []
        self.data_path = None
        self.dataframe = None

    def _reset_seed(self):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _load_csv(self, csv_file: str) -> pd.DataFrame:
        """Loads a CSV file, handling index column."""
        with open(csv_file) as f:
            first_line = f.readline()
        return pd.read_csv(csv_file, index_col=0 if first_line.startswith(",") else None)

    def preload(
        self, data_path: str, csv_path: str = None, subject_col: str = "sub"
    ) -> pd.DataFrame:
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

        # Sample participants (if needed)
        if self.n_subjects < len(dataframe):
            self.dataframe = dataframe.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)[: self.n_subjects]
        else:
            self.dataframe = dataframe

        # Update instance attributes
        LOG.info(f"Logging subjects and targets from {data_path}...")
        self.data_path = data_path
        self.subject_list = self.dataframe[subject_col].tolist()

        return self.dataframe

    def _check_index_and_load_csv(self, csv_file) -> pd.DataFrame:
        # check parent folder if csv file not found, for backwards compatibility
        if not os.path.exists(csv_file):
            parent_dir = os.path.dirname(os.path.dirname(csv_file))
            csv_file = os.path.join(parent_dir, os.path.basename(csv_file))
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

    def set_data(self, data, targets, groups=None, target_labels=None):
        """
        Sets the data, targets, and groups for the dataset, with optional random subject selection.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            The dataset containing all subjects' data.
        targets : torch.Tensor or np.ndarray
            The target labels corresponding to the data.
        groups : torch.Tensor or np.ndarray
            The group labels indicating the subject for each sample.
        target_labels : list, optional
            The unique target labels. If None, they will be inferred from the targets.
        """
        if groups is None:
            groups = np.arange(len(data))

        assert (
            len(data) == len(targets) == len(groups)
        ), "Data, targets, and groups must have the same length."

        targets, target_labels = _string_to_int(targets, target_labels)
        # Convert data, targets, and groups to PyTorch tensors if they're not already
        data, targets, groups = self._format_data(data, targets, groups)

        # Randomly select a subset of subjects if n_subjects is specified
        if self.n_subjects is not None:
            unique_subjects = np.unique(groups)
            if len(unique_subjects) > self.n_subjects:
                random_subjects = torch.randperm(len(unique_subjects))[: self.n_subjects]
                random_subjects = unique_subjects[random_subjects]
                subject_mask = [sub in random_subjects for sub in groups]
                data = data[subject_mask]
                targets = targets[subject_mask]
                groups = groups[subject_mask]

        # Update subject list
        self.subject_list = np.unique(groups)
        self.n_subjects = len(self.subject_list)

        # Handle stratified sampling if n_samples is specified
        indexes = []
        for subject in self.subject_list:
            sub_index = (
                list(stratified_sampling(data, targets, self.n_samples, subject, groups))
                if self.n_samples is not None
                else np.where(groups == subject)[0].tolist()
            )
            # Apply self.scaler directly on the selected data
            data[sub_index] = torch.stack(
                [self.scaler(sensor_data) for sensor_data in data[sub_index]]
            )
            indexes += sub_index

        if self.n_samples is not None:
            targets = targets[indexes]
            groups = groups[indexes]
            data = data[indexes]

        # Update instance attributes
        self.data = data
        self.set_targets(targets, target_labels)
        self.set_groups(groups)

    def load(self, *args, **kwargs):
        """
        Wrapper for the load_from_path method.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to load_from_path.
        **kwargs : dict
            Keyword arguments to pass to load_from_path.

        Returns
        -------
        None
        """
        self.load_from_path(*args, **kwargs)

    def load_from_path(
        self,
        data_path: str = None,
        csv_path: str = None,
        one_sub: str = None,
        verbose: int = 2,
        target_col: str = "label",
        subject_col: str = "sub",
    ) -> None:
        """
        Loads data from the data_path.

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
        target_col : str, optional
            Column name for targets in the CSV file.
        """

        # Ensure data_path is set
        if data_path is None:
            assert self.data_path is not None, "data_path must be set"
            data_path = self.data_path

        # Load participants info
        self.preload(data_path, csv_path, subject_col=subject_col)

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
        numpy_filepath = os.path.join(self.data_path)
        data, targets, groups = [], [], []
        for file in os.listdir(numpy_filepath):
            self._reset_seed()
            sub = file.split("_")[0]  # The subject ID is placed first in the filename
            if one_sub is not None and sub != one_sub:
                continue
            if sub not in self.subject_list:
                continue

            row = self.dataframe.loc[self.dataframe[subject_col] == sub]
            sub_data = self._load_sub(os.path.join(numpy_filepath, file))
            if sub_data is None:
                continue  # skip subject if there are no data in the loaded file

            # Process data and targets
            target = row[target_col].item()
            processed_targets = self._process_targets(target, len(sub_data))

            if len(sub_data) == len(processed_targets):
                data.append(torch.Tensor(sub_data))
                targets.append(torch.Tensor(processed_targets))
                groups += [sub] * len(processed_targets)
            else:
                LOG.warning(
                    f"Warning: Number of trials for {sub} does not match number of targets."
                )
                continue

        data = torch.cat(data, 0)
        targets = torch.cat(targets, 0)

        self.set_data(data, targets, groups, target_labels=self.target_labels)

    def _format_data(self, data, targets, groups=None):
        """Formats the data, targets, and groups."""
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if groups is not None and not isinstance(groups, np.ndarray):
            groups = np.array(groups)

        if len(data.shape) < 4:
            data = data.unsqueeze(1)

        if self.sensors is not None:
            sub_data = sub_data[:, self.sensors, :, :]

        return data, targets, groups

    def set_groups(self, groups):
        self._clean_groups(groups)

    def _clean_groups(self, groups=None):
        if groups is not None:
            self.groups = groups
        if type(self.groups[0]) != int:
            self.groups, _ = _string_to_int(self.groups)
        self.groups = torch.tensor(self.groups, dtype=int)

    def _process_targets(self, target: str, n_samples: int) -> list:
        """Process target(s) for a subject."""
        if isinstance(target, str):
            targets = target.split(", ")
            targets = list(map(strip_string, targets))
        else:
            targets = [target]
        if len(targets) == 1:
            if targets[0] in self.subject_list:
                targets = [self.subject_list.index(targets[0])] * n_samples
            else:
                targets = [targets[0]] * n_samples
        if self.target_labels is None:
            self.target_labels = list(set(targets))
        targets = [self.target_labels.index(t) for t in targets]
        return torch.tensor(targets)

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
        data = np.array([list(map(self.scaler, sensor_data)) for sensor_data in data])
        data = torch.from_numpy(data)

        if len(data.shape) < 4:
            data = data.unsqueeze(1)

        return data

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

    def split_data(
        self, train_size: float = None, valid_size: float = None, test_size: float = None
    ):
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
        if train_size is None:
            train_size = self.split_sizes[0]
        if valid_size is None:
            valid_size = self.split_sizes[1]
            # if test_size is None:
            test_size = self.split_sizes[2]

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
        return torch.utils.data.TensorDataset(self.data[index], self.targets[index])

    def set_targets(self, targets, target_labels=None):
        if type(targets[0]) in (str, np.str_):
            targets, target_labels = _string_to_int(targets, target_labels)

        self.targets = torch.Tensor(targets)
        self.target_labels = target_labels


class ContinuousDataset(EpochedDataset):
    """
    Creates a dataset from continuous data by loading continuous data and splitting it in segments according to set parameters.

    Parameters
    ----------
    window : float, optional
        Window size in seconds. Defaults to 2.
    overlap : float, optional
        Overlap between windows (0-1). Defaults to 0.
    offset : int, optional
        Offset in seconds. Defaults to 10.
    sfreq : float, optional
        Sampling frequency. Defaults to 500.
    n_subjects : int, optional
        Number of subjects. Defaults to None (all subjects).
    scaling : str, optional
        The scaling method. available options are "zscore" and "minmax". By default, minmax.
    n_samples : int, optional
        Number of samples per subject. Defaults to None.
    split_sizes(tuple or int), optional
        A tuple of (train_size, valid_size, test_size) for splits or a float <= 1,
        in which case the valid sizes and test sizes are deduced to be as half or the remaining.
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
    targets : torch.Tensor
        targets.
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
        scaling: str = "minmax",
        n_samples: int = None,
        split_sizes: tuple = (0.8, 0.1, 0.1),
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
        scaling (str): The scaling method used.
        n_samples (int): Number of samples per subject.
        split_sizes(tuple or int): a tuple of (train_size, valid_size, test_size)
            for splits or a float <= 1, in which case the valid sizes and test sizes
            are deduced to be as half or the remaining.
        sensortype (str): Sensor type.
        lso (bool): Leave subjects out.
        random_state (int): Random state for reproducibility.
        """

        super().__init__(
            sfreq, n_subjects, scaling, n_samples, split_sizes, sensortype, lso, random_state
        )

        if scaling == "zscore":
            self.scaler = lambda x: zscore(x, axis=-1)
        elif scaling == "minmax":
            self.scaler = lambda x: (x - x.min()) / (x.max() - x.min())
        else:
            raise ValueError(f"{scaling} is an invalid scaling option.")
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
                        trial = [self.scaler(sensor_data) for sensor_data in trial]
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
