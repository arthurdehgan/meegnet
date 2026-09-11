import os
import random
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from scipy.stats import zscore
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from meegnet.utils import strip_string, stratified_sampling, string_to_int

LOG = logging.getLogger('meegnet')


def _label_to_trials(label):
	"""Parses a participants-info label into a list of per-trial labels.

	Returns None when the label cannot be interpreted (missing/NaN/empty list).
	Scalar labels (e.g. 'MALE', a subject id) yield a single-element list.
	"""
	if isinstance(label, float) and np.isnan(label):
		return None
	if not isinstance(label, str):
		return [label]
	stripped = label.strip()
	if stripped.startswith('[') and stripped.endswith(']'):
		inner = stripped[1:-1]
		trials = [strip_string(item.strip()) for item in inner.split(', ')]
		if not trials or any(not trial for trial in trials):
			return None
		return trials
	return [strip_string(label)]


def _split_holdout(dataframe, test_size, random_state, target_col=None, subject_col='sub'):
	"""Splits participants_info dataframe into a held-out test set and a train set.

	If target_col given and present, holdout stratified with sklearn's StratifiedGroupKFold
	over per-trial labels, keeping whole subjects as groups
	Otherwise (generally a problem), seeded random slice used.

	Parameters
	----------
	dataframe : pd.DataFrame
		The full participants dataframe.
	test_size : float
		Fraction of the total pool to hold out. 0 or less holds nothing out.
	random_state : int
		Seed used for the split.
	target_col : str, optional
		Column holding the target labels. None disables stratification.
	subject_col : str, optional
		Column holding the subject ids.

	Returns
	-------
	tuple
		(test_dataframe, pool_dataframe)
	"""
	if test_size is None or test_size <= 0:
		return dataframe.iloc[0:0], dataframe

	n_splits = max(2, round(1 / test_size))
	if target_col is not None and target_col in dataframe.columns and n_splits >= 2:
		labels, groups = [], []
		parsed = True
		for sub, label in zip(dataframe[subject_col], dataframe[target_col]):
			trials = _label_to_trials(label)
			if trials is None:
				parsed = False
				break
			labels.extend(trials)
			groups.extend([sub] * len(trials))
		if parsed and len(labels) > 0:
			try:
				skfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
				splits = list(skfold.split(np.zeros(len(labels)), np.asarray(labels), np.asarray(groups)))
				_, test_trial_idx = splits[0]
				test_subjects = set(np.asarray(groups)[test_trial_idx].tolist())
				mask = dataframe[subject_col].isin(test_subjects)
				return dataframe[mask], dataframe[~mask]
			except ValueError:
				pass
	LOG.info(
		'Holdout could not be stratified (unparseable or missing labels, degenerate classes); '
		'using a random subject slice.'
	)
	shuffled = dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)
	n_test = round(len(shuffled) * test_size)
	return shuffled.iloc[-n_test:], shuffled.iloc[:-n_test]


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
			sampler = torch.utils.data.WeightedRandomSampler(weights, replacement=True, num_samples=batch_size)
		else:
			sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

		if weights is None:
			weights = torch.ones(len(dataset))

		batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)

		self._infinite_iterator = iter(
			torch.utils.data.DataLoader(
				dataset, num_workers=num_workers, batch_sampler=_InfiniteSampler(batch_sampler), pin_memory=pin_memory
			)
		)

	def __iter__(self):
		while True:
			yield next(self._infinite_iterator)

	def __len__(self):
		raise ValueError


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
	data_path : str, optional
	    Path to the folder containing the dataset (with the downsampled_{sfreq} subfolder).
	    Can also be set later through load()/load_from_path().
	csv_path : str, optional
	    Full path to the participants info CSV file. Defaults to participants_info.csv in data_path.
	target_col : str, optional
	    Name of the CSV column holding the target labels (e.g. "event_labels" for event
	    classification). Stored on the instance and reused by load()/load_from_path() when
	    no explicit target_col is passed; falls back to "label" when unset.

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
	csv_path : str
	    The path to the participants info CSV file.
	"""

	def __init__(
		self,
		sfreq: float = 500,
		n_subjects: int | None = None,
		scaling: str = 'minmax',
		n_samples: int | None = None,
		split_sizes: tuple = (0.7, 0.2, 0.1),
		sensortype: str | None = None,
		lso: bool = False,
		random_state: int = 0,
		target_labels: list | None = None,
		data_path: str | None = None,
		csv_path: str | None = None,
		target_col: str | None = None,
	):
		if isinstance(split_sizes, float):
			# train = fraction of all data, test holdout fixed at 10%, valid takes the remainder.
			if not 0 < split_sizes < 0.9:
				raise ValueError(f'train_size must be between 0 and 0.9, got {split_sizes}')
			split_sizes = split_sizes, 0.9 - split_sizes, 0.1

		self._assert_sizes(*split_sizes)
		self.split_sizes = split_sizes
		# Fraction of the total subject pool held out as a test set at load time.
		self.test_size = split_sizes[2]

		self.sfreq = sfreq
		self.n_subjects = n_subjects
		if scaling == 'zscore':
			self.scaler = lambda x: zscore(x, axis=-1)
		elif scaling == 'minmax':
			self.scaler = lambda x: (x - x.min()) / (x.max() - x.min())
		else:
			raise ValueError(f'{scaling} is an invalid scaling option.')
		self.n_samples = n_samples
		self.lso = lso
		self.sensors = self._select_sensors(sensortype)
		self.random_state = random_state
		self.target_labels = target_labels
		self.target_col = target_col
		self._reset_seed()

		self.data = []
		self.targets = []
		self.groups = []
		self.subject_list = []
		self.data_path = data_path
		self.csv_path = csv_path
		self.dataframe = None
		self.test_data = []
		self.test_targets = []
		self.test_groups = []
		self.test_subjects = []
		self.test_dataframe = None

	def _reset_seed(self):
		np.random.seed(self.random_state)
		random.seed(self.random_state)
		torch.manual_seed(self.random_state)

	@staticmethod
	def _resolve_target_col(target_col: str | None, instance_col: str | None) -> str:
		"""Explicit arg > instance state > 'label' fallback."""
		return target_col if target_col is not None else (instance_col or 'label')

	def _load_csv(self, csv_file: str) -> pd.DataFrame:
		"""Loads a CSV file, handling index column."""
		with open(csv_file) as f:
			first_line = f.readline()
		return pd.read_csv(csv_file, index_col=0 if first_line.startswith(',') else None)

	def preload(
		self, data_path: str, csv_path: str = None, subject_col: str = 'sub', target_col: str | None = None
	) -> pd.DataFrame:
		"""
		Loads the subject list from a CSV file.

		Parameters
		----------
		data_path : str
		    Path to the folder containing the dataset.
		csv_path : str, optional
		    Full path to the CSV file. When given but not found or None, falls back to
		    "participants_info.csv" inside data_path.
		subject_col : str, optional
		    Column holding the subject ids.
		target_col : str, optional
		    Column holding the target labels, used to stratify the holdout. Explicit
		    arg overrides the instance state; falls back to 'label' when unset.

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

		self.target_col = self._resolve_target_col(target_col, self.target_col)

		if self.lso:
			test_dataframe, dataframe = _split_holdout(
				dataframe, self.test_size, self.random_state, target_col=self.target_col, subject_col=subject_col
			)
		else:
			test_dataframe = dataframe.iloc[0:0]
		self.test_dataframe = test_dataframe
		self.test_subjects = test_dataframe[subject_col].tolist()

		# Sample participants (if needed)
		if self.n_subjects is not None and self.n_subjects < len(dataframe):
			self.dataframe = dataframe.sample(frac=1, random_state=self.random_state).reset_index(drop=True)[
				: self.n_subjects
			]
		else:
			self.dataframe = dataframe

		# Update instance attributes
		LOG.info(f'Loading subjects and targets from {data_path}...')
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
		if first_line.startswith(','):
			df = pd.read_csv(csv_file, index_col=0)
		else:
			df = pd.read_csv(csv_file)
		return df

	def _load_csv(self, data_path, csv_name='participants_info.csv') -> pd.DataFrame:
		csv_file = os.path.join(data_path, csv_name)
		return self._check_index_and_load_csv(csv_file)

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

		assert len(data) == len(targets) == len(groups), 'Data, targets, and groups must have the same length.'

		targets = np.asarray(targets)
		if not np.issubdtype(targets.dtype, np.integer):
			targets, target_labels = string_to_int(targets, target_labels)
		# Convert data, targets, and groups to PyTorch tensors if they're not already
		data, targets, groups = self._format_data(data, targets, groups)

		# Randomly select a subset of subjects if n_subjects is specified.
		# Kept for backwards compatibility (tests, set_data callers):
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
			data[sub_index] = torch.stack([self.scaler(sensor_data) for sensor_data in data[sub_index]])
			indexes += sub_index

		if self.n_samples is not None:
			targets = targets[indexes]
			groups = groups[indexes]
			data = data[indexes]

		# Update instance attributes
		self.data = data
		self.set_targets(targets, target_labels)
		self.set_groups(groups)

	def _set_test_data(self, data, targets, groups):
		"""Set held-out test subjects data.

		Same processing as set_data.
		"""
		assert len(data) == len(targets) == len(groups), 'Data, targets, and groups must have the same length.'
		targets = np.asarray(targets)
		if not np.issubdtype(targets.dtype, np.integer):
			targets, _ = string_to_int(targets, self.target_labels)
		data, targets, groups = self._format_data(data, targets, groups)
		self.test_data = data
		self.test_targets = torch.Tensor(targets)
		self.test_groups = groups

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
		target_col: str | None = None,
		subject_col: str = 'sub',
	) -> None:
		"""
		Loads data from the data_path.

		Parameters
		----------
		data_path : str, optional
		    Path to the data. Defaults to self.data_path if None.
		csv_path : str, optional
		    Full path to the CSV file. Defaults to participants_info.csv in data_path.
		one_sub : str, optional
		    Subject ID or "random" to select a random subject.
		verbose : int, optional
		    Logging verbosity level (0-2).
		target_col : str, optional
		    Column name for targets in the CSV file. When None, reuses the instance's
		    target_col (set through __init__ or preload), falling back to 'label'.
		"""

		# Ensure data_path is set
		data_path = data_path if data_path is not None else self.data_path
		if data_path is None:
			raise ValueError('data_path must be set')

		# csv defaults to data_path root (before downsampled folder resolution)
		if csv_path is None:
			csv_path = self.csv_path
		if csv_path is None:
			csv_path = os.path.join(data_path, 'participants_info.csv')

		# Use downsampled subfolder if available
		downsampled_path = os.path.join(data_path, f'downsampled_{self.sfreq}')
		if os.path.isdir(downsampled_path):
			LOG.info(f'Found downsampled folder, using {downsampled_path}')
			data_path = downsampled_path

		# Load participants info
		self.preload(data_path, csv_path, subject_col=subject_col, target_col=target_col)

		# Set logging verbosity
		verbosity_levels = {0: logging.NOTSET, 1: logging.WARNING, 2: logging.INFO}
		LOG.setLevel(verbosity_levels.get(verbose, logging.INFO))

		# Select subject(s) to load
		if one_sub == 'random':
			one_sub = self.random_sub()
		elif one_sub is not None and one_sub not in self.subject_list:
			raise AttributeError(f'{one_sub} not a valid subject')
		elif one_sub is not None:
			self.subject_list = [one_sub]

		# Load data for selected subject(s). Test subjects kept separate in their own bucket.
		numpy_filepath = os.path.join(self.data_path)
		buckets = {'train': ([], [], []), 'test': ([], [], [])}
		for file in os.listdir(numpy_filepath):
			self._reset_seed()
			sub = file.split('_')[0]  # The subject ID is placed first in the filename
			if one_sub is not None and sub != one_sub:
				continue
			if sub in self.subject_list:
				bucket, dataframe = 'train', self.dataframe
			elif sub in self.test_subjects:
				bucket, dataframe = 'test', self.test_dataframe
			else:
				continue

			row = dataframe.loc[dataframe[subject_col] == sub]
			sub_data = self._load_sub(os.path.join(numpy_filepath, file))
			if sub_data is None:
				continue  # skip subject if there are no data in the loaded file

			# Process data and targets
			target = row[self.target_col].item()
			processed_targets = self._process_targets(target, len(sub_data))

			if len(sub_data) == len(processed_targets):
				buckets[bucket][0].append(torch.Tensor(sub_data))
				buckets[bucket][1].extend(processed_targets)
				buckets[bucket][2].extend([sub] * len(processed_targets))
			else:
				LOG.warning(f'Warning: Number of trials for {sub} does not match number of targets.')
				continue

		if len(buckets['train'][0]) == 0:
			LOG.warning('No valid data loaded — subject skipped or all trials mismatched.')
			return
		data = torch.cat(buckets['train'][0], 0)

		self.set_data(data, buckets['train'][1], buckets['train'][2])

		# Process the held-out test bucket with the same label mapping as the train data.
		if len(buckets['test'][0]) > 0:
			self._set_test_data(torch.cat(buckets['test'][0], 0), buckets['test'][1], buckets['test'][2])

		# Within-subject mode (lso=False, e.g. subject classification). Create
		# trial-level test set after all subjects are loaded
		if not self.lso:
			self._carve_trial_holdout()

	def _carve_trial_holdout(self):
		"""Carves a fraction of each subject's trials into the test set (lso=False mode).

		Every subject stays in the training pool and contributes test trials, so a
		subject-classification model is trained on all subjects and evaluated on
		unseen trials of every subject.
		"""
		if self.test_size is None or self.test_size <= 0 or len(self.data) == 0:
			return
		generator = torch.Generator().manual_seed(self.random_state)
		test_indexes = []
		for sub in np.unique(self.groups.numpy()):
			sub_index = np.where(self.groups == sub)[0]
			n_trials = len(sub_index)
			n_test = round(n_trials * self.test_size)
			if n_trials >= 10:
				n_test = max(1, n_test)
			if n_test <= 0:
				continue
			perm = torch.randperm(n_trials, generator=generator).tolist()
			test_indexes.extend(sub_index[np.array(perm[:n_test])].tolist())
		if len(test_indexes) == 0:
			return
		mask = torch.ones(len(self.data), dtype=torch.bool)
		mask[test_indexes] = False
		self.test_data = self.data[test_indexes]
		self.test_targets = self.targets[test_indexes]
		self.test_groups = self.groups[test_indexes]
		self.data = self.data[mask]
		self.targets = self.targets[mask]
		self.groups = self.groups[mask]
		self.test_subjects = list(self.subject_list)

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
			data = data[:, self.sensors, :, :]

		return data, targets, groups

	def set_groups(self, groups):
		self._clean_groups(groups)

	def _clean_groups(self, groups=None):
		if groups is not None:
			self.groups = groups
		if not isinstance(self.groups[0], int):
			self.groups, _ = string_to_int(self.groups)
		self.groups = torch.tensor(self.groups, dtype=int)

	def _process_targets(self, target: str, n_samples: int) -> list:
		"""Process target(s) for a subject."""
		if isinstance(target, str):
			targets = target.split(', ')
			targets = list(map(strip_string, targets))
		else:
			targets = [target]
		if len(targets) == 1:
			if targets[0] in self.subject_list:
				targets = [self.subject_list.index(targets[0])] * n_samples
			else:
				targets = [targets[0]] * n_samples
		return targets

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
			LOG.warning(f'There was a problem loading subject {filepath}')
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
			'MAG': [0],
			'GRAD': [1, 2],
			'plannar': [1, 2],
			'GRAD1': [1],
			'plannar1': [1],
			'GRAD2': [2],
			'plannar2': [2],
			'ALL': [0, 1, 2],
		}
		return sensor_mapping.get(sensortype)

	def _assert_sizes(self, train_size, valid_size, test_size=None):
		"""Asserts that the sum of the data ratios is equal to 1."""
		if test_size is None:
			test_size = 0
		assert sum((train_size, valid_size, test_size)) == 1, 'sum of data ratios must be equal to 1'

	def _subject_labels(self):
		"""Majority trial label per subject (subject index order 0..n_subjects-1)."""
		labels = []
		for sub in range(self.n_subjects):
			trial_labels = self.targets[np.where(self.groups == sub)[0]].numpy()
			values, counts = np.unique(trial_labels, return_counts=True)
			labels.append(values[np.argmax(counts)])
		return np.asarray(labels)

	def _stratified_subject_split(self, sizes):
		"""Stratified subject-whole split by each subject's majority trial label.

		Keeps whole subjects per split. Splits off a test set first when sizes[2] > 0,
		then a validation set from the remainder. Falls back to a plain random subject
		split in _leave_subjects_out_split when sklearn cannot stratify (e.g. more
		classes than members of a split, singleton classes).
		"""
		total = sum(sizes)
		valid_frac = sizes[1] / total
		test_frac = sizes[2] / total if len(sizes) > 2 else 0.0
		subjects = np.arange(self.n_subjects)
		labels = self._subject_labels()

		if test_frac > 0:
			n_test = round(self.n_subjects * test_frac)
			sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=self.random_state)
			rest_idx, test_idx = next(sss.split(subjects, labels))
			rest_subjects = subjects[rest_idx]
			rest_labels = labels[rest_idx]
			test_subjects = subjects[test_idx]
			n_rest = len(rest_subjects)
			valid_frac_rest = sizes[1] / (sizes[0] + sizes[1])
			n_valid = round(n_rest * valid_frac_rest)
		else:
			rest_subjects = subjects
			rest_labels = labels
			test_subjects = np.array([], dtype=int)
			n_valid = round(self.n_subjects * valid_frac)

		sss = StratifiedShuffleSplit(n_splits=1, test_size=n_valid, random_state=self.random_state)
		train_idx, valid_idx = next(sss.split(rest_subjects, rest_labels))
		train_subjects = rest_subjects[train_idx]
		valid_subjects = rest_subjects[valid_idx]

		return (
			[idx for sub in train_subjects for idx in np.where(self.groups == sub)[0].tolist()],
			[idx for sub in valid_subjects for idx in np.where(self.groups == sub)[0].tolist()],
			[idx for sub in test_subjects for idx in np.where(self.groups == sub)[0].tolist()],
		)

	def _leave_subjects_out_split(self, sizes, generator):
		"""Leaves subjects out split, stratified by subject labels when possible."""
		try:
			return self._stratified_subject_split(sizes)
		except ValueError:
			LOG.info(
				'Subject-level stratification failed (too few subjects per class or class count too high); '
				'using a random subject split.'
			)
			indexes = [[], [], []]
			for i, split in enumerate(random_split(np.arange(self.n_subjects), sizes, generator)):
				indexes[i] = [idx for sub in split for idx in np.where(self.groups == sub)[0].tolist()]
			return tuple(indexes)

	def _leave_subjects_out_cv_split(self, n_folds, fold, generator):
		"""Subject-level K-fold split: fold subjects = validation, rest = training."""
		assert self.n_subjects >= n_folds, 'Cannot do subject-level K-fold with more folds than subjects.'
		subjects = torch.randperm(self.n_subjects, generator=generator).tolist()
		fold_subjects = [subjects[i::n_folds] for i in range(n_folds)]
		valid_subjects = fold_subjects[fold]
		train_subjects = [sub for i, subs in enumerate(fold_subjects) if i != fold for sub in subs]
		train_index = [idx for sub in train_subjects for idx in np.where(self.groups == sub)[0].tolist()]
		valid_index = [idx for sub in valid_subjects for idx in np.where(self.groups == sub)[0].tolist()]
		return train_index, valid_index, None

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

	def _within_subject_cv_split(self, n_folds, fold, generator):
		"""Trial-level K-fold split: every subject has trials in train and valid."""
		index_groups = [[] for _ in range(self.n_subjects)]
		for index, group in enumerate(self.groups):
			index_groups[group].append(index)
		train_index = []
		valid_index = []
		for group in index_groups:
			perm = torch.randperm(len(group), generator=generator).tolist()
			buckets = [perm[i::n_folds] for i in range(n_folds)]
			valid_index.extend(group[p] for p in buckets[fold])
			train_index.extend(group[p] for i in range(n_folds) if i != fold for p in buckets[i])
		return train_index, valid_index, None

	def split_data(self, train_size: float = None, valid_size: float = None, test_size: float = None):
		"""
		Splits data into training, validation, and test sets.

		With explicit sizes (legacy), the loaded data is split three ways. With no
		arguments (default), test subjects are already held out at load time (see
		preload) and only train/valid indexes are returned over the loaded data.

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
		    Indices for the splits. Test element is None in the default path
		    (use testDataset() for the held-out test data).
		"""
		if train_size is not None or valid_size is not None or test_size is not None:
			# Legacy path kept for backwards compatibility (train_baseline.py, tests):
			# explicit sizes split the loaded data into train/valid/test.
			if train_size is None:
				train_size = self.split_sizes[0]
			if valid_size is None:
				valid_size = self.split_sizes[1]
				test_size = self.split_sizes[2]
			if test_size is None:
				test_size = 0

			self._assert_sizes(train_size, valid_size, test_size)
			generator = torch.Generator().manual_seed(self.random_state)

			sizes = (train_size, valid_size, test_size)
			if self.lso:
				return self._leave_subjects_out_split(sizes, generator)
			elif self.groups is not None:
				return self._within_subject_split(sizes, generator)
			else:
				return random_split(np.arange(len(self)), sizes, generator)

		# Default: renormalize train/valid ratios over the loaded subjects;
		# test set is made of held-out subjects, kept separate
		total = self.split_sizes[0] + self.split_sizes[1]
		sizes = (self.split_sizes[0] / total, self.split_sizes[1] / total)
		generator = torch.Generator().manual_seed(self.random_state)
		if self.lso:
			train_index, valid_index, _ = self._leave_subjects_out_split(sizes, generator)
			return train_index, valid_index, None
		elif self.groups is not None:
			train_index, valid_index = self._within_subject_split(sizes, generator)
			return train_index, valid_index, None
		train_index, valid_index = random_split(np.arange(len(self)), sizes, generator)
		return train_index, valid_index, None

	def split_data_cv(self, fold: int, n_folds: int = 5, train_size: float = None, valid_size: float = None):
		"""
		Splits data into training and validation sets for one fold of K-fold cross-validation.

		The fold acts as the validation set (used for early stopping); all remaining
		loaded data is used for training. The test set is the holdout (subject-level for
		lso=True, per-subject trial-level for lso=False), applied identically to every
		fold (see testDataset()).

		With lso=True, folds are stratified by trial labels with sklearn's
		StratifiedGroupKFold, keeping whole subjects as groups; falls back to random
		subject folds when stratification is impossible (subject classif).

		Parameters
		----------
		fold : int
		    The fold to use as validation (0-indexed).
		n_folds : int, optional
		    Number of folds. Defaults to 5.
		train_size : float, optional
		    Kept for backwards compatibility, unused.
		valid_size : float, optional
		    Kept for backwards compatibility, unused.

		Returns
		-------
		tuple
		    (train_index, valid_index, None). Use testDataset() for the test set.
		"""
		assert 0 <= fold < n_folds, f'fold must be between 0 and {n_folds - 1}'

		generator = torch.Generator().manual_seed(self.random_state)
		if self.lso:
			return self._stratified_group_cv_split(n_folds, fold, generator)
		else:
			return self._within_subject_cv_split(n_folds, fold, generator)

	def _stratified_group_cv_split(self, n_folds, fold, generator):
		"""Subject-whole K-fold stratified by trial labels."""
		try:
			skfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
			splits = list(skfold.split(np.zeros(len(self)), self.targets.numpy(), self.groups.numpy()))
			train_index, valid_index = splits[fold]
			return train_index.tolist(), valid_index.tolist(), None
		except ValueError:
			# e.g. singleton classes (subject classification): keep the legacy random subject folds
			LOG.info('StratifiedGroupKFold failed; falling back to random subject folds.')
			return self._leave_subjects_out_cv_split(n_folds, fold, generator)

	def torchDataset(self, index):
		"""Returns a Torch dataset instance of the torch Dataset class for the given index."""
		return torch.utils.data.TensorDataset(self.data[index], self.targets[index])

	def testDataset(self, index=None):
		"""Returns a Torch dataset over the held-out test subject data (empty if no holdout)."""
		if len(self.test_data) == 0:
			return torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
		if index is None:
			return torch.utils.data.TensorDataset(self.test_data, self.test_targets)
		return torch.utils.data.TensorDataset(self.test_data[index], self.test_targets[index])

	def set_targets(self, targets, target_labels=None):
		targets = np.asarray(targets)
		if not np.issubdtype(targets.dtype, np.integer):
			targets, target_labels = string_to_int(targets, target_labels)

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
	    in which case the test size is deduced to be 10% of the total pool and the valid size the remainder.
	sensortype : str, optional
	    Sensor type. Defaults to None.
lso : bool, optional
		Leave subjects out. Defaults to False.
	random_state : int, optional
		Random state for reproducibility. Defaults to 0.
	data_path : str, optional
		Path to the folder containing the dataset (with the downsampled_{sfreq} subfolder).
		Can also be set later through load()/load_from_path().
	csv_path : str, optional
		Full path to the participants info CSV file. Defaults to participants_info.csv in data_path.
	target_col : str, optional
		Name of the CSV column holding the target labels. Stored on the instance and
		reused by load()/load_from_path() when no explicit target_col is passed;
		falls back to "label" when unset.

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
		scaling: str = 'minmax',
		n_samples: int = None,
		split_sizes: tuple = (0.7, 0.2, 0.1),
		sensortype: str = None,
lso: bool = False,
		random_state: int = 0,
		data_path: str | None = None,
		csv_path: str | None = None,
		target_col: str | None = None,
	) -> None:
		"""
				Initializes the ContinuousDataset.

		Args:
				window (int): Window size in seconds.
				overlap (float): Window overlap.
				offset (int): Offset in seconds.
				sfreq (int): Sampling frequency.
				n_subjects (int): Number of subjects.
				scaling (str): The scaling method used.
				n_samples (int): Number of samples per subject.
				split_sizes(tuple or int): a tuple of (train_size, valid_size, test_size)
				    for splits or a float <= 1, in which case the test size is deduced to be
				    10%% of the total pool and the valid size the remainder.
				sensortype (str): Sensor type.
				lso (bool): Leave subjects out.
				random_state (int): Random state for reproducibility.
				data_path (str): Path to the folder containing the dataset.
				csv_path (str): Full path to the participants info CSV file.
				target_col (str): Name of the CSV column holding the target labels.
		"""

		super().__init__(
			sfreq,
			n_subjects,
			scaling,
			n_samples,
			split_sizes,
			sensortype,
			lso,
			random_state,
			data_path=data_path,
			csv_path=csv_path,
			target_col=target_col,
		)

		if scaling == 'zscore':
			self.scaler = lambda x: zscore(x, axis=-1)
		elif scaling == 'minmax':
			self.scaler = lambda x: (x - x.min()) / (x.max() - x.min())
		else:
			raise ValueError(f'{scaling} is an invalid scaling option.')
		assert 0 <= overlap < 1, 'Overlap must be between 0 and 1.'
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
			sub_data = np.load(filepath)
		except IOError:
			LOG.warning(f'There was a problem loading subject {filepath}')
			return None
		except ValueError:
			LOG.warning(f'There was a problem loading subject {filepath}')
			return None
		return self._split_window(sub_data)

	def _split_window(self, sub_data):
		step = int(self.window * self.sfreq * (1 - self.overlap))
		start = int(self.offset * self.sfreq)
		if len(sub_data.shape) < 3:
			sub_data = sub_data[np.newaxis, :]
		data = []
		for i in range(start, sub_data.shape[-1], step):
			trial = sub_data[:, :, i : i + step]
			if trial.shape[-1] == step:
				if not np.isnan(trial).any():
					trial = [self.scaler(sensor_data) for sensor_data in trial]
					data.append(trial)
		return torch.Tensor(np.array(data))


# For backwards compatibility purposes
RestDataset = ContinuousDataset
Dataset = EpochedDataset
