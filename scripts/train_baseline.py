import logging
import os
from multiprocessing import shared_memory

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from meegnet.dataloaders import ContinuousDataset, EpochedDataset
from meegnet.parsing import get_model_name, parser, save_config
from meegnet.utils import compute_psd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
LOG = logging.getLogger('meegnet')


def process_sensor(
	sensor_type,
	sensor_pos,
	sensor,
	train_index,
	valid_index,
	train_shm_name,
	test_shm_name,
	labels,
	test_labels,
	fs,
	shape,
	test_shape,
):
	LOG.info(f'Processing sensor_type {sensor_type}, sensor {sensor}')

	train_shm = shared_memory.SharedMemory(name=train_shm_name)
	data = np.ndarray(shape, dtype=np.float32, buffer=train_shm.buf)
	test_shm = shared_memory.SharedMemory(name=test_shm_name)
	test_data = np.ndarray(test_shape, dtype=np.float32, buffer=test_shm.buf)

	data_slice = data[:, sensor_pos, sensor, :]
	test_slice = test_data[:, sensor_pos, sensor, :]

	psd_data = compute_psd(data_slice, fs=fs)
	test_psd = compute_psd(test_slice, fs=fs)

	X_train, y_train = psd_data[train_index], labels[train_index]
	X_valid, y_valid = psd_data[valid_index], labels[valid_index]
	X_test, y_test = test_psd, test_labels

	param_distributions = {
		'C': np.logspace(-2, 2, 10),
		'penalty': ['l1', 'l2'],
		'solver': ['liblinear'],
		'max_iter': [100, 200, 500, 1000],
		'tol': [1e-4, 1e-3, 1e-2, 1e-1],
		'fit_intercept': [True, False],
		'class_weight': [None, 'balanced'],
	}

	model = LogisticRegression()

	random_search = RandomizedSearchCV(
		model, param_distributions, n_iter=100, cv=5, scoring='accuracy', random_state=42
	)
	random_search.fit(X_train, y_train)

	best_params = random_search.best_params_
	cv_accuracy = random_search.best_score_

	best_model = random_search.best_estimator_
	best_model.fit(X_train, y_train)

	valid_accuracy = accuracy_score(y_valid, best_model.predict(X_valid))
	test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
	train_accuracy = accuracy_score(y_train, best_model.predict(X_train))

	results = {
		'sensor_type': sensor_type,
		'sensor': sensor,
		'train_accuracy': train_accuracy,
		'validation_accuracy': valid_accuracy,
		'cv_accuracy': cv_accuracy,
		'test_accuracy': test_accuracy,
		'best_parameters': best_params,
	}
	LOG.info(f'Finished processing sensor_type {sensor_type}, sensor {sensor}')
	train_shm.close()
	test_shm.close()
	return results


if __name__ == '__main__':
	###############
	### PARSING ###
	###############

	args = parser.parse_args()
	save_config(vars(args), args.config)

	if not args.data_path:
		parser.error('--data-path is required')

	n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)
	name = get_model_name(args)

	####################
	### LOADING DATA ###
	####################

	if args.epoched:
		dataset = EpochedDataset(
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			split_sizes=args.train_size,
			sensortype=args.sensors,
			lso=args.lso,
			random_state=args.seed,
			data_path=args.data_path,
			csv_path=args.csv_path,
		)
	else:
		dataset = ContinuousDataset(
			window=args.segment_length,
			overlap=args.overlap,
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			split_sizes=args.train_size,
			sensortype=args.sensors,
			lso=args.lso,
			random_state=args.seed,
			data_path=args.data_path,
			csv_path=args.csv_path,
		)

	dataset.load()
	LOG.info(f'dataset contains a total of {len(dataset)} trials.')

	train_index, valid_index, _ = dataset.split_data()

	# Convert data to numpy float32 for memory efficiency
	LOG.info(f'Converting data to numpy array and np.float32 from {type(dataset.data)}')
	data = np.array(dataset.data, dtype=np.float32)
	labels = dataset.targets.numpy()
	test_data = np.array(dataset.test_data, dtype=np.float32)
	test_labels = dataset.test_targets.numpy()

	train_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
	shared_data = np.ndarray(data.shape, dtype=np.float32, buffer=train_shm.buf)
	np.copyto(shared_data, data)

	test_shm = shared_memory.SharedMemory(create=True, size=test_data.nbytes)
	shared_test = np.ndarray(test_data.shape, dtype=np.float32, buffer=test_shm.buf)
	np.copyto(shared_test, test_data)

	########################
	### START PROCESSING ###
	########################

	LOG.info('Starting parallel processing...')
	sensor_types = dataset.sensors if dataset.sensors is not None else list(range(data.shape[1]))
	all_results = Parallel(n_jobs=-1)(
		delayed(process_sensor)(
			sensor_type,
			sensor_pos,
			sensor,
			train_index,
			valid_index,
			train_shm.name,
			test_shm.name,
			labels,
			test_labels,
			args.sfreq,
			data.shape,
			test_data.shape,
		)
		for sensor_pos, sensor_type in enumerate(sensor_types)
		for sensor in range(data.shape[2])
	)

	########################
	### CLEAN UP MEMORY ###
	########################

	train_shm.close()
	train_shm.unlink()
	test_shm.close()
	test_shm.unlink()

	#######################
	### FIND BEST RESULT ###
	#######################

	best_result = max(all_results, key=lambda x: x['validation_accuracy'])
	LOG.info('Best sensor combination:')
	LOG.info(best_result)

	#######################
	### SAVING RESULTS ###
	#######################

	output_file = os.path.join(args.save_path, name, f'baseline_performance_{name}.npy')
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	np.save(output_file, {'results': all_results, 'best': best_result})

	LOG.info('Performance metrics saved.')
