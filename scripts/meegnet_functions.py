from meegnet.dataloaders import EpochedDataset, ContinuousDataset
import os
import logging
import numpy as np


def load_info():
	return np.load('../camcan_sensor_locations.npy', allow_pickle=True).tolist()


def load_single_subject(sub, n_samples, args, verbose=2):
	data_path = getattr(args, 'data_path', None) or args.save_path
	csv_path = getattr(args, 'csv_path', None)
	target_col = getattr(args, 'target_col', None)
	if args.epoched:
		dataset = EpochedDataset(
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			sensortype=args.sensors,
			lso=args.lso,
			random_state=args.seed,
			data_path=data_path,
			csv_path=csv_path,
			target_col=target_col,
		)
	else:
		dataset = ContinuousDataset(
			window=args.segment_length,
			overlap=args.overlap,
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			sensortype=args.sensors,
			lso=args.lso,
			random_state=args.seed,
			data_path=data_path,
			csv_path=csv_path,
			target_col=target_col,
		)

	dataset.load(one_sub=sub, verbose=verbose)
	return dataset


def prepare_logging(name, args, LOG, fold=None, *, model_name=None):
	"""Sets up logging to `{model_name}[_fold{N}]_{name}.log` inside args.save_path/{model_name}.

	`name` is the log type suffix (e.g. 'training', 'saliencies').
	`model_name` should be the base model name as built by meegnet.parsing.get_model_name.
	"""
	log_name = model_name if model_name is not None else f'{args.model_name}_{args.seed}_{args.sensors}'
	if fold is not None:
		log_name += f'_fold{fold}'
	log_name += f'_{name}.log'
	log_dir = os.path.join(args.save_path, model_name)
	os.makedirs(log_dir, exist_ok=True)
	log_file = os.path.join(log_dir, log_name)
	logging.basicConfig(filename=log_file, filemode='a', force=True)
	LOG.info(f'Starting logging in {log_file}')


def get_input_size(args, default_values):
	if args.feature == 'bins':
		trial_length = int(default_values['TRIAL_LENGTH_BINS'])
	elif args.feature == 'bands':
		trial_length = int(default_values['TRIAL_LENGTH_BANDS'])
	elif args.feature == 'temporal':
		trial_length = int(default_values['TRIAL_LENGTH_TIME'])

	if not args.epoched:
		trial_length = int(args.segment_length * args.sfreq)

	if args.sensors == 'MAG':
		n_channels = int(default_values['N_CHANNELS_MAG'])
	elif args.sensors == 'GRAD':
		n_channels = int(default_values['N_CHANNELS_GRAD'])
	else:
		n_channels = int(default_values['N_CHANNELS_OTHER'])

	return (
		(1, n_channels, trial_length)
		if args.flat
		else (n_channels // int(default_values['N_CHANNELS_MAG']), int(default_values['N_CHANNELS_MAG']), trial_length)
	)
