import os
import logging
import configparser
import numpy as np
from meegnet.dataloaders import EpochedDataset, ContinuousDataset
from torch.nn import MSELoss
from meegnet.parsing import parser, save_config, get_model_name
from meegnet.network import Model
from meegnet_functions import get_input_size, prepare_logging

LOG = logging.getLogger('meegnet')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


if __name__ == '__main__':
	###############
	### PARSING ###
	###############

	args = parser.parse_args()
	save_config(vars(args), args.config)

	if not args.data_path:
		parser.error('--data-path is required')

	# script_path = os.getcwd()
	config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../default_values.ini')
	if not os.path.exists(config_path):
		config_path = '/home/kikuko/meegnet/default_values.ini'
	default_values = configparser.ConfigParser()
	assert os.path.exists(config_path), 'default_values.ini not found'
	default_values.read(config_path)
	default_values = default_values['config']

	fold = None if args.fold == -1 else int(args.fold)
	folds = list(range(int(args.n_folds))) if args.crossval else [fold]

	input_size = get_input_size(args, default_values)
	name = get_model_name(args)

	n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)

	######################
	### LOGGING CONFIG ###
	######################

	if args.log:
		prepare_logging('training', args, LOG, None if args.crossval else fold, model_name=name)

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
	n_outputs = len(np.unique(dataset.targets))

	#####################
	### LOADING MODEL ###
	#####################

	n_folds = len(folds)
	cv_accuracies = []

	for fold in folds:
		if args.log and args.crossval:
			prepare_logging('training', args, LOG, fold, model_name=name)
		LOG.info('Training model:')
		model_name = name + (f'_fold{fold}' if fold is not None else '')
		my_model = Model(
			model_name, args.net_option, input_size, n_outputs, learning_rate=float(args.lr), save_path=args.save_path
		)

		LOG.info(my_model.name)
		LOG.info(my_model.net)

		LOG.info(f'dataset contains a total of {len(dataset)} trials.')

		######################
		### TRAINING MODEL ###
		######################

		my_model.train(dataset, fold=fold, min_epoch=args.min_epoch)

		#####################
		### TESTING MODEL ###
		#####################

		# fold kept for Compatibility: Model.test now evaluates the subject holdout from preload
		# so cross-validation folds is ignored here.
		test_loss, test_acc = my_model.test(dataset, fold=fold)
		my_model.tracker.set_test_metrics(test_loss, test_acc)
		cv_accuracies.append(test_acc)

		if args.crossval:
			LOG.info(f'Fold {fold + 1}/{n_folds} test accuracy: {100 * test_acc:.2f}%')

	if args.crossval:
		cv_mean = np.mean(cv_accuracies)
		cv_std = np.std(cv_accuracies)
		LOG.info(f'{n_folds}-fold cross-validation mean accuracy: {100 * cv_mean:.2f}% +/- {100 * cv_std:.2f}%')
