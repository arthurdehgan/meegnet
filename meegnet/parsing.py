import configparser
import os
import tempfile
import time

import configargparse

parser = configargparse.ArgParser()


def save_config(args: dict, config_filepath: str = 'config.ini'):
	"""Saves a dictionnary to a given path.

	Configuration saved in configs directory, with timestamp in filename and SLURM job id when available.

	Args:
	    args (dict): The argument dictionnary to be saved to the config filepath.
	    config_filepath (str): The filepath to be used to save the config file.
	"""
	ini_base = os.path.splitext(os.path.basename(config_filepath))[0]
	config_dir = os.path.join(os.path.dirname(os.path.abspath(config_filepath)), 'configs')
	os.makedirs(config_dir, exist_ok=True)
	run_id = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d-%H%M%S')
	target = os.path.join(config_dir, f'{ini_base}-{run_id}.ini')

	config_object = configparser.ConfigParser()
	config_object.add_section('config')
	for key, value in args.items():
		config_object.set('config', key.replace('_', '-'), str(value))

	fd, tmp = tempfile.mkstemp(dir=config_dir, suffix='.tmp')
	try:
		with os.fdopen(fd, 'w') as conf:
			config_object.write(conf)
		os.replace(tmp, target)
	except BaseException:
		os.unlink(tmp)
		raise


def get_model_name(args) -> str:
	"""Builds the model name used for saving/loading .pt, .mat and log files.

	Format: `{model_name}_{net_option}_{max_subj}_{seed}_{sensors}[_custom_net extras]`
	"""
	name = f'{args.model_name}_{args.net_option}_{args.max_subj}_{args.seed}_{args.sensors}'
	suffixes = ''
	if args.net_option == 'custom_net':
		if args.batchnorm:
			suffixes += '_BN'
		if args.maxpool != 0:
			suffixes += f'_maxpool{args.maxpool}'

		name += f'_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}'
		name += suffixes

	return name


parser.add('-c', '--config', is_config_file=True, default='../default.ini', help='config file path')
parser.add(
	'--testsplit',
	type=int,
	default=-1,
	help='Will remove the 20% holdout set by default and usit for cross-val. Using 5-Fold instead of 4-Fold.',
)
parser.add('--flat', action='store_true', help='will flatten sensors dimension')
parser.add(
	'--dataset', default='rest', choices=['rest', 'passive', 'smt', 'mixed'], help='The camcan MEG dataset to load.'
)
parser.add('--lso', action='store_true', help='wether or not to use Leave Subject Out when splitting data')
parser.add(
	'--randomsearch',
	action='store_true',
	help='Launches one cross-val on a subset of data or full random search depending on testsplit parameter',
)
parser.add(
	'--fold',
	default=-1,
	type=int,
	help='will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if testsplit option is true',
)
parser.add('--n-folds', default=5, type=int, help='number of folds to use when crossval is enabled')
parser.add(
	'--epoched',
	action='store_true',
	help='Flag data as epoched if it is already epoched in the data files. otherwise the dataloader will segment data according to set parameters',
)
parser.add(
	'--crossval',
	action='store_true',
	help='Wether to do a K-FOLD cross-validation (no holdout test set) over all subjects.',
)
parser.add(
	'--segment-length',
	type=float,
	default=2,
	help='The length (in seconds) of the segment to consider for resting-state data.',
)
parser.add(
	'--sfreq', type=int, default=500, help='The sampling frequency of the data, mainly for loading the correct data'
)
parser.add('--maxpool', type=int, default=0, help='adds a maxpool layer in between conv layers')
parser.add('--batchnorm', action='store_true', help='adds a batchnorm layer in between conv layers')
parser.add('--permute-labels', action='store_true', help='Permutes the labesl in order to test for chance level')
parser.add('--printmem', action='store_true', help='Shows RAM information before and during the data loading process.')
parser.add('--log', action='store_true', help='stores all prints in a logfile instead of printing them')
parser.add('--lr', type=float, default=0.00001, help='the starting learning rate of the optimizer')
parser.add('--hlayers', type=int, default=1, help='number of hidden layers')
parser.add('--patience', type=int, default=20, help='patience for early stopping')
parser.add(
	'--min-epoch', type=int, default=20, help='Minimum number of epochs to train before early stopping can trigger.'
)
parser.add('--model-name', type=str, default='net', help='Name of the network for file_save')
parser.add('--num-workers', type=int, default=4, help='number of workers to load data while gpu is processing')
parser.add(
	'--train-size',
	type=float,
	default=0.7,
	help='Proportion of the total pool used for the train set. The test holdout is 10%% of the total pool and '
	'validation takes the remainder (default 0.7 -> 7/9 train, 2/9 valid of the non-test pool).',
)
parser.add(
	'--model-path', type=str, default=None, help='The default path to save all computed data, model and visualisations.'
)
parser.add(
	'--save-path', type=str, default='.', help='The default path to save all computed data, model and visualisations.'
)
parser.add(
	'--data-path', type=str, default=None, help='The path where the downsampled data samples can be found.'
)
parser.add(
	'--csv-path',
	type=str,
	default=None,
	help='Full path to participants_info.csv. Defaults to participants_info.csv in data-path.',
)
parser.add('--raw-path', type=str, default=None, help='The path where the raw data can be found.')
parser.add(
	'--visu-path',
	type=str,
	default=None,
	help='The path where the visualisation matrices wil be saved to and loaded from.',
)
parser.add('--seed', default=42, type=int, help='Seed to use for random splits.')
parser.add('--max-subj', default=1000, type=int, help='maximum number of subjects to use (1000 uses all subjects)')
parser.add(
	'-s', '--sensors', default='MAG', choices=['GRAD', 'MAG', 'ALL'], help='The type of sensors to keep, default=MAG'
)
parser.add('--feature', default='temporal', choices=['temporal', 'bands', 'bins'], help='Data type to use.')
parser.add('-b', '--batch-size', default=128, type=int, help='The batch size used for learning.')
parser.add('-d', '--dropout', default=0.25, type=float, help='The dropout rate of the linear layers')
parser.add(
	'--times',
	action='store_true',
	help='Instead of running the training etc, run a series of test in order to choose best set of workers and batch sizes to get faster epochs.',
)
parser.add(
	'--chunkload',
	action='store_true',
	help='Chunks the data and loads data batch per batch. Will be slower but is necessary when RAM size is too low to handle whole dataset.',
)
parser.add('--debug', action='store_true', help='loads dummy data in the net to ensure everything is working fine')
parser.add(
	'--dropout-option',
	default='same',
	choices=['same', 'double', 'inverted'],
	help='sets if the first dropout and the second are the same or if the first one or the second one should be bigger',
)
parser.add('-l', '--linear', default=100, type=int, help='The size of the second linear layer')
parser.add(
	'-n',
	'--nchan',
	default=100,
	type=int,
	help='the number of channels for the first convolution, the other channel numbers scale with this one.',
)
parser.add(
	'--compute-psd',
	action='store_true',
	help='wether or not to to compute psd using saliency windows in the compute_saliency_maps.py script.',
)
parser.add(
	'--net-option',
	default='meegnet',
	choices=['custom', 'MEEGNet', 'meegnet', 'EEGNet', 'eegnet', 'vgg', 'VGG', 'vanPutNet', 'vanput', 'mlp', 'MLP'],
)
parser.add('--n-samples', type=int, default=-1, help='limit of number of samples per subjects')
parser.add('-f', '--filters', default=8, type=int, help='The size of the first convolution')
parser.add('--overlap', type=float, default=0, help='the overlap value between segments for continous data.')
parser.add(
	'--confidence',
	type=float,
	default=0.95,
	help='the confidence threshold needed for a trial to be selected for visualisation in the compute_ and visu_saliency_maps.py script.',
)
parser.add(
	'--w-size',
	type=int,
	default=300,
	help='The window size for saliency based psd computation in the compute_saliency_maps.py script.',
)
