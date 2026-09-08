"""This script is intended to work for the camcan MEG dataset with maxfilter
and transform to default common space (mf_transdef) data in BIDS format.

The Camcan dataset is not open access and you need to send a request on the
websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed
through the argparser.

Parallel model: a multiprocessing.Pool of worker processes, each one loading,
resampling and saving a whole subject before moving to the next. No shared queue,
no giant-array pickling across processes, no producer/consumer threads. Processing
is idempotent: subjects already saved (present in participants_info.csv and whose
.npy exists) are skipped, so a rerun resumes where it stopped. Use --max-procs to
cap worker count (raw data on external drives and RAM usage scale with it).

example on how to run the script:
python prepare_data_parallel.py --config="config.ini" --raw-path="/home/user/data/camcan/" --save-path="/home/user/data"
"""

import fcntl
import logging
import multiprocessing as mp
import os

import mne
import numpy as np
import pandas as pd

from meegnet.parsing import parser, save_config

LOG = logging.getLogger('meegnet')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def _good_csv_columns(dataset):
	columns = ['sub', 'age', 'label', 'hand', 'Coil', 'MT_TR']
	if dataset != 'rest':
		columns.append('event_labels')
	return columns


def _read_csv(path: str, columns):
	try:
		if not os.path.exists(path) or os.path.getsize(path) == 0:
			return pd.DataFrame({}, columns=columns)
		return pd.read_csv(path, index_col=0)
	except (pd.errors.EmptyDataError, pd.errors.ParserError):
		return pd.DataFrame({}, columns=columns)


def _append_bad_subject(save_path: str, sub: str, info: str, message: str):
	LOG.info(message)
	path = os.path.join(save_path, 'bad_participants_info.csv')
	columns = ['sub', 'error']
	with open(path, 'a+') as fh:
		fcntl.flock(fh, fcntl.LOCK_EX)
		df = _read_csv(path, columns)
		fh.seek(0)
		fh.truncate()
		df = pd.concat([df, pd.DataFrame([{key: val for key, val in zip(df.columns, [sub, info])}])], ignore_index=True)
		df.to_csv(fh)


def _append_good_subject(save_path: str, dataset: str, sub: str, row):
	path = os.path.join(save_path, 'participants_info.csv')
	columns = _good_csv_columns(dataset)
	with open(path, 'a+') as fh:
		fcntl.flock(fh, fcntl.LOCK_EX)
		df = _read_csv(path, columns)
		cached_sub = df['sub'].tolist() if 'sub' in df.columns else []
		if sub in cached_sub:
			return
		fh.seek(0)
		fh.truncate()
		df = pd.concat([df, pd.DataFrame([{key: val for key, val in zip(df.columns, row)}])], ignore_index=True)
		df.to_csv(fh)


def _already_processed(save_path: str, dataset: str, sub: str, filepath: str) -> bool:
	bad_csv_path = os.path.join(save_path, 'bad_participants_info.csv')
	if sub in _read_csv(bad_csv_path, ['sub', 'error'])['sub'].tolist():
		return True
	good_csv_path = os.path.join(save_path, 'participants_info.csv')
	return sub in _read_csv(good_csv_path, _good_csv_columns(dataset))['sub'].tolist() and os.path.exists(filepath)


def _process_one_subject(job):
	sub_folder, data_path, save_path, dataset, epoched, sfreq = job
	result = {'sub_folder': sub_folder, 'status': 'done', 'info': ''}
	try:
		if dataset == 'rest':
			assert not epoched, "Can't load epoched resting state data as there are no events for it"

		data_filepath = os.path.join(
			data_path,
			'cc700/meg/pipeline/release005/BIDSsep/',
			f'derivatives_{dataset}',
			'aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/',
		)
		user = os.listdir(os.path.join(data_path, 'dataman/useraccess/processed/'))[0]
		source_csv_path = os.path.join(data_path, f'dataman/useraccess/processed/{user}/standard_data.csv')
		df = pd.read_csv(source_csv_path)

		fif_file = ''
		file_list = os.listdir(os.path.join(data_filepath, sub_folder))
		while not fif_file.endswith('.fif'):
			fif_file = os.path.join(data_filepath, sub_folder, file_list.pop())
		sub = sub_folder.split('-')[1]

		if epoched:
			assert dataset != 'rest', 'Cannot generate epochs for resting-state data'
			filename = f'{sub}_{dataset}_epoched.npy'
		else:
			filename = f'{sub}_{dataset}.npy'
		out_path = os.path.join(save_path, f'downsampled_{sfreq}')
		os.makedirs(out_path, exist_ok=True)
		filepath = os.path.join(out_path, filename)

		if _already_processed(save_path, dataset, sub, filepath):
			result['status'] = 'skipped'
			return result

		raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
		bads = raw.info['bads']
		if bads != []:
			_append_bad_subject(save_path, sub, 'bad channels', f'{sub} was dropped because of bad channels {bads}')
			result['status'] = 'bad'
			return result
		if epoched and dataset == 'passive':
			try:
				events = mne.find_events(raw)
			except ValueError as e:
				_append_bad_subject(save_path, sub, 'wrong event timings', f'{sub} could not be used because of {e}')
				result['status'] = 'bad'
				return result
			unique_events = set(events[:, -1])
			if unique_events == {6, 7, 8, 9}:
				event_dict = {6: 'auditory1', 7: 'auditory2', 8: 'auditory3', 9: 'visual'}
				labels = [event_dict[event] for event in events[:, -1]]
			else:
				_append_bad_subject(
					save_path,
					sub,
					f'wrong event found: {unique_events}',
					f'a different event has been found in {sub}: {unique_events}',
				)
				result['status'] = 'bad'
				return result
			data = mne.Epochs(raw, events, tmin=-0.15, tmax=0.65, preload=True)
		else:
			data = raw

		data = data.resample(sfreq=sfreq)
		data = np.array([data.get_data(picks='mag'), data.get_data(picks='planar1'), data.get_data(picks='planar2')])
		if dataset == 'passive':
			data = data.swapaxes(0, 1)
		np.save(filepath, data)

		row = df[df['CCID'] == sub].values.tolist()[0]
		if dataset == 'passive':
			row.append(labels)
		_append_good_subject(save_path, dataset, sub, row)
		return result
	except Exception as e:  # noqa: BLE001 - one bad subject must not kill the pool
		LOG.error('Failed %s: %s: %s', sub_folder, type(e).__name__, e)
		result['status'] = 'error'
		result['info'] = f'{type(e).__name__}: {e}'
		return result


if __name__ == '__main__':
	parser.add(
		'--max-procs',
		type=int,
		default=None,
		help='Number of parallel workers (default: min(4, os.cpu_count())). Cap it when reading from slow '
		'external drives or when RAM is limited.',
	)
	args = parser.parse_args()
	save_config(vars(args), args.config)

	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	######################
	### LOGGING CONFIG ###
	######################

	if args.log:
		log_file = os.path.join(args.save_path, 'prepare_data.log')
		logging.basicConfig(filename=log_file, filemode='a')
		LOG.info(f'Starting logging in {log_file}')

	########################
	### ASSERTION CHECKS ###
	########################

	assert args.raw_path is not None, 'The --raw-path parameter has to be set.'
	assert os.path.exists(args.raw_path), f'The --raw-path "{args.raw_path}" parameter doesnt exist.'
	check_path = os.listdir(args.raw_path)
	assert 'cc700' in check_path and 'dataman' in check_path, (
		'The --raw-path must contain the cc700 and dataman folders in order for this script to work properly.'
	)

	#######################
	### FIXING UP PATHS ###
	#######################

	data_filepath = os.path.join(
		args.raw_path,
		'cc700/meg/pipeline/release005/BIDSsep/',
		f'derivatives_{args.dataset}',
		'aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/',
	)
	subjects = sorted(os.listdir(data_filepath))
	subj_count = len(subjects)

	#######################
	### PARALLEL POOL   ###
	#######################

	n_workers = args.max_procs or min(4, os.cpu_count() or 1)
	n_workers = min(n_workers, subj_count)
	LOG.info(f'Processing {subj_count} subjects with {n_workers} workers...')
	jobs = [(sub, args.raw_path, args.save_path, args.dataset, args.epoched, args.sfreq) for sub in subjects]
	with mp.Pool(processes=n_workers) as pool:
		for res in pool.imap_unordered(_process_one_subject, jobs, chunksize=1):
			LOG.info(f'Finished {res["sub_folder"]}: {res["status"]}' + (f' ({res["info"]})' if res['info'] else ''))
	LOG.info('Processing done !')
