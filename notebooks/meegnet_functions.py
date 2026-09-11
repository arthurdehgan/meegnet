from meegnet.dataloaders import Dataset, RestDataset


def load_single_subject(sub, n_samples, lso, args):
	data_path = getattr(args, 'data_path', None) or args.save_path
	csv_path = getattr(args, 'csv_path', None)
	if args.datatype == 'rest':
		dataset = RestDataset(
			window=args.segment_length,
			overlap=args.overlap,
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			sensortype=args.sensors,
			lso=lso,
			random_state=args.seed,
			data_path=data_path,
			csv_path=csv_path,
		)
	else:
		dataset = Dataset(
			sfreq=args.sfreq,
			n_subjects=args.max_subj,
			n_samples=n_samples,
			sensortype=args.sensors,
			lso=lso,
			random_state=args.seed,
			data_path=data_path,
			csv_path=csv_path,
		)
	dataset.load(one_sub=sub)
	return dataset
