from meegnet.dataloaders import Dataset, RestDataset


def load_single_subject(sub, n_samples, lso, args):
    if args.datatype == "rest":
        dataset = RestDataset(
            window=args.segment_length,
            overlap=args.overlap,
            sfreq=args.sfreq,
            n_subjects=args.max_subj,
            n_samples=n_samples,
            sensortype=args.sensors,
            lso=lso,
            random_state=args.seed,
        )
    else:
        dataset = Dataset(
            sfreq=args.sfreq,
            n_subjects=args.max_subj,
            n_samples=n_samples,
            sensortype=args.sensors,
            lso=lso,
            random_state=args.seed,
        )
    dataset.load(args.save_path, one_sub=sub)
    return dataset
