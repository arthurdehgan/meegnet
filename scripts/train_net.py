import os
import logging
import toml
from meegnet.dataloaders import Dataset, RestDataset
from meegnet.parsing import parser, save_config
from meegnet.network import Model

LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


if __name__ == "__main__":
    ###########
    # PARSING #
    ###########

    args = parser.parse_args()
    save_config(vars(args), args.config)
    with open("/home/kikuko/meegnet/default_values.toml", "r") as f:
        default_values = toml.load(f)

    fold = None if args.fold == -1 else int(args.fold)
    if args.clf_type in ("eventclf", "toneclf"):
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run event classification"

    if args.feature == "bins":
        trial_length = default_values["TRIAL_LENGTH_BINS"]
    elif args.feature == "bands":
        trial_length = default_values["TRIAL_LENGTH_BANDS"]
    elif args.feature == "temporal":
        trial_length = default_values["TRIAL_LENGTH_TIME"]

    if args.clf_type == "subclf":
        trial_length = int(args.segment_length * args.sfreq)

    if args.sensors == "MAG":
        n_channels = default_values["N_CHANNELS_MAG"]
    elif args.sensors == "GRAD":
        n_channels = default_values["N_CHANNELS_GRAD"]
    else:
        n_channels = default_values["N_CHANNELS_OTHER"]

    input_size = (
        (1, n_channels, trial_length)
        if args.flat
        else (
            n_channels // default_values["N_CHANNELS_MAG"],
            default_values["N_CHANNELS_MAG"],
            trial_length,
        )
    )

    ################
    # Starting log #
    ################

    if args.log:
        log_name = f"{args.model_name}_{args.seed}_{args.sensors}"
        if fold is not None:
            log_name += f"_fold{args.fold}"
        log_name += ".log"
        log_file = os.path.join(args.save_path, log_name)
        logging.basicConfig(filename=log_file, filemode="a")
        LOG.info(f"Starting logging in {log_file}")

    ################
    # Loading data #
    ################

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)
    if args.clf_type == "subclf":
        data_path = os.path.join(args.save_path, f"downsampled_{args.sfreq}")
        n_subjects = len(os.listdir(data_path))
        n_outputs = min(n_subjects, args.max_subj)
        lso = False
    if args.clf_type == "toneclf":
        n_outputs = 3
        lso = True
    else:
        n_outputs = 2
        lso = True

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

    dataset.load(args.save_path)

    LOG.info(f"{args.clf_type} - Training model:")
    name = f"{args.model_name}_{args.seed}_{args.sensors}"
    suffixes = ""
    if args.net_option == "custom_net":
        if args.batchnorm:
            suffixes += "_BN"
        if args.maxpool != 0:
            suffixes += f"_maxpool{args.maxpool}"

        name += f"_dropout{args.dropout}_filter{args.filters}_nchan{args.nchan}_lin{args.linear}_depth{args.hlayers}"
        name += suffixes

    my_model = Model(
        name, args.net_option, input_size, n_outputs, save_path=args.save_path
    )

    LOG.info(my_model.name)
    LOG.info(my_model.net)

    my_model.train(dataset)
    my_model.test(dataset)
    # LOG.info("Evaluating model:")
    # evaluate(fold, datasets, args.net_option, args=args)
