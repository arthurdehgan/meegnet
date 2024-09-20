import os
import logging
import configparser
from meegnet.dataloaders import Dataset, RestDataset
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet_functions import get_name, get_input_size, prepare_logging

LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


if __name__ == "__main__":

    ###############
    ### PARSING ###
    ###############

    args = parser.parse_args()
    save_config(vars(args), args.config)

    config_path = "../default_values.ini"
    default_values = configparser.ConfigParser()
    assert os.path.exists(config_path), "default_values.ini not found"
    default_values.read(config_path)
    default_values = default_values["config"]

    fold = None if args.fold == -1 else int(args.fold)
    if args.clf_type == "eventclf":
        assert (
            args.datatype != "rest"
        ), "datatype must be set to passive in order to run event classification"

    input_size = get_input_size(args)
    name = get_name(args)

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)
    if args.clf_type == "subclf":
        data_path = os.path.join(args.save_path, f"downsampled_{args.sfreq}")
        n_subjects = len(os.listdir(data_path))
        n_outputs = min(n_subjects, args.max_subj)
        lso = False
    else:
        n_outputs = 2
        lso = True

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        prepare_logging("training", args, LOG, fold)

    ####################
    ### LOADING DATA ###
    ####################

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

    #####################
    ### LOADING MODEL ###
    #####################

    LOG.info(f"{args.clf_type} - Training model:")
    my_model = Model(name, args.net_option, input_size, n_outputs, save_path=args.save_path)

    LOG.info(my_model.name)
    LOG.info(my_model.net)

    ######################
    ### TRAINING MODEL ###
    ######################

    my_model.train(dataset)

    #####################
    ### TESTING MODEL ###
    #####################

    my_model.test(dataset)

    # LOG.info("Evaluating model:")
    # evaluate(fold, datasets, args.net_option, args=args)
