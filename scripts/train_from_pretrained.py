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
        parser.error("--data-path is required")

    # script_path = os.getcwd()
    # config_path = os.path.join(script_path, "../default_values.ini")
    config_path = "/home/kikuko/meegnet/default_values.ini"
    default_values = configparser.ConfigParser()
    assert os.path.exists(config_path), "default_values.ini not found"
    default_values.read(config_path)
    default_values = default_values["config"]

    fold = None if args.fold == -1 else int(args.fold)

    input_size = get_input_size(args, default_values)
    name = get_model_name(args)

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        prepare_logging('training', args, LOG, fold, model_name=name)

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
            target_col=args.target_col,
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
            target_col=args.target_col,
        )

    dataset.load(target_col=args.target_col)
    n_outputs = len(np.unique(dataset.targets))

    #####################
    ### LOADING MODEL ###
    #####################

    model_path = "/scratch/kikuko/data/mixed_audit/mixed_audit_meegnet_42_ALL_551.pt"
    LOG.info('Loading existing model:')

    my_model = Model(
        name, args.net_option, input_size, n_outputs, learning_rate=args.lr,
        save_path=os.path.join(args.save_path, args.model_name)
    )
    my_model.load(model_path)
    
    ######################
    ### TRAINING MODEL ###
    ######################

    LOG.info(my_model.name)
    LOG.info(my_model.net)

    my_model.train(dataset)

    #####################
    ### TESTING MODEL ###
    #####################

    # LOG.info("Evaluating model:")
    # evaluate(fold, datasets, args.net_option, args=args)
    # fold kept for BC: Model.test now evaluates the subject holdout from preload.
    my_model.test(dataset)

