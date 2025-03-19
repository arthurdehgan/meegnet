import os
import configparser
import logging
import pandas as pd
from meegnet.parsing import parser, save_config
from meegnet.network import Model
from meegnet.viz import compute_saliency_maps
from meegnet_functions import load_single_subject, get_name, get_input_size, prepare_logging


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

    script_path = os.getcwd()
    config_path = os.path.join(script_path, "../default_values.ini")
    default_values = configparser.ConfigParser()
    assert os.path.exists(config_path), "default_values.ini not found"
    default_values.read(config_path)
    default_values = default_values["config"]

    fold = None if args.fold == -1 else int(args.fold)

    input_size = get_input_size(args, default_values)
    name = get_name(args)

    n_samples = None if int(args.n_samples) == -1 else int(args.n_samples)

    n_outputs = 2

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        prepare_logging("saliencies", args, LOG, fold)

    ##############################
    ### PREPARING SAVE FOLDERS ###
    ##############################

    if args.compute_psd:
        psd_path = os.path.join(args.save_path, "saliency_based_psds", name)
        if not os.path.exists(psd_path):
            os.makedirs(psd_path)

    sal_path = os.path.join(
        args.save_path, "saliency_maps", name, f"{args.confidence}confidence"
    )
    if not os.path.exists(sal_path):
        os.makedirs(sal_path)

    ####################
    ### LOADING DATA ###
    ####################

    csv_file = os.path.join(args.save_path, f"participants_info.csv")
    dataframe = (
        pd.read_csv(csv_file, index_col=0)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)[: args.max_subj]
    )
    subj_list = dataframe["sub"]

    #####################
    ### LOADING MODEL ###
    #####################

    if args.model_path is None:
        model_path = args.save_path
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        LOG.info(f"{model_path} does not exist. Creating folders")
        os.makedirs(model_path)

    my_model = Model(name, args.net_option, input_size, n_outputs, save_path=args.save_path)
    # my_model.from_pretrained()
    my_model.load()

    #################
    ### MAIN LOOP ###
    #################

    for sub in subj_list:
        dataset = load_single_subject(sub, n_samples, args)
        if len(dataset) != 0:
            compute_saliency_maps(
                dataset,
                dataset.target_labels,
                sub,
                sal_path,
                my_model.net,
                threshold=args.confidence,
                epoched=args.epoched,
            )
