import os
import gc
import sys
import logging
from itertools import product
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from scipy.io import savemat, loadmat
from tsne_torch import TorchTSNE as TSNE
from utils import nice_time as nt
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loaders
from parser import parser
from cnn import FullNet, load_checkpoint


if __name__ == "__main__":

    ###############
    ### PARSING ###
    ###############

    parser.add_argument(
        "-f", "--filters", default=8, type=int, help="The size of the first convolution"
    )

    args = parser.parse_args()
    data_path = args.path
    save_path = args.save
    data_type = args.feature
    batch_size = args.batch_size
    max_subj = args.max_subj
    ch_type = args.elec
    features = args.feature
    chunkload = args.chunkload
    filters = args.filters
    nchan = args.nchan
    dropout = args.dropout
    dropout_option = args.dropout_option
    linear = args.linear
    seed = args.seed
    mode = args.mode
    train_size = args.train_size
    num_workers = args.num_workers
    model_name = args.model_name
    patience = args.patience
    log = args.log

    #####################
    ### PARSER CHECKS ###
    #####################

    assert data_path is not None, "Empty data_path argument!"
    assert save_path is not None, "Empty save_path argument!"
    if not data_path.endswith("/"):
        data_path += "/"
    if not save_path.endswith("/"):
        save_path += "/"

    ####################
    ### Starting log ###
    ####################

    if log:
        logging.basicConfig(
            filename=save_path + model_name + ".log",
            filemode="a",
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

    ###########################
    ### Torchsummary checks ###
    ###########################

    torchsum = True
    try:
        from torchsummary import summary
    except:
        logging.warning("Warning: Error loading torchsummary")
        torchsum = False

    #########################
    ### CUDA verification ###
    #########################

    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

    ##################
    ### data types ###
    ##################

    if ch_type == "MAG":
        n_channels = 102
    elif ch_type == "GRAD":
        n_channels = 204
    elif ch_type == "ALL":
        n_channels = 306
    else:
        raise (f"Error: invalid channel type: {ch_type}")

    if features == "bins":
        bands = False
        trial_length = 241
    if features == "bands":
        bands = False
        trial_length = 5
    elif features == "temporal":
        trial_length = TIME_TRIAL_LENGTH

    #########################
    ### preparing network ###
    #########################

    input_size = (n_channels // 102, 102, trial_length)

    net = FullNet(
        f"{model_name}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}",
        input_size,
        filters,
        nchan,
        linear,
        dropout,
        dropout_option,
    ).to(device)

    if torchsum:
        # logging.info(summary(net, input_size))
        pass
    else:
        logging.info(net)

    # We create loaders and datasets (see dataloaders.py)
    _, validloader, _ = create_loaders(
        data_path,
        train_size,
        batch_size,
        max_subj,
        ch_type,
        data_type,
        seed=seed,
        num_workers=num_workers,
        chunkload=chunkload,
        include=(0, 1, 0),
    )

    model_filepath = save_path + net.name + ".pt"
    logging.info(net.name)

    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    net.eval()
    for batch in validloader:
        X, y = batch
        y = y.view(-1).to(device)
        X = X.view(-1, *net.input_size).to(device)
        out = net.feature_extraction(X)
        break  # TODO this is for testing ! remove after testing

    X_emb = TSNE(
        n_components=2, perplexity=30, n_iter=1000, verbose=True
    ).fit_transform(out)
    np.save("TSNE.npy", X_emb)
