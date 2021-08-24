import os
import sys
import logging
import torch
from torch import nn
from scipy.io import loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loaders
from parsing import parser
from utils import load_checkpoint, train
from network import AutoEncoder


if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    args = parser.parse_args()
    data_path = args.path
    if not data_path.endswith("/"):
        data_path += "/"
    save_path = args.save
    if not save_path.endswith("/"):
        save_path += "/"
    data_type = args.feature
    batch_size = args.batch_size
    max_subj = args.max_subj
    ch_type = args.elec
    features = args.feature
    debug = args.debug
    chunkload = args.chunkload
    linear = args.linear
    seed = args.seed
    mode = args.mode
    train_size = args.train_size
    num_workers = args.num_workers
    model_name = args.model_name
    patience = args.patience
    learning_rate = args.lr
    log = args.log
    printmem = args.printmem

    ################
    # Starting log #
    ################

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

    #######################
    # Torchsummary checks #
    #######################

    torchsum = True
    try:
        from torchsummary import summary
    except:
        logging.warning("Warning: Error loading torchsummary")
        torchsum = False

    #################
    # Parser checks #
    #################

    if printmem and chunkload:
        logging.info(
            "Warning: chunkload and printmem selected, but chunkload does not allow for printing memory as it loads in chunks during training"
        )

    #####################
    # CUDA verification #
    #####################

    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

    ##############
    # data types #
    ##############

    nchan = 102
    if ch_type == "MAG":
        n_channels = 102
    elif ch_type == "GRAD":
        n_channels = 204
    elif ch_type == "ALL":
        n_channels = 306

    #####################
    # preparing network #
    #####################

    trial_length = TIME_TRIAL_LENGTH
    input_size = (n_channels // 102, nchan, trial_length)

    net = AutoEncoder(
        f"{model_name}_nchan{n_channels}",
        input_size,
    ).to(device)
    lin_size = input_size[0] * input_size[1] * input_size[2]

    if torchsum:
        logging.info(summary(net, (1, lin_size)))
    else:
        logging.info(net)

    # We create loaders and datasets (see dataloaders.py)
    trainloader, validloader, testloader = create_loaders(
        data_path,
        train_size,
        batch_size,
        max_subj,
        ch_type,
        data_type,
        seed=seed,
        num_workers=num_workers,
        chunkload=chunkload,
        debug=debug,
        printmem=printmem,
    )

    if mode == "overwrite":
        save = True
        load = False
    elif mode in ("continue", "evaluate"):
        save = True
        load = True
    else:
        save = False
        load = False

    model_filepath = save_path + net.name + ".pt"
    logging.info(net.name)
    # Actual training (loading nework if existing and load option is True)
    if mode != "evaluate":
        train(
            net,
            trainloader,
            validloader,
            model_filepath,
            criterion=nn.MSELoss(),
            save_model=save,
            load_model=load,
            debug=debug,
            p=patience,
            lr=learning_rate,
            mode=mode,
        )

    # Loading best saved model
    if os.path.exists(model_filepath):
        _, net_state, _ = load_checkpoint(model_filepath)
        net.load_state_dict(net_state)
    else:
        logging.warning(
            f"Error: Can't evaluate model {model_filepath}, file not found."
        )
        sys.exit()

    # evaluating
    logging.info("Evaluating on valid set:")
    results = loadmat(model_filepath[:-2] + "mat")
    logging.info(f"loss: {results['loss_score']}")
    logging.info(f"best epoch: {results['best_epoch']}/{results['n_epochs']}")
    sys.exit()
