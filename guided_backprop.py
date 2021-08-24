import os
import torch
import logging
from scipy.io import savemat
from torch.nn import ReLU
from parser import parser
from cnn import FullNet, load_checkpoint
from dataloaders import create_loaders
from params import TIME_TRIAL_LENGTH
from captum.attr import GuidedBackprop


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
    ages = (args.age_min, args.age_max)

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

    net_name = f"{model_name}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}"

    if log:
        logging.basicConfig(
            filename=save_path + f"tSNE_{net_name}.log",
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
        net_name,
        input_size,
        filters,
        nchan,
        linear,
        dropout,
        dropout_option,
    ).to(device)

    logging.info(net)

    file_exists = False
    if os.path.exists(save_path + f"Guided_Bprop_{net_name}.mat"):
        logging.warning("GuidedBackprop for this architecture already exists")
        file_exists = True

    if mode == "overwrite" or not file_exists:
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
            ages=ages,
        )

        model_filepath = save_path + f"{net_name}.pt"
        logging.info(net.name)

        _, net_state, _ = load_checkpoint(model_filepath)
        net.load_state_dict(net_state)

        gbp = GuidedBackprop(net)
        gbps = []
        targets = []
        for i, batch in enumerate(validloader):
            X, y = batch
            X = X.view(-1, *net.input_size).to(device)
            y = y.to(torch.int64).to(device)
            for trial, target in zip(X, y):
                gbps.append(
                    gbp.attribute(torch.unsqueeze(trial.float(), 0), target).cpu()
                )
                targets.append(target.cpu().numpy())

            gbps = torch.cat(gbps).numpy()
            savemat(
                save_path + f"gbp_{net_name}.mat", {"gbp": gbps, "targets": targets}
            )
