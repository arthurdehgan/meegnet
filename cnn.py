from itertools import product
import os
import logging
import torch
from scipy.io import loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loaders
from network import FullNet
from utils import train, load_checkpoint, nice_time as nt
from parsing import parser

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
    times = args.times
    patience = args.patience
    learning_rate = args.lr
    log = args.log
    printmem = args.printmem
    ages = (args.age_min, args.age_max)

    ##############
    # CUDA CHECK #
    ##############

    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

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

    ##############
    # data types #
    ##############

    if ch_type == "MAG":
        n_channels = 102
    elif ch_type == "GRAD":
        n_channels = 204
    elif ch_type == "ALL":
        n_channels = 306

    if features == "bins":
        bands = False
        trial_length = 241
    if features == "bands":
        bands = False
        trial_length = 5
    elif features == "temporal":
        trial_length = TIME_TRIAL_LENGTH

    #######################
    # learning parameters #
    #######################

    if debug:
        logging.debug("ENTERING DEBUG MODE")
        max_subj = 20
        dropout = 0.5
        dropout_option = "same"
        patience = 1

    #####################
    # preparing network #
    #####################

    input_size = (n_channels // 102, 102, trial_length)

    # net = vanPutNet("vanputnet_512linear_GRAD", input_size).to(device)
    net = FullNet(
        # f"{model_name}_{dropout_option}_dropout{dropout}_filter{filters}_nchan{n_channels}_lin{linear}",
        f"{model_name}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}",
        input_size,
        filters,
        nchan,
        linear,
        dropout,
        dropout_option,
    ).to(device)

    if times:
        # OBSOLETE: DO NOT WORK ANYMORE, WILL CAUSE CRASHES TODO
        # overrides default mode !
        # tests different values of workers and batch sizes to check which is the fastest
        num_workers = [16, 32, 64, 128]
        batch_sizes = [16, 32]
        perfs = []
        for nw, bs in product(num_workers, batch_sizes):
            tl, vl, _ = create_loaders(
                data_path,
                train_size,
                bs,
                max_subj,
                ch_type,
                data_type,
                num_workers=nw,
                debug=debug,
                chunkload=chunkload,
            )
            tpb, et = train(net, tl, vl, "", lr=learning_rate, timing=True)
            perfs.append((nw, bs, tpb, et))

        for x in sorted(perfs, key=lambda x: x[-1]):
            logging.info(f"\n{x[0]} {x[1]} {nt(x[2])} {nt(x[3])}")

    else:

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
            include=(1, 1, 0),
            ages=ages,
        )

        if torchsum:
            logging.info(summary(net, input_size))
        else:
            logging.info(net)

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
                save_model=save,
                load_model=load,
                debug=debug,
                p=patience,
                lr=learning_rate,
                mode=mode,
                save_path=save_path,
            )

        # Loading best saved model
        if os.path.exists(model_filepath):
            _, net_state, _ = load_checkpoint(model_filepath)
            net.load_state_dict(net_state)
        else:
            logging.warning(
                f"Error: Can't evaluate model {model_filepath}, file not found."
            )
            exit()

        # testing
        logging.info("Evaluating on valid set:")
        results = loadmat(model_filepath[:-2] + "mat")
        logging.info(
            f"loss: {results['loss_score']} // accuracy: {results['acc_score']}"
        )
        logging.info(f"best epoch: {results['best_epoch']}/{results['n_epochs']}")
        exit()

        # # Final testing
        # print("Evaluating on test set:")
        # tloss, tacc = evaluate(net, testloader)
        # print("loss: ", tloss, " // accuracy: ", tacc)
        # if save:
        #     results = loadmat(model_filepath[:-2] + "mat")
        #     print("best epoch: ", f"{results['best_epoch']}/{results['n_epochs']}")
        #     results["test_acc"] = tacc
        #     results["test_loss"] = tloss
        #     savemat(save_path + net.name + ".mat", results)
