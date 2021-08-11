import collections
import os
import logging
from itertools import product
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat, loadmat
from params import TIME_TRIAL_LENGTH
from dataloaders import create_loaders
from network import FullNet
from utils import train, accuracy, load_checkpoint, nice_time as nt
from algorithms import ERM, IRM
from misc import seed_hash
from parsing import parser


def gen_hparam(name, default_val, random_val_fn, random_seed):
    """Define a hyperparameter. random_val_fn takes a RandomState and
    returns a random hyperparameter value."""
    param = {}
    random_state = np.random.RandomState(seed_hash(random_seed, name))
    param[name] = (default_val, random_val_fn(random_state))
    return param


if __name__ == "__main__":

    ###########
    # PARSING #
    ###########

    parser.add_argument(
        "--algo",
        default="ERM",
        choices=["ERM", "IRM"],
        help="The algo to use for training (only ERM and IRM implemented for now",
    )
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
    algo = args.algo
    learning_rate = args.lr
    log = args.log
    printmem = args.printmem
    samples = args.samples
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

    # We create loaders and datasets (see dataloaders.py)
    task_train, task_valid, _ = create_loaders(
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
        dattype="task",
        samples=samples,
        infinite=True,
    )
    rest_train, rest_valid, _ = create_loaders(
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
        samples=samples,
    )
    train_loaders = [task_train, rest_train]
    valid_loaders = [task_valid, rest_valid]

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
    hparams = {
        "lr": learning_rate,
        "weight_decay": 0.0,
        "irm_lambda": 1e2,
        "irm_penalty_anneal_iters": 500,
    }
    # hparams.update(
    #     gen_hparam("irm_lambda", 1e2, lambda r: 10 ** r.uniform(-1, 5), seed)
    # )
    # hparams.update(
    #     gen_hparam(
    #         "irm_penalty_anneal_iters", 500, lambda r: int(10 ** r.uniform(0, 4)), seed
    #     )
    # )
    if algo == "ERM":
        algorithm = ERM(input_size, 2, 2, net, hparams, device)
    elif algo == "IRM":
        algorithm = IRM(input_size, 2, 2, net, hparams, device)

    algorithm.to(device)

    k = 0
    epoch = 0
    best_vloss = 1000
    valid_accs, valid_losses = [], []
    train_accs, train_losses = [], []
    while k < patience:
        epoch += 1
        # TRAIN
        algorithm.network.train()
        n_batches = len(rest_train)
        train_minibatches_iterator = zip(*train_loaders)
        valid_minibatches_iterator = zip(*valid_loaders)
        tloss, counter = 0, 0
        for i, minibatches in enumerate(train_minibatches_iterator):
            result = algorithm.update(minibatches)
            loss = result["loss"]
            n = len(minibatches[1])
            tloss += loss * n
            counter += n

            closs = tloss / float(counter)
            progress = (
                f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {closs:.5f}"
            )
            if n_batches > 10:
                if i % (n_batches // 10) == 0:
                    logging.info(progress)
            else:
                logging.info(progress)
        tloss /= float(counter)

        # EVAL
        algorithm.network.eval()
        with torch.no_grad():
            vloss, vacc, counter = 0, 0, 0
            for minibatches in valid_minibatches_iterator:
                all_x = torch.cat([x for x, y in minibatches]).to(device)
                all_y = torch.cat([y for x, y in minibatches]).long().to(device)
                out = algorithm.predict(all_x)
                loss = F.cross_entropy(out, all_y)
                acc = accuracy(out, all_y)
                n = all_y.size(0)
                vloss += loss.sum().data.cpu().numpy() * n
                vacc += acc.sum().data.cpu().numpy() * n
                counter += n
            vloss /= float(counter)
            vacc /= float(counter)

        valid_accs.append(vacc)
        valid_losses.append(vloss)
        train_losses.append(tloss)

        best_vacc = 0.5
        if vloss < best_vloss:
            best_net = algorithm.network
            optimizer = algorithm.optimizer
            best_vacc = vacc
            best_vloss = vloss
            best_epoch = epoch
            k = 0
            if save:
                results = {
                    "acc_score": [best_vacc],
                    "loss_score": [best_vloss],
                    "acc": valid_accs,
                    "valid_loss": valid_losses,
                    "train_loss": train_losses,
                    "best_epoch": best_epoch,
                    "n_epochs": epoch,
                    "patience": patience,
                    "current_patience": k,
                }
                savemat(save_path + net.name + ".mat", results)
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": best_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, model_filepath)
        else:
            k += 1

        logging.info("Epoch: {}".format(epoch))
        logging.info(" [LOSS] TRAIN {} / VALID {}".format(loss, best_vloss))
        logging.info(" [ACC] VALID {}".format(best_vacc))

    checkpoint_vals = collections.defaultdict(lambda: [])
