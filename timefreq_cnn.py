import os
import gc
import sys
from itertools import product
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import savemat, loadmat
from utils import nice_time as nt
from params import TIME_TRIAL_LENGTH, NBINS
from dataloaders import create_loaders
from parser import parser

parser.add_argument(
    "-f",
    "--filters",
    type=tuple,
    default=(7, 7),
    help="The size of the first filters for convolution: tuple of the form (freq_filter (int), time_filter (int))",
)


def accuracy(y_pred, target):
    # Compute accuracy from 2 vectors of labels.
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


class Flatten(nn.Module):
    # Flatten layer used to connect between feature extraction and classif parts of a net.
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def load_checkpoint(filename):
    # Function to load a network state from a filename.
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model_state = checkpoint["state_dict"]
    optimizer_state = checkpoint["optimizer"]
    return start_epoch, model_state, optimizer_state


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    # Saves a checkpoint of the network
    torch.save(state, filename)


def train(
    net,
    trainloader,
    validloader,
    model_filepath,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    save_model=False,
    load_model=False,
    debug=False,
    timing=False,
    mode="overwrite",
    p=20,
    lr=0.00001,
):
    # The train function trains and evaluates the network multiple times and prints the
    # loss and accuracy for each batch and each epoch. Everything is saved in a dictionnary
    # with the best checkpoint of the network.

    if debug:
        optimizer = optimizer(net.parameters())
    else:
        optimizer = optimizer(net.parameters(), lr=lr)

    # Load if asked and if the checkpoint exists in the specified path
    epoch = 0
    if load_model and os.path.exists(model_filepath):
        epoch, net_state, optimizer_state = load_checkpoint(model_filepath)
        net.load_state_dict(net_state)
        optimizer.load_state_dict(optimizer_state)
        results = loadmat(model_filepath[:-2] + "mat")
        best_vacc = results["acc_score"]
        best_vloss = results["loss_score"]
        valid_accs = results["acc"]
        train_accs = results["train_acc"]
        valid_losses = results["valid_loss"]
        train_losses = results["train_loss"]
        best_epoch = results["best_epoch"]
        epoch = results["n_epochs"]
        try:  # For backward compatibility purposes
            if mode == "continue":
                j = 0
                lpatience = patience
            else:
                j = results["current_patience"]
                lpatience = results["patience"]
        except:
            j = 0
            lpatience = patience

        if lpatience != patience:
            print(
                f"Warning: current patience ({patience}) is different from loaded patience ({lpatience})."
            )
            answer = input("Would you like to continue anyway ? (y/n)")
            while answer not in ["y", "n"]:
                answer = input("Would you like to continue anyway ? (y/n)")
            if answer == "n":
                exit()

    elif load_model:
        print(f"Couldn't find any checkpoint named {net.name} in {save_path}")
        j = 0

    else:
        j = 0

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    net.train()

    # The training and evaluation loop with patience early stop. j tracks the patience state.
    while j < p:
        epoch += 1
        n_batches = len(trainloader)
        if timing:
            t1 = time()
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            X, y = batch

            y = y.view(-1).long().to(device)
            X = X.view(-1, *net.input_size).float().to(device)

            net.train()
            out = net.forward(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            progress = f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {loss:.5f}"

            if timing:
                tpb = (time() - t1) / (i + 1)
                et = tpb * n_batches
                progress += f"// time per batch = {tpb:.5f} // epoch time = {nt(et)}"

            print(progress, end="\r")

            condition = i >= 999 or i == n_batches - 1
            if timing and condition:
                return tpb, et

        train_loss, train_acc = evaluate(net, trainloader, criterion)
        valid_loss, valid_acc = evaluate(net, validloader, criterion)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        if valid_loss < best_vloss:
            best_vacc = valid_acc
            best_vloss = valid_loss
            best_net = net
            best_epoch = epoch
            j = 0
            if save_model:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": best_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, model_filepath)
                net.save_model(save_path)
        else:
            j += 1

        print("Epoch: {}".format(epoch))
        print(" [LOSS] TRAIN {} / VALID {}".format(train_loss, valid_loss))
        print(" [ACC] TRAIN {} / VALID {}".format(train_acc, valid_acc))
        if save_model:
            results = {
                "acc_score": [best_vacc],
                "loss_score": [best_vloss],
                "acc": valid_accs,
                "train_acc": train_accs,
                "valid_loss": valid_losses,
                "train_loss": train_losses,
                "best_epoch": best_epoch,
                "n_epochs": epoch,
                "patience": patience,
                "current_patience": j,
            }
            savemat(save_path + net.name + ".mat", results)

    return net


def evaluate(net, dataloader, criterion=nn.CrossEntropyLoss()):
    # function to evaluate a network on a dataloader. will return loss and accuracy
    net.eval()
    with torch.no_grad():
        LOSSES = 0
        ACCURACY = 0
        COUNTER = 0
        for batch in dataloader:
            X, y = batch
            y = y.view(-1).long().to(device)
            X = X.view(-1, *net.input_size).float().to(device)

            nbins = net.nbins
            out = net.forward(X)
            loss = criterion(out, y)
            acc = accuracy(out, y)
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            ACCURACY += acc.sum().data.cpu().numpy() * n
            COUNTER += n
        floss = LOSSES / float(COUNTER)
        faccuracy = ACCURACY / float(COUNTER)
    return floss, faccuracy


class customNet(nn.Module):
    def __init__(
        self,
        model_name,
        freq_input_size,
        time_input_size,
        freqfilter=7,
        timefilter=7,
        nchan=5,
        n_linear=150,
        dropout=0.5,
    ):
        super(customNet, self).__init__()
        self.nbins = freq_input_size[-1]
        self.input_size = tuple(
            list(time_input_size[:-1]) + [time_input_size[-1] + self.nbins]
        )
        self.name = model_name
        print(model_name)

        # self.freqnet = nn.Sequential(
        #     *nn.ModuleList(
        #         [
        #             Flatten(),
        #             nn.Dropout(dropout, inplace=True),
        #         ]
        #     )
        # )

        self.freqnet = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Conv2d(freq_input_size[0], nchan, (nchan, 1)),
                    nn.ReLU(),
                    nn.Conv2d(nchan, nchan, (1, freqfilter)),
                    Flatten(),
                    nn.Dropout(dropout),
                ]
            )
        )

        self.timenet = nn.Sequential(
            *nn.ModuleList(
                [
                    nn.Conv2d(time_input_size[0], nchan, (nchan, 1)),
                    nn.ReLU(),
                    nn.Conv2d(nchan, nchan, (1, timefilter)),
                    Flatten(),
                    nn.Dropout(dropout, inplace=True),
                ]
            )
        )

        time_lin = self._get_lin_size(self.timenet, time_input_size)
        freq_lin = self._get_lin_size(self.freqnet, freq_input_size)
        self.classif = nn.Linear(freq_lin + time_lin, 2)

    def _get_lin_size(self, layers, input_size):
        """Layers must be of type nn.Sequential"""
        return layers(torch.zeros((1, *input_size))).shape[-1]

    def forward(self, x):
        time, freq = x[:, :, :, : -self.nbins], x[:, :, :, -self.nbins :]
        a = self.timenet(time)
        b = self.freqnet(freq)
        x = torch.cat((a, b), dim=1)
        out = self.classif(x)
        return out

    # def forward(self, x):
    #     time, freq = x[:, :, :, : self.nbins], x[:, :, :, self.nbins :]
    #     a = self.freqnet(freq)
    #     b = self.timenet(time)
    #     x = torch.cat((Flatten()(a), Flatten()(b)), dim=1)
    #     out = self.classif(x)
    #     return out

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"

        orig_stdout = sys.stdout
        with open(filepath + self.name + ".txt", "a") as f:
            sys.stdout = f
            print(self)
            sys.stdout = orig_stdout


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("Warning: gpu device not available")
        device = "cpu"

    ###############
    ### PARSING ###
    ###############

    args = parser.parse_args()
    data_path = args.path
    save_path = args.save
    if not save_path.endswith("/"):
        save_path += "/"
    batch_size = args.batch_size
    max_subj = args.max_subj
    ch_type = args.elec
    debug = args.debug
    chunkload = args.chunkload
    freqfilter, timefilter = args.filters
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

    ####################
    ### Starting log ###
    ####################

    if log:
        old_stdout = sys.stdout
        logfile = open(save_path + model_name + ".log", "a")
        sys.stdout = logfile

    ##################
    ### data types ###
    ##################

    data_type = "both"
    if ch_type == "MAG":
        n_channels = 102
    elif ch_type == "GRAD":
        n_channels = 204
    elif ch_type == "ALL":
        n_channels = 306
    else:
        raise (f"Error: invalid channel type: {ch_type}")

    trial_length = TIME_TRIAL_LENGTH
    nbins = NBINS

    ###########################
    ### learning parameters ###
    ###########################

    if debug:
        print("ENTERING DEBUG MODE")
        dropout = 0.5
        dropout_option = "same"
        patience = 2

    #########################
    ### preparing network ###
    #########################

    freq_input_size = (n_channels // 102, nchan, nbins)
    time_input_size = (n_channels // 102, nchan, trial_length)

    net = customNet(
        model_name=f"{model_name}_d{dropout}_f{freqfilter}_{timefilter}_nchan{n_channels}",
        freq_input_size=freq_input_size,
        time_input_size=time_input_size,
        freqfilter=freqfilter,
        timefilter=timefilter,
        nchan=nchan,
        n_linear=linear,
        dropout=dropout,
    ).to(device)

    if times:
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
            print("\n", (x[0], x[1], nt(x[2]), nt(x[3])))

    else:
        print(net)

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
        )

        if debug:
            save = False
            load = False
        elif mode == "overwrite":
            save = True
            load = False
        elif mode in ("continue", "evaluate"):
            save = True
            load = True
        else:
            save = False
            load = False

        model_filepath = save_path + net.name + ".pt"
        print(net.name)
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
            )

        # Loading best saved model
        if os.path.exists(model_filepath):
            _, net_state, _ = load_checkpoint(model_filepath)
            net.load_state_dict(net_state)
        else:
            print(f"Error: Can't evaluate model {model_filepath}, file not found.")
            exit()

        # testing
        print("Evaluating on valid set:")
        results = loadmat(model_filepath[:-2] + "mat")
        print("loss: ", results["loss_score"], " // accuracy: ", results["acc_score"])
        print("best epoch: ", f"{results['best_epoch']}/{results['n_epochs']}")

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

        if log:
            logfile.close()
