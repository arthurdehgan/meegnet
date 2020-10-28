import os
import gc
import sys
import argparse
from itertools import product
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from scipy.io import savemat, loadmat
from utils import elapsed_time
from params import TIME_TRIAL_LENGTH, DATA_PATH
from dataloaders import create_loaders

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--max-subj",
    default=2000,
    type=int,
    help="maximum number of subjects to use (1000 uses all subjects)",
)
parser.add_argument(
    "-e",
    "--elec",
    default="MAG",
    choices=["GRAD", "MAG", "ALL"],
    help="The type of electrodes to keep, default=MAG",
)
parser.add_argument(
    "--feature",
    default="temporal",
    choices=["temporal", "bands", "bins"],
    help="Data type to use.",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    help="The batch size used for learning.",
)
parser.add_argument(
    "-d",
    "--dropout",
    default=0.25,
    type=float,
    help="The dropout rate of the linear layers",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="loads dummy data in the net to ensure everything is working fine",
)
parser.add_argument(
    "--dropout_option",
    default="same",
    choices=["same", "double", "inverted"],
    help="sets if the first dropout and the second are the same or if the first one or the second one should be bigger",
)
parser.add_argument(
    "-l", "--linear", type=int, help="The size of the second linear layer"
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["overwrite", "continue", "empty_run"],
    default="continue",
    help="CHANGE THIS TODO",
)
parser.add_argument(
    "-f", "--filters", type=int, help="The size of the first convolution"
)
parser.add_argument(
    "-n",
    "--nchan",
    type=int,
    help="the number of channels for the first convolution, the other channel numbers scale with this one",
)


def accuracy(y_pred, target):
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model_state = checkpoint["state_dict"]
    optimizer_state = checkpoint["optimizer"]
    return start_epoch, model_state, optimizer_state


def save_checkpoint(state, filename="checkpoint.pth.tar"):
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
):
    if debug:
        optimizer = optimizer(net.parameters())
    else:
        optimizer = optimizer(net.parameters(), lr=LEARNING_RATE)

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
    else:
        epoch = 0

    p = PATIENCE
    j = 0
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    net.train()
    while j < p:
        epoch += 1
        N_BATCHES = len(trainloader)
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            X, y = batch

            y = y.view(-1).to(device)
            X = X.view(-1, 1, N_CHANNELS, TRIAL_LENGTH).float().to(device)

            net.train()
            out = net.forward(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            print(
                f"Epoch: {epoch} // Batch {i+1}/{N_BATCHES} // loss = {loss}", end="\r"
            )

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
                net.save_model(SAVE_PATH)
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
            }
            savemat(SAVE_PATH + net.name + ".mat", results)

    return net


def evaluate(net, dataloader, criterion=nn.CrossEntropyLoss()):
    net.eval()
    with torch.no_grad():
        LOSSES = 0
        ACCURACY = 0
        COUNTER = 0
        for batch in dataloader:
            X, y = batch
            y = y.view(-1).to(device)
            X = X.view(-1, 1, N_CHANNELS, TRIAL_LENGTH).float().to(device)

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


class FullNet(nn.Module):
    def __init__(
        self,
        model_name,
        input_size,
        filter_size=50,
        n_channels=5,
        n_linear=150,
        dropout=0.3,
        dropout_option="same",
    ):
        if dropout_option == "same":
            dropout1 = dropout
            dropout2 = dropout
        else:
            assert (
                dropout < 0.5
            ), "dropout cannot be higher than .5 in this configuration"
            if dropout_option == "double":
                dropout1 = dropout
                dropout2 = dropout * 2
            elif dropout_option == "inverted":
                dropout1 = dropout * 2
                dropout2 = dropout
            else:
                print(f"{dropout_option} is not a valid option")

        super(FullNet, self).__init__()
        layers = nn.ModuleList(
            [
                nn.Conv2d(1, 5 * n_channels, (1, filter_size)),
                nn.BatchNorm2d(5 * n_channels),
                nn.ReLU(),
                nn.Conv2d(5 * n_channels, 5 * n_channels, (N_CHANNELS, 1)),
                nn.BatchNorm2d(5 * n_channels),
                nn.ReLU(),
                nn.MaxPool2d((1, 5)),
                nn.Conv2d(5 * n_channels, 8 * n_channels, (1, int(filter_size / 10))),
                nn.BatchNorm2d(8 * n_channels),
                nn.ReLU(),
                nn.MaxPool2d((1, 5)),
                nn.Conv2d(8 * n_channels, 16 * n_channels, (1, int(filter_size / 5))),
                nn.BatchNorm2d(16 * n_channels),
                nn.ReLU(),
                Flatten(),
            ]
        )

        lin_size = nn.Sequential(*layers)(torch.zeros(input_size)).shape[-1]

        layers.extend(
            (
                nn.Dropout(dropout1),
                nn.Linear(lin_size, n_linear),
                nn.Dropout(dropout2),
                nn.Linear(n_linear, 2),
            )
        )

        self.model = nn.Sequential(*layers)
        self.name = model_name

    def forward(self, x):
        return self.model(x)

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"

        orig_stdout = sys.stdout
        with open(filepath + self.name + ".txt", "a") as f:
            sys.stdout = f
            summary(self, (1, N_CHANNELS, TRIAL_LENGTH))
            sys.stdout = orig_stdout


class vanPutNet(nn.Module):
    def __init__(self, model_name, input_size, dropout=0.25):

        super(vanPutNet, self).__init__()
        layers = nn.ModuleList(
            [
                nn.Conv2d(1, 100, 3),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(100, 100, 3),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(100, 300, (2, 3)),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(300, 300, (1, 7)),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(dropout),
                nn.Conv2d(300, 100, (1, 3)),
                nn.Conv2d(100, 100, (1, 3)),
                Flatten(),
            ]
        )

        lin_size = nn.Sequential(*layers)(torch.zeros(input_size)).shape[-1]

        layers.append(nn.Linear(lin_size, 2))

        self.model = nn.Sequential(*layers)

        self.name = model_name

    def forward(self, x):
        return self.model(x)

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"

        orig_stdout = sys.stdout
        with open(filepath + self.name + ".txt", "a") as f:
            sys.stdout = f
            summary(self, (1, N_CHANNELS, TRIAL_LENGTH))
            sys.stdout = orig_stdout


if __name__ == "__main__":

    gc.enable()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = parser.parse_args()
    DATA_TYPE = args.feature
    BATCH_SIZE = args.batch_size
    MAX_SUBJ = args.batch_size
    CH_TYPE = args.elec
    if CH_TYPE == "MAG":
        N_CHANNELS = 102
    elif CH_TYPE == "GRAD":
        N_CHANNELS = 204
    elif CH_TYPE == "all":
        N_CHANNELS = 306

    if args.feature == "bins":
        bands = False
        TRIAL_LENGTH = 241
    if args.feature == "bands":
        bands = False
        TRIAL_LENGTH = 5
    elif args.feature == "temporal":
        TRIAL_LENGTH = TIME_TRIAL_LENGTH

    PATIENCE = 20
    LEARNING_RATE = 0.001
    TRAIN_SIZE = 0.8
    SEED = 420
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    SAVE_PATH = "./models/"

    debug = args.debug
    filters = args.filters
    nchan = args.nchan
    dropout = args.dropout
    dropout_option = args.dropout_option
    linear = args.linear

    if debug:
        print("ENTERING DEBUG MODE")
        nchan = 102
        dropout = 0
        dropout_option = "same"

    # net = FullNet("", filters, nchan)
    # lin_size = compute_lin_size(np.zeros((2, 1, N_CHANNELS, TRIAL_LENGTH)), net)

    # net = FullNet(
    #     f"model_{dropout_option}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}",
    #     filters,
    #     nchan,
    #     linear,
    #     dropout,
    #     dropout_option,
    #     lin_size,
    # )
    # lin_size = compute_lin_size(np.zeros((2, 1, N_CHANNELS, TRIAL_LENGTH)), net)

    input_size = (BATCH_SIZE, 1, N_CHANNELS, TRIAL_LENGTH)
    net = vanPutNet("van_Putten_network", input_size, dropout=dropout).to(device)
    print(net)
    print(summary(net, (1, N_CHANNELS, TRIAL_LENGTH)))

    a = time()
    trainloader, validloader, testloader = create_loaders(
        DATA_PATH,
        TRAIN_SIZE,
        BATCH_SIZE,
        MAX_SUBJ,
        CH_TYPE,
        DATA_TYPE,
    )
    print(elapsed_time(time(), a))

    if args.mode == "overwrite":
        save = True
        load = False
    elif args.mode == "continue":
        save = True
        load = True
    else:
        save = False
        loaf = False

    model_filepath = SAVE_PATH + net.name + ".pt"
    train(
        net,
        trainloader,
        validloader,
        model_filepath,
        save_model=save,
        load_model=load,
        debug=debug,
    )
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)

    print(evaluate(net, testloader))
