import os
import sys
import argparse
from itertools import product
from time import time
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.optim as optim
from torchsummary import summary
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from utils import elapsed_time
from params import DATA_PATH, CHAN_DF, SUB_DF

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--elec", default="MAG", help="The type of electrodes to keep, default=MAG"
)
parser.add_argument("--feature", default="freq", help="")
parser.add_argument(
    "-d", "--dropout", type=float, help="The dropout rate of the linear layers"
)
parser.add_argument(
    "--dropout_option",
    choices=["same", "double", "inverted"],
    help="sets if the first dropout and the second are the same or if the first one or the second one should be bigger",
)
parser.add_argument(
    "-l", "--linear", type=int, help="The size of the second linear layer"
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


def normalize(data):
    return data - data.mean(axis=0)[None, :]


def load_freq_data(dataframe, dpath=DATA_PATH, ch_type="MAG"):
    if ch_type == "MAG":
        elec_index = list(range(2, 306, 3))
    elif ch_type == "GRAD":
        elec_index = list(range(0, 306, 3))
        elec_index += list(range(1, 306, 3))
    elif ch_type == "all":
        elec_index = list(range(306))

    X = None
    y = []
    i = 0
    for row in dataframe.iterrows():
        print(f"loading subject {i+1}...")
        sub, lab = row[1]["participant_id"], row[1]["gender"]
        try:
            sub_data = np.array(np.load(dpath + f"{sub}_psd.npy"))[:, elec_index]
        except:
            print("There was a problem loading subject ", sub)

        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        y += [lab] * len(sub_data)
        i += 1
    return torch.Tensor(X).float(), torch.Tensor(y).long()


def load_data(dataframe, dpath=DATA_PATH, ch_type="MAG"):
    if ch_type == "MAG":
        elec_index = list(range(2, 306, 3))
    elif ch_type == "GRAD":
        elec_index = list(range(0, 306, 3))
        elec_index += list(range(1, 306, 3))
    elif ch_type == "all":
        elec_index = list(range(306))
    X = None
    y = []
    i = 0
    for row in dataframe.iterrows():
        print(f"loading subject {i+1}...")
        sub, lab = row[1]["participant_id"], row[1]["gender"]
        sub_data = np.load(dpath + f"{sub}_ICA_transdef_mf.npy")[elec_index]
        sub_data = [
            normalize(sub_data[:, i : i + TRIAL_LENGTH])
            for i in range(OFFSET, sub_data.shape[-1], TRIAL_LENGTH)
        ]
        for sub in sub_data:
            if sub.shape[-1] != TRIAL_LENGTH:
                sub_data.remove(sub)
        sub_data = np.array(sub_data)
        X = sub_data if X is None else np.concatenate((X, sub_data), axis=0)
        y += [lab] * len(sub_data)
        # y += [i] * len(sub_data)
        i += 1
    return torch.Tensor(X).float(), torch.Tensor(y).long()


# def load_subject(sub, data_path=DATA_PATH, data=None, timepoints=500, ch_type="all"):
#     df = pd.read_csv("{}/cleansub_data_camcan_participant_data.csv".format(data_path))
#     df = df.set_index("participant_id")
#     gender = (df["gender"])[sub]
#     # subject_file = '{}/{}/rest/rest_raw.fif'.format(DATA_PATH, sub)
#     subject_file = "{}_rest.mat".format(data_path + sub)
#     # trial = read_raw_fif(subject_file,
#     #                      preload=True).pick_types(meg=True)[:][0]
#     trial = np.load(subject_file)
#     if ch_type == "all":
#         mask = [True for _ in range(len(trial))]
#         n_channels = 306
#     elif ch_type == "mag":
#         mask = CHAN_DF["mag_mask"]
#         n_channels = 102
#     elif ch_type == "grad":
#         mask = CHAN_DF["grad_mask"]
#         n_channels = 204
#     else:
#         raise ("Error : bad channel type selected")
#     trial = trial[mask]
#
#     n_trials = trial.shape[-1] // timepoints
#     for i in range(1, n_trials - 1):
#         curr = trial[:, i * timepoints : (i + 1) * timepoints]
#         curr = curr.reshape(1, n_channels, timepoints)
#         data = curr if data is None else np.concatenate((data, curr))
#     labels = [gender] * (n_trials - 2)
#     data = data.astype(np.float32, copy=False)
#     return data, labels


def load_checkpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model_state = checkpoint["state_dict"]
    optimizer_state = checkpoint["optimizer"]
    return start_epoch, model_state, optimizer_state


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def decorateur(func):
    def load_sub_wrapper(*args, **kwargs):
        sys.stdout.write("loading subjects data...")
        sys.stdout.flush()
        return_val = func(*args, **kwargs)
        sys.stdout.write("\rloading subjects data... done\n")
        sys.stdout.flush()
        return return_val

    return load_sub_wrapper


def create_loaders(train_size, batch_size, max_subj=632):
    data_df = SUB_DF[["participant_id", "gender"]]
    idx = np.random.permutation(range(len(data_df)))
    data_df = data_df.iloc[idx]
    data_df = data_df.iloc[:max_subj]
    N = len(data_df)
    train_size = int(N * train_size)
    remaining_size = N - train_size
    valid_size = int(remaining_size / 2)
    test_size = remaining_size - valid_size

    torch.manual_seed(torch.initial_seed())
    train_index, test_index, valid_index = utils.random_split(
        np.arange(N), [train_size, test_size, valid_size]
    )

    # X_test, y_test = load_data(data_df.iloc[test_index[:]], ch_type='MAG')
    # X_valid, y_valid = load_data(data_df.iloc[valid_index[:]], ch_type='MAG')
    # X_train, y_train = load_data(data_df.iloc[train_index[:]], ch_type='MAG')
    X_test, y_test = load_freq_data(data_df.iloc[test_index[:]], ch_type="MAG")
    X_valid, y_valid = load_freq_data(data_df.iloc[valid_index[:]], ch_type="MAG")
    X_train, y_train = load_freq_data(data_df.iloc[train_index[:]], ch_type="MAG")

    train_dataset = utils.TensorDataset(X_train, y_train)
    valid_dataset = utils.TensorDataset(X_valid, y_valid)
    test_dataset = utils.TensorDataset(X_test, y_test)
    dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    validloader = utils.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataloader, validloader, testloader


def train(
    net,
    dataloader,
    validloader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    save_model=False,
    load_model=False,
):
    optimizer = optimizer(net.parameters(), lr=LEARNING_RATE)
    model_filepath = SAVE_PATH + net.name + ".pt"

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
        for batch in dataloader:
            optimizer.zero_grad()
            X, y = batch
            y = y.view(-1)
            X = X.view(-1, 1, N_CHANNELS, TRIAL_LENGTH).cuda()
            y = y.cuda()

            net.train()
            out = net.forward(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(net, dataloader, criterion)
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

        print("epoch: {}".format(epoch))
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
            y = y.view(-1)
            X = X.view(-1, 1, N_CHANNELS, TRIAL_LENGTH).cuda()
            y = y.cuda()
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


def compute_lin_size(X, network):
    X = torch.Tensor(X)
    return network.feature_extraction.forward(X).shape[-1]


class FullNet(nn.Module):
    def __init__(
        self,
        model_name,
        filter_size=50,
        n_channels=5,
        n_linear=150,
        dropout=0.3,
        dropout_option="same",
        lin_size=200,
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
        self.feature_extraction = nn.Sequential(
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
        )
        self.model = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(lin_size, n_linear),
            nn.Dropout(dropout2),
            nn.Linear(n_linear, 2),
            nn.Softmax(dim=-1),
        )

        self.name = model_name

    def forward(self, x):
        return self.model(self.feature_extraction(x))

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"

        orig_stdout = sys.stdout
        with open(filepath + self.name + ".txt", "a") as f:
            sys.stdout = f
            summary(self, (1, N_CHANNELS, TRIAL_LENGTH))
            sys.stdout = orig_stdout


class vanPutNet(nn.Module):
    def __init__(self, model_name, lin_size=1):

        super(vanPutNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 100, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(100, 100, 3),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(100, 300, (2, 3)),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(300, 300, (1, 7)),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(300, 100, (1, 3)),
            nn.Conv2d(100, 100, (1, 3)),
            Flatten(),
        )
        self.classif = nn.Sequential(nn.Linear(lin_size, 2), nn.Softmax(dim=-1))

        self.name = model_name

    def forward(self, x):
        return self.classif(self.feature_extraction(x))

    def save_model(self, filepath="."):
        if not filepath.endswith("/"):
            filepath += "/"

        orig_stdout = sys.stdout
        with open(filepath + self.name + ".txt", "a") as f:
            sys.stdout = f
            summary(self, (1, N_CHANNELS, TRIAL_LENGTH))
            sys.stdout = orig_stdout


if __name__ == "__main__":

    args = parser.parse_args()
    if args.elec == "MAG":
        N_CHANNELS = 102
    elif args.elec == "GRAD":
        N_CHANNELS = 204
    elif args.elec == "all":
        N_CHANNELS = 306

    if args.feature == "freq":
        TRIAL_LENGTH = 241
    elif args.feature == "time":
        TRIAL_LENGTH = 400

    PATIENCE = 20
    MAX_SUBJ = 632
    BATCH_SIZE = 64
    LEARNING_RATE = 0.00001
    TRAIN_SIZE = 0.8
    OFFSET = 2000
    SEED = 420
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    SAVE_PATH = "./models/"

    filters = args.filters
    nchan = args.nchan
    dropout = args.dropout
    dropout_option = args.dropout_option
    linear = args.linear

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
    net = vanPutNet(f"")
    lin_size = compute_lin_size(np.zeros((2, 1, N_CHANNELS, TRIAL_LENGTH)), net)
    net = vanPutNet(f"van_Putten_network", lin_size)
    net = net.cuda()
    print(summary(net, (1, N_CHANNELS, TRIAL_LENGTH)))

    a = time()
    trainloader, validloader, testloader = create_loaders(
        TRAIN_SIZE, BATCH_SIZE, MAX_SUBJ
    )
    print(elapsed_time(time(), a))

    # train(net, trainloader, validloader, True, True)

    model_filepath = SAVE_PATH + net.name + ".pt"
    _, net_state, _ = load_checkpoint(model_filepath)
    net.load_state_dict(net_state)
    print(evaluate(net, testloader))
