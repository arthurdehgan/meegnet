import os
import sys
import logging
from time import time
import torch
import mne
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.autograd import Variable
from mne.time_frequency.multitaper import psd_array_multitaper
from scipy.io import savemat, loadmat
from path import Path as path


def extract_bands(data, f=None):
    add_axis = False
    if len(data.shape) < 3:
        data = data[np.newaxis, :, :]
        add_axis = True
    if f is None:
        f = np.asarray([float(i / 2) for i in range(data.shape[-1])])
    # data = data[:, :, (f >= 8) * (f <= 12)].mean(axis=2)
    data = [
        data[:, :, (f >= 0.5) * (f <= 4)].mean(axis=-1)[..., None],
        data[:, :, (f >= 4) * (f <= 8)].mean(axis=-1)[..., None],
        data[:, :, (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[:, :, (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[:, :, (f >= 30) * (f <= 60)].mean(axis=-1)[..., None],
        data[:, :, (f >= 60) * (f <= 90)].mean(axis=-1)[..., None],
        data[:, :, (f >= 90) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=2)
    if add_axis:
        return data[0]
    return data


def compute_psd(data, fs):
    # f, psd = welch(data, fs=fs)
    mne.set_log_level(verbose=False)
    psd, f = psd_array_multitaper(data, sfreq=fs, fmax=150)
    return extract_bands(psd, f)


def cuda_check():
    if torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        return "cpu"


def check_PD(mat):
    if len(mat.shape) > 2:
        out = []
        for submat in mat:
            out.append(check_PD(submat))
        return np.array(out)
    else:
        return np.all(np.linalg.eigvals(mat) > 0)


def accuracy(y_pred, target):
    # Compute accuracy from 2 vectors of labels.
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


def load_checkpoint(filename):
    # Function to load a network state from a filename.
    logging.info("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model_state = checkpoint["state_dict"]
    optimizer_state = checkpoint["optimizer"]
    return start_epoch, model_state, optimizer_state


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
    patience=20,
    lr=0.00001,
    save_path="./models",
    permute_labels=False,
):
    # The train function trains and evaluates the network multiple times and prints the
    # loss and accuracy for each batch and each epoch. Everything is saved in a dictionnary
    # with the best checkpoint of the network.

    if debug:
        optimizer = optimizer(net.parameters())
    else:
        optimizer = optimizer(net.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

    # Load if asked and if the checkpoint exists in the specified path
    epoch = 0
    if load_model:
        if os.path.exists(model_filepath):
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
            if mode == "continue":
                patience_state = 0
                loaded_patience = patience
            else:
                patience_state = results["current_patience"]
                loaded_patience = results["patience"]

            if loaded_patience != patience:
                logging.warning(
                    f"Warning: current patience ({patience}) is different from loaded patience ({loaded_patience})."
                )
                answer = input("Would you like to continue anyway ? (y/n)")
                while answer not in ["y", "n"]:
                    answer = input("Would you like to continue anyway ? (y/n)")
                if answer == "n":
                    sys.exit()
        else:
            logging.warning(
                f"Warning: Couldn't find any checkpoint named {net.name} in {save_path}"
            )
            patience_state = 0
    else:
        patience_state = 0

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    best_vacc = 0.5
    best_net = net
    best_epoch = 0
    net.train()

    # The training and evaluation loop with patience early stop. patience_state tracks the patience state.
    while patience_state < patience:
        epoch += 1
        n_batches = len(trainloader)
        if timing:
            t1 = time()
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            X, y = batch

            if permute_labels:
                idx = torch.randperm(y.nelement())
                y = y.view(-1)[idx].view(y.size())

            y = y.view(-1).to(device)
            X = X.view(-1, *net.input_size).to(device)

            net.train()
            out = net.forward(X.float())
            loss = criterion(out, Variable(y.long()))
            loss.backward()
            optimizer.step()

            progress = f"Epoch: {epoch} // Batch {i+1}/{n_batches} // loss = {loss:.5f}"

            if timing:
                tpb = (time() - t1) / (i + 1)
                et = tpb * n_batches
                progress += (
                    f"// time per batch = {tpb:.5f} // epoch time = {nice_time(et)}"
                )

            if n_batches > 10:
                if i % (n_batches // 10) == 0:
                    logging.info(progress)
            else:
                logging.info(progress)

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
            patience_state = 0
            if save_model:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": best_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, model_filepath)
                net.save_model(save_path)
        else:
            patience_state += 1

        logging.info("Epoch: {}".format(epoch))
        logging.info(" [LOSS] TRAIN {} / VALID {}".format(train_loss, valid_loss))
        logging.info(" [ACC] TRAIN {} / VALID {}".format(train_acc, valid_acc))
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
                "current_patience": patience_state,
            }
            savemat(os.path.join(save_path, net.name + ".mat"), results)

    return net


def evaluate(net, dataloader, criterion=nn.CrossEntropyLoss()):
    # function to evaluate a network on a dataloader. will return loss and accuracy
    net.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        logging.warning("Warning: gpu device not available")
        device = "cpu"

    with torch.no_grad():
        LOSSES = 0
        ACCURACY = 0
        COUNTER = 0
        for batch in dataloader:
            X, y = batch
            y = y.view(-1).to(device)
            X = X.view(-1, *net.input_size).to(device)

            out = net.forward(X.float())
            loss = criterion(out, Variable(y.long()))
            acc = accuracy(out, y)
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            ACCURACY += acc.sum().data.cpu().numpy() * n
            COUNTER += n
        floss = LOSSES / float(COUNTER)
        faccuracy = ACCURACY / float(COUNTER)
    return floss, faccuracy


def load_psd_cc_subjects(PSD_PATH, sub_info_path, window, overlap):
    df = pd.read_csv(sub_info_path)
    sub_list = df["Observations"].tolist()
    labels = list(df["gender_code"] - 1)
    psd = []
    for sub in sub_list:
        file_path = path(PSD_PATH) / "PSD_{}_{}_{}".format(sub, window, overlap)
        try:
            psd.append(loadmat(file_path)["data"].ravel())
        except IOError:
            logging.info(sub, "Not Found")
    return np.array(psd), np.array(labels)


def nice_time(time):
    """Returns time in a humanly readable format."""
    m, h, j = 60, 3600, 24 * 3600
    nbj = time // j
    nbh = (time - j * nbj) // h
    nbm = (time - j * nbj - h * nbh) // m
    nbs = time - j * nbj - h * nbh - m * nbm
    if time > m:
        if time > h:
            if time > j:
                nt = "%ij, %ih:%im:%is" % (nbj, nbh, nbm, nbs)
            else:
                nt = "%ih:%im:%is" % (nbh, nbm, nbs)
        else:
            nt = "%im:%is" % (nbm, nbs)
    elif time < 1:
        nt = "<1s"
    else:
        nt = "%is" % nbs
    return nt


def elapsed_time(t0, t1):
    """Time lapsed between t0 and t1.

    Returns the time (from time.time()) between t0 and t1 in a
    more readable fashion.

    Parameters
    ----------
    t0: float,
        time.time() initial measure of time
        (eg. at the begining of the script)
    t1: float,
        time.time() time at the end of the script
        or the execution of a function.

    """
    lapsed = abs(t1 - t0)
    return nice_time(lapsed)
