import os
import sys
import logging
from time import time
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from scipy.io import savemat, loadmat
import pandas as pd
from path import Path as path


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
    p=20,
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
        if mode == "continue":
            j = 0
            lp = p
        else:
            j = results["current_patience"]
            lp = results["patience"]

        if lp != p:
            logging.warning(
                f"Warning: current patience ({p}) is different from loaded patience ({lp})."
            )
            answer = input("Would you like to continue anyway ? (y/n)")
            while answer not in ["y", "n"]:
                answer = input("Would you like to continue anyway ? (y/n)")
            if answer == "n":
                sys.exit()

    elif load_model:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.warning(
            f"Warning: Couldn't find any checkpoint named {net.name} in {save_path}"
        )
        j = 0

    else:
        j = 0

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    best_vacc = 0.5
    best_net = net
    best_epoch = 0
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

            y = y.view(-1).to(device)
            X = X.view(-1, *net.input_size).to(device)

            if permute_labels:
                idx = torch.randperm(y.nelement())
                y = y.view(-1)[idx].view(y.size())

            net.train()
            out = net.forward(X)
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
            j = 0
            if save_model:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": best_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, model_filepath)
                net.save_model(save_path)
        else:
            j += 1

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
                "patience": p,
                "current_patience": j,
            }
            savemat(save_path + net.name + ".mat", results)

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

            out = net.forward(X)
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
            print(sub, "Not Found")
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
