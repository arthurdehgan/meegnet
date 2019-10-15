import warnings
from itertools import product
import mne
from mne.viz import plot_topomap
import scipy as sp
from scipy.stats import binom
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from parser import args
from params import RAW_PATH, DATA_PATH, SAVE_PATH, SUB_DF

N_ELEC = 102


if __name__ == "__main__":

    ftype = args.feature
    classifier = args.clf
    classif = args.label
    channel_types = args.elec

    file_path = RAW_PATH + "matdata/CC120264_ICA_transdef_mf.mat"
    tmp = loadmat(file_path)
    ch_names, ch_pos = tmp["ch_info"][:306, 0], tmp["ch_pos"]
    mags_index = np.arange(2, 306, 3)
    if channel_types == "MAG":
        ch_names = ch_names[mags_index]
        ch_pos = ch_pos[mags_index]
    elif channel_types == "GRAD":
        grads_index = np.array(set(np.arange(306)) - set(mags_index))
        ch_names = ch_names[grads_index]
        ch_pos = ch_pos[grads_index]

    if classif == "gender":
        n_trials = 6000
        chance_level = binom.isf(0.05, n_trials, 0.5) / n_trials
    if classif == "subject":
        n_subj = 628
        n_trials = 6000  # TODO CHANGE, it is wrong
        chance_level = binom.isf(0.05, n_trials, 1 / n_subj) / n_trials
    if classif == "age":
        n_trials = 6000  # TODO change, it is wrong
        chance_level = binom.isf(0.05, n_trials, 1 / 7) / n_trials
    all_scores = []
    for i, elec in enumerate(ch_names):
        elec = elec.strip()
        RES_PATH = SAVE_PATH + f"results/{classif}/{ftype}/{classifier}/"
        file_path = RES_PATH + f"test_scores_elec{elec}.npy"
        try:
            all_scores.append(float(np.load(file_path)))
            if args.verbose > 1:
                print(elec)
        except:
            print("could not load", file_path)

    all_scores = np.asarray(all_scores)
    mask_params = dict(
        marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
    )
    # pval_corr = np.asarray([pval_corr[i] for i in range(2, 306, 3)])

    tt_mask = np.full((len(ch_names)), False, dtype=bool)
    tt_mask[all_scores > chance_level] = True

    fig, ax = plt.subplots()
    im, cn = plot_topomap(
        all_scores,
        ch_pos,
        res=128,
        cmap="Spectral_r",
        vmin=0.45,
        show=False,
        names=ch_names,
        show_names=False,
        mask=tt_mask,
        mask_params=mask_params,
        contours=1,
    )

    cb = fig.colorbar(im)
    mne.viz.tight_layout()
    plt.savefig(
        SAVE_PATH + "figures/" + f"MAG_{classifier}_{classif}_{ftype}.png",
        resolution=300,
    )
