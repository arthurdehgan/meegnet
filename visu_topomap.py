from itertools import product
import mne
from mne.viz import plot_topomap
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from params import DATA_PATH, SAVE_PATH, CHAN_DF, SUB_DF

N_ELEC = 102


if __name__ == "__main__":

    dtype = "rest"
    rtype = "bands"
    classifiers = ["LDA", "QDA"]
    classifs = ["gender", "age", "subject"]

    data_path = "/home/arthur/data/raw_camcan/data/data/CC110033/"
    file_path = data_path + f"{dtype}/{dtype}_raw.fif"
    a = mne.io.read_raw_fif(file_path, preload=True).pick_types(meg=True)
    ch_names = a.info["ch_names"]

    for classif, classifier in product(classifs, classifiers):
        if classif == "gender":
            chance_level = 0.5
        if classif == "subject":
            chance_level = 1 / 622
        if classif == "age":
            chance_level = 1 / 7
        all_scores = []
        for i, elec in enumerate(ch_names):
            RES_PATH = SAVE_PATH + f"results/{classif}/{dtype}/{rtype}/{classifier}/"
            if i % 3 == 0:
                file_path = RES_PATH + f"test_scores_elec{elec}.npy"
                all_scores.append(float(np.load(file_path)))

        all_scores = np.asarray(all_scores)
        mask_params = dict(
            marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
        )
        CHANNEL_NAMES = [ch_names[i] for i in range(2, 306, 3)]
        # pval_corr = np.asarray([pval_corr[i] for i in range(2, 306, 3)])

        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[all_scores > chance_level] = True

        fig, ax = plt.subplots()
        im, cn = plot_topomap(
            all_scores,
            a.pick_types(meg="mag").info,
            res=128,
            cmap="Spectral_r",
            show=False,
            names=CHANNEL_NAMES,
            show_names=False,
            mask=tt_mask,
            mask_params=mask_params,
            contours=1,
        )

        cb = fig.colorbar(im)
        mne.viz.tight_layout()
        plt.savefig(
            SAVE_PATH + "figures/" + f"MAG_{classifier}_{classif}_{dtype}_{rtype}.png",
            resolution=300,
        )
