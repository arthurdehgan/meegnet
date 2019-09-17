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
    ctype = "mf"
    rtype = "bands"
    classifiers = ["LDA"]
    # classifs = ["gender", "age", "subject"]
    classifs = ["gender"]

    fname = f"sub-CC110033_ses-{dtype}_task-{dtype}_proc-sss.fif"
    data_path = f"/home/arthur/data/camcan/data/meg_{dtype}_{ctype}/"
    file_path = data_path + f"sub-CC110033/ses-{dtype}/meg/{fname}"
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
            RES_PATH = (
                SAVE_PATH + f"results/{classif}/{dtype}_{ctype}/{rtype}/{classifier}/"
            )
            try:
                file_path = RES_PATH + f"test_scores_elec{elec}.npy"
                all_scores.append(float(np.load(file_path)))
                print(CHAN_DF.iloc[i])
            except:
                pass

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
            SAVE_PATH
            + "figures/"
            + f"MAG_{classifier}_{classif}_{dtype}_{ctype}_{rtype}.png",
            resolution=300,
        )
