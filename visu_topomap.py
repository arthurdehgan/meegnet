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
    classif = "gender"
    RES_PATH = SAVE_PATH + f"results/{rtype}/"

    data_path = "/home/arthur/data/raw_camcan/data/data/CC110033/"
    file_path = data_path + f"{dtype}/{dtype}_raw.fif"
    a = mne.io.read_raw_fif(file_path, preload=True).pick_types(meg=True)
    ch_names = a.info["ch_names"]

    for classifier in classifiers:
        all_scores = []
        for i, elec in enumerate(ch_names):
            if i % 3 == 0:
                file_path = (
                    RES_PATH + f"{classifier}_{classif}_test_scores_elec{elec}.npy"
                )
                all_scores.append(float(np.load(file_path)))

        all_scores = np.asarray(all_scores)
        mask_params = dict(
            marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
        )
        CHANNEL_NAMES = [ch_names[i] for i in range(2, 306, 3)]
        # pval_corr = np.asarray([pval_corr[i] for i in range(2, 306, 3)])

        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[all_scores > 0.53] = True

        fig, ax = plt.subplots()
        im, cn = plot_topomap(
            all_scores,
            a.pick_types(meg="mag").info,
            res=128,
            cmap="Spectral_r",
            show=False,
            names=CHANNEL_NAMES,
            show_names=False,
            vmin=0.5,
            vmax=0.76,
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
