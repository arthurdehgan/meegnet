import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.viz import plot_topomap


def load_info(data_path):
    # Chargement des données de potition des capteurs:
    data_path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(
        f"{data_path}/MEG/sample/sample_audvis_raw.fif", preload=False
    )
    return raw.pick_types(meg="mag").info


def generate_topomap(data, info, vmin=None, vmax=None, res=128, cmap="viridis"):
    # fig, ax = plt.subplots()
    im, cn = plot_topomap(
        data.ravel(),
        info,
        res=128,
        cmap="viridis",
        vmax=data.max() if vmax is None else vmax,
        vmin=data.min() if vmin is None else vmin,
        show=False,
        show_names=False,
        contours=1,
        extrapolate="local",
    )

    # cb = fig.colorbar(im)
    mne.viz.tight_layout()
    return im
