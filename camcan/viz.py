from PIL import Image
import mne
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.viz import plot_topomap


def make_gif(image_list, output=None, duration=100, loop=0):
    if output is None:
        output = ".".join(image_list[0].split(".")[:-1] + ["gif"])
    frames = [Image.open(image) for image in image_list]
    frame_one = frames[0]
    frame_one.save(
        output,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=loop,
    )


def load_info():
    # Chargement des données de potition des capteurs:
    data_path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(
        f"{data_path}/MEG/sample/sample_audvis_raw.fif", preload=False
    )
    # TODO check if we are correct to do this for grad topoplots, maybe use planar
    # (planar1 doesnt work and for grad they do a different computation)
    return raw.pick_types(meg="mag").info


def generate_topomap(
    data,
    info,
    vmin=None,
    vmax=None,
    res=128,
    cmap="viridis",
    colorbar=False,
    tight_layout=False,
):
    if colorbar:
        fig, ax = plt.subplots()
    im, cn = plot_topomap(
        data.ravel(),
        info,
        res=128,
        cmap=cmap,
        vmax=data.max() if vmax is None else vmax,
        vmin=data.min() if vmin is None else vmin,
        show=False,
        show_names=False,
        contours=1,
        extrapolate="local",
    )

    if colorbar:
        _ = fig.colorbar(im)
    if tight_layout:
        mne.viz.tight_layout()
    return im
