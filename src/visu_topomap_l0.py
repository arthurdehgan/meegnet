import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from parsing import parser
from mne.viz import plot_topomap
from viz import generate_topomap, load_info


def save_topomap(filepath, weights, info):
    for ch, channel in zip(("GRAD1", "GRAD2", "MAG"), weights):
        fig, ax = plt.subplots()
        _ = generate_topomap(channel, info)
        filename_elements = filepath.split("/")[-1].split("_")
        filename_elements.insert(1, ch)
        filename = "_".join(filename_elements)
        plt.savefig("/".join(filepath.split("/")[:-1] + [filename]), dpi=300)


if __name__ == "__main__":
    parser.add_argument(
        "--average",
        action="store_true",
        help="wether to or not to average all filters",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="will only do a specific fold if specified. must be between 0 and 3, or 0 and 4 if notest option is true",
    )
    args = parser.parse_args()

    model_name = args.model_name
    average = args.average
    linear = args.linear
    seed = args.seed
    batch_size = args.batch_size
    fold = None if args.fold is None else int(args.fold)
    ch_type = args.elec
    dropout = args.dropout
    filters = args.filters
    nchan = args.nchan
    hlayers = args.hlayers
    batchnorm = args.batchnorm
    maxpool = args.maxpool
    data_path = args.path
    if not data_path.endswith("/"):
        data_path += "/"
    save_path = args.save
    if not save_path.endswith("/"):
        save_path += "/"

    # name = f"{model_name}_{seed}_9_b{batch_size}_fold{fold}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}_depth{hlayers}"
    name = f"{model_name}_{seed}_fold{fold}_{ch_type}_dropout{dropout}_filter{filters}_nchan{nchan}_lin{linear}_depth{hlayers}"
    if batchnorm:
        name += "_BN"
    if maxpool != 0:
        name += f"_maxpool{maxpool}"

    info = load_info(data_path)

    i = 0
    weights = np.load(f"{name}_layer{i}_weights.npy")

    # for i, conv_filter in enumerate(weights):
    #     for ch, channel in zip(("GRAD1", "GRAD2", "MAG"), conv_filter):
    #         fig, ax = plt.subplots()
    #         im, cn = plot_topomap(
    #             channel.ravel(),
    #             info,
    #             res=128,
    #             cmap="viridis",
    #             vmax=vmax,
    #             vmin=vmin,
    #             show=False,
    #             show_names=False,
    #             contours=1,
    #             extrapolate="head",
    #         )

    #         cb = fig.colorbar(im)
    #         mne.viz.tight_layout()
    #         plt.savefig(
    #             f"{save_path}figures/{name}_{ch}_{i}_conv_filter_l0_topomap.png",
    #             dpi=300,
    #         )

    if average:
        avg_w = weights.mean(axis=0)
        filename = f"{save_path}figures/{name}_avg_conv_filter_l0_topomap.png"
        save_topomap(filename, avg_w, info)
    else:
        for i, filte in enumerate(weights):
            filename = f"{save_path}figures/{name}_conv_filter_l0_topomap{i}.png"
            save_topomap(filename, filte, info)
