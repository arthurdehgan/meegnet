"""This script is intenden to work for the camcan MEG dataset with maxfilter and transform to default common space (mf_transdef) data.

The Camcan dataset is not open access and you need to send a request on the websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed through the argparser.

TODO:
    Maybe add a prompt on the amount of disk space and time required ? (later)
    Crawl through all of the data with the option to select data_tupe (rest/passive/smt)
        for smt and passive, cut in trials with labels and change sfreq

"""
import os
import mne
import logging
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from camcan.parsing import parser

# "Arthur_Dehgan_1160"
parser.add_argument(
    "--user",
    type=str,
    help="name of the user folder for loading csv data",
)
parser.add_argument(
    "--sfreq",
    type=int,
    default=200,
    help="Sets the frequency to downsample the data to.",
)
args = parser.parse_args()

if args.log:
    logging.basicConfig(
        filename=args.save + "prepare_data.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )


def prepare_data(sub, source_path, save_path, down_freq=200, user=args.user):
    data_path = os.path.join(
        source_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.dattype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    csv_path = os.path.join(
        source_path,
        "dataman/useraccess/processed/",
        user,
        "standard_data.csv",
    )
    df = pd.read_csv(csv_path)
    for file in os.listdir(os.path.join(data_path, sub)):
        if file.endswith("fif"):
            fif_file = os.path.join(data_path, sub, file)
            break

    data = mne.io.read_raw_fif(fif_file).resample(sfreq=down_freq)
    bads = data.info["bads"]
    if bads == []:
        channels = []
        logging.info(sub)
        logging.info(data.info)
        channels.append(deepcopy(data).pick_types(meg="mag")[:][0])
        channels.append(deepcopy(data).pick_types(meg="planar1")[:][0])
        channels.append(data.pick_types(meg="planar2")[:][0])
        data = np.array(channels)
        logging.info(data.shape)
        save_path = os.path.join(save_path, f"downsampled_{args.sfreq}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub = sub.split("-")[1]
        np.save(os.path.join(save_path, f"{args.dattype}_{sub}.npy"), data)
        row = df[df["CCID"] == sub].values.tolist()[0]
        return row
    else:
        logging.info(f"{sub} was dropped because of bad channels {bads}")
    return


if __name__ == "__main__":
    """Main function."""
    data_path = os.path.join(
        args.path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.dattype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    a = Parallel(n_jobs=6)(
        delayed(prepare_data)(sub, args.path, args.save, args.sfreq)
        for sub in os.listdir(data_path)
    )
    dataframe = pd.DataFrame(a, columns=["sub", "age", "sex", "hand", "Coil", "MT_TR"])
    dataframe.to_csv(os.path.join(args.save, "partifipants_info.csv")
