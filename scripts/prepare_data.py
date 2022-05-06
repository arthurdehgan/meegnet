"""This script is intenden to work for the camcan MEG dataset with maxfilter and transform to default common space (mf_transdef) data.

The Camcan dataset is not open access and you need to send a request on the websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed through the argparser.

TODO:
    add a prompt on the amount of disk space and time required

"""
import os
import logging
import mne
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from camcan.parsing import parser

# "Arthur_Dehgan_1160"
parser.add_argument(
    "--user",
    type=str,
    help="name of the camcan user folder for loading csv data. Usually of the form forename_name_1234",
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
        filename=args.save_path + "prepare_data.log",
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


def prepare_data(
    sub,
    source_path,
    save_path,
    camcan_user,
    s_freq=200,
    dattype="rest",
    epoched=False,
):
    if dattype == "rest":
        assert (
            not epoched
        ), "Can't load epoched resting state data as there are no events for it"
    row = None
    data_path = os.path.join(
        source_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{dattype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    csv_path = os.path.join(
        source_path,
        "dataman/useraccess/processed/",
        camcan_user,
        "standard_data.csv",
    )
    df = pd.read_csv(csv_path)
    for file in os.listdir(os.path.join(data_path, sub)):
        if file.endswith("fif"):
            fif_file = os.path.join(data_path, sub, file)
            break

    raw = mne.io.read_raw_fif(fif_file).resample(sfreq=s_freq)
    bads = raw.info["bads"]
    if bads == []:
        channels = []
        logging.info(sub)
        logging.info(raw.info)
        if epoched and dattype != "rest":  # dattype in ("passive","smt"):
            events = mne.find_events(raw)
            event_dict = {6: "auditory1", 7: "auditory2", 8: "auditory3", 9: "visual"}
            data = mne.Epochs(raw, events, tmin=-0.15, tmax=0.65, preload=True)
            labels = [event_dict[event] for event in events[:, -1]]
            filename = f"{dattype}_{sub}_epoched.npy"
        else:
            data = raw
            filename = f"{dattype}_{sub}.npy"
        channels.append(data.get_data(picks="mag"))
        channels.append(data.get_data(picks="planar1"))
        channels.append(data.get_data(picks="planar2"))
        data = np.array(channels)
        if dattype == "passive":
            data = data.swapaxes(0, 1)
        logging.info(data.shape)
        save_path = os.path.join(save_path, f"downsampled_{args.sfreq}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub = sub.split("-")[1]
        np.save(os.path.join(save_path, filename), data)
        row = df[df["CCID"] == sub].values.tolist()[0]
        if dattype in ("passive", "smt"):
            row.append(labels)
        print(row)
        print(data.shape)
    else:
        logging.info(f"{sub} was dropped because of bad channels {bads}")
    return row


if __name__ == "__main__":
    data_path = os.path.join(
        args.data_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.dattype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    a = Parallel(n_jobs=6)(
        delayed(prepare_data)(
            sub,
            args.data_path,
            args.save_path,
            camcan_user=args.user,
            s_freq=args.sfreq,
            epoched=args.epoched,
            dattype=args.dattype,
        )
        for sub in os.listdir(data_path)
    )
    columns = ["sub", "age", "sex", "hand", "Coil", "MT_TR"]
    if args.dattype != "rest":
        columns.append("event_labels")
    dataframe = pd.DataFrame(a, columns=columns)
    dataframe.to_csv(
        os.path.join(args.save_path, f"participants_info_{args.dattype}.csv")
    )
