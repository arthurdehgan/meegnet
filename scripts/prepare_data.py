"""This script is intended to work for the camcan MEG dataset with maxfilter
and transform to default common space (mf_transdef) data in BIDS format.

The Camcan dataset is not open access and you need to send a request on the
websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed
through the argparser.

example on how to run the script:
python prepare_data.py --data-path="/home/user/data/camcan/" --save-path="/home/user/data" --user="Firstname_Name_1160" --dattype="passive" --epoched

TODO:
    add a prompt on the amount of disk space and time required
    add option of not doing epoched with passive and smt data

"""
import os
import logging
import mne
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from meegnet.parsing import parser

parser.add_argument(
    "--user",
    type=str,
    required=True,
    help="name of the camcan user folder for loading csv data.\
    Usually of the form Firstname_Name_ID",
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
    sub_folder: str,
    source_path: str,
    save_path: str,
    camcan_user: str,
    s_freq: int = 200,
    dattype: str = "rest",
    epoched: bool = False,
):
    """prepare_data.

    Parameters
    ----------
    sub_folder : str
        the subject folder in BIDS format of the form: "sub-[SUB_ID]".
    source_path : str
        the path to the folder containing the Cam-CAN dataset
        (with cc700 folder in it) as copied from the CC servers.
    save_path : str
        the path where all the data will be saved in subfolder "downsampled_{s_freq}"
        and the csv file will be saved.
    camcan_user : str
        your camcan user in order to find metadata about the subjects. Should be firstname_name_ID.
    s_freq : int
        the frequency to downsample the data to.
    dattype : str
        the type of data to prepare. Must be in ['rest', 'passive', 'smt'].
    epoched : bool
        epoch the data based on events found in it or not.
        Epoching the data will apply baseline correction (mode: mean).

    Returns
    -------
    row : list
        a list of values from the camcan metadata csv file.
        The row will be added to the participants_info_{dattype}.csv file.
    """
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
    source_csv_path = os.path.join(
        source_path,
        "dataman/useraccess/processed/",
        camcan_user,
        "standard_data.csv",
    )
    df = pd.read_csv(source_csv_path)

    fif_file = ""
    file_list = os.listdir(os.path.join(data_path, sub_folder))
    while not fif_file.endswith(".fif"):
        fif_file = os.path.join(data_path, sub_folder, file_list.pop())

    sub = sub_folder.split("-")[1]
    if epoched:
        filename = f"{dattype}_{sub}_epoched.npy"
    else:
        filename = f"{dattype}_{sub}.npy"
    out_path = os.path.join(save_path, f"downsampled_{args.sfreq}")
    out_file = os.path.join(out_path, filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    bad_csv_path = os.path.join(
        args.save_path, f"bad_participants_info_{args.dattype}.csv"
    )
    columns = ["sub", "error"]
    bad_subs_df = (
        pd.read_csv(bad_csv_path, index_col=0)
        if os.path.exists(bad_csv_path)
        else pd.DataFrame({}, columns=columns)
    )
    good_csv_path = os.path.join(
        args.save_path, f"participants_info_{args.dattype}.csv"
    )
    columns = ["sub", "age", "sex", "hand", "Coil", "MT_TR"]
    if args.dattype != "rest":
        columns.append("event_labels")
    good_subs_df = (
        pd.read_csv(good_csv_path, index_col=0)
        if os.path.exists(good_csv_path)
        else pd.DataFrame([{}], columns=columns)
    )

    if sub in good_subs_df["sub"].tolist() + bad_subs_df["sub"].tolist():
        return

    raw = mne.io.read_raw_fif(fif_file)
    bads = raw.info["bads"]
    if bads == []:
        channels = []
        logging.info(sub_folder)
        logging.info(raw.info)
        if epoched and dattype in ("passive", "smt"):  # dattype != "rest"
            try:
                events = mne.find_events(raw)
            except ValueError as e:
                logging.info(f"{sub} could not be used because of {e}")
                row = [sub, "wrong event timings"]
                bad_subs_df = bad_subs_df._append(
                    {key: val for key, val in zip(columns, row)}, ignore_index=True
                )
                bad_subs_df.to_csv(bad_csv_path)
                return
            if set(events[:, -1]) == {6, 7, 8, 9}:
                event_dict = {
                    6: "auditory1",
                    7: "auditory2",
                    8: "auditory3",
                    9: "visual",
                }
                labels = [event_dict[event] for event in events[:, -1]]
            else:
                logging.info(f"a different event has been found in {sub}")
                row = [sub, "wrong event found"]
                good_subs_df = good_subs_df._append(
                    {key: val for key, val in zip(columns, row)}, ignore_index=True
                )
                good_subs_df.to_csv(good_csv_path)
                return
            data = mne.Epochs(raw, events, tmin=-0.15, tmax=0.65, preload=True)
        else:
            data = raw

        data = data.resample(sfreq=s_freq)
        channels.append(data.get_data(picks="mag"))
        channels.append(data.get_data(picks="planar1"))
        channels.append(data.get_data(picks="planar2"))
        data = np.array(channels)
        if dattype == "passive":
            data = data.swapaxes(0, 1)
        logging.info(data.shape)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(out_file, data)
        row = df[df["CCID"] == sub].values.tolist()[0]
        if dattype in ("passive", "smt"):
            row.append(labels)
    else:
        logging.info(f"{sub} was dropped because of bad channels {bads}")
        row = [sub, "bad channels"]
        bad_subs_df = bad_subs_df._append(
            {key: val for key, val in zip(columns, row)}, ignore_index=True
        )
        bad_subs_df.to_csv(bad_csv_path)
        return

    good_subs_df = good_subs_df._append(
        {key: val for key, val in zip(columns, row)}, ignore_index=True
    )
    good_subs_df.to_csv(good_csv_path)
    return


if __name__ == "__main__":
    data_path = os.path.join(
        args.data_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.dattype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    table = Parallel(n_jobs=1)(
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
