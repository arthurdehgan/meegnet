"""This script is intended to work for the camcan MEG dataset with maxfilter
and transform to default common space (mf_transdef) data in BIDS format.

The Camcan dataset is not open access and you need to send a request on the
websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed
through the argparser.

example on how to run the script:
python prepare_data.py --config="config.toml" --data-path="/home/user/data/camcan/" --save-path="/home/user/data"

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
import toml


def bad_subj_found(sub: str, info: str, message: str, df_path: str):
    """
    Open, edits and saves the dataframe (bad_subj_df) with provided error info.

    Parameters
    ----------
    sub : str
        The subject identifier.
    info : str
        Information about the subject.
    message : str
        The log message to be logged.
    df_path : str
        The path to the CSV file where the DataFrame is stored.

    Returns
    -------
    None
    """
    logging.info(message)
    row = [sub, info]
    with open(df_path, "r") as f:
        df = pd.read_csv(f, index_col=0)
    df = df._append({key: val for key, val in zip(df.columns, row)}, ignore_index=True)
    with open(df_path, "w") as f:
        df.to_csv(f)


def prepare_data(
    sub_folder: str,
    data_path: str,
    save_path: str,
    s_freq: int = 200,
    datatype: str = "rest",
    epoched: bool = False,
):
    """prepare_data.

    Parameters
    ----------
    sub_folder : str
        the subject folder".
    data_path : str
        the path to the folder containing the Cam-CAN dataset
        (with cc700 folder in it) as copied from the CC servers.
    save_path : str
        the path where all the data will be saved in subfolder "downsampled_{s_freq}"
        and the csv file will be saved.
    s_freq : int
        the frequency to downsample the data to.
    datatype : str
        the type of data to prepare. Must be in ['rest', 'passive', 'smt'].
    epoched : bool
        epoch the data based on events found in it or not.
        Epoching the data will apply baseline correction (mode: mean).

    Returns
    -------
    row : list
        a list of values from the camcan metadata csv file.
        The row will be added to the participants_info_{datatype}.csv file.
    """
    if datatype == "rest":
        assert (
            not epoched
        ), "Can't load epoched resting state data as there are no events for it"
    row = None

    data_filepath = os.path.join(
        data_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{datatype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )
    user = os.listdir(os.path.join(data_path, "dataman/useraccess/processed/"))[0]
    source_csv_path = os.path.join(
        data_path,
        f"dataman/useraccess/processed/{user}/standard_data.csv",
    )
    with open(source_csv_path, "r") as f:
        df = pd.read_csv(f)

    fif_file = ""
    file_list = os.listdir(os.path.join(data_filepath, sub_folder))
    while not fif_file.endswith(".fif"):
        fif_file = os.path.join(data_filepath, sub_folder, file_list.pop())

    sub = sub_folder.split("-")[1]
    if epoched:
        filename = f"{datatype}_{sub}_epoched.npy"
    else:
        filename = f"{datatype}_{sub}.npy"
    out_path = os.path.join(save_path, f"downsampled_{args.sfreq}")
    out_file = os.path.join(out_path, filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    bad_csv_path = os.path.join(args.save_path, f"bad_participants_info_{args.datatype}.csv")
    if os.path.exists(bad_csv_path):
        with open(bad_csv_path, "r") as f:
            bad_subs_df = pd.read_csv(f, index_col=0)
    else:
        bad_subs_df = pd.DataFrame({}, columns=["sub", "error"])
    with open(bad_csv_path, "w") as f:
        bad_subs_df.to_csv(f)

    good_csv_path = os.path.join(args.save_path, f"participants_info_{args.datatype}.csv")
    columns = ["sub", "age", "sex", "hand", "Coil", "MT_TR"]
    if args.datatype != "rest":
        columns.append("event_labels")
    if os.path.exists(good_csv_path):
        with open(good_csv_path, "r") as f:
            good_subs_df = pd.read_csv(f, index_col=0)
    else:
        good_subs_df = pd.DataFrame({}, columns=columns)
    with open(good_csv_path, "w") as f:
        good_subs_df.to_csv(f)

    if sub in good_subs_df["sub"].tolist() + bad_subs_df["sub"].tolist():
        return

    raw = mne.io.read_raw_fif(fif_file)
    bads = raw.info["bads"]
    if bads == []:
        channels = []
        logging.info(sub_folder)
        logging.info(raw.info)
        if epoched and datatype in ("passive", "smt"):  # datatype != "rest"
            try:
                events = mne.find_events(raw)
            except ValueError as e:
                bad_subj_found(
                    sub=sub,
                    info="wrong event timings",
                    message=f"{sub} could not be used because of {e}",
                    df_path=bad_csv_path,
                )
                return
            unique_events = set(events[:, -1])
            if unique_events == {6, 7, 8, 9}:
                event_dict = {
                    6: "auditory1",
                    7: "auditory2",
                    8: "auditory3",
                    9: "visual",
                }
                labels = [event_dict[event] for event in events[:, -1]]
            else:
                bad_subj_found(
                    sub=sub,
                    info=f"wrong event found: {unique_events}",
                    message=f"a different event has been found in {sub}: {unique_events}",
                    df_path=bad_csv_path,
                )
                return
            data = mne.Epochs(raw, events, tmin=-0.15, tmax=0.65, preload=True)
        else:
            data = raw

        data = data.resample(sfreq=s_freq)
        channels.append(data.get_data(picks="mag"))
        channels.append(data.get_data(picks="planar1"))
        channels.append(data.get_data(picks="planar2"))
        data = np.array(channels)
        if datatype == "passive":
            data = data.swapaxes(0, 1)
        logging.info(data.shape)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(out_file, data)
        row = df[df["CCID"] == sub].values.tolist()[0]
        if datatype in ("passive", "smt"):
            row.append(labels)
    else:
        bad_subj_found(
            sub=sub,
            info="bad channels",
            message=f"{sub} was dropped because of bad channels {bads}",
            df_path=bad_csv_path,
        )
        return

    good_subs_df = good_subs_df._append(
        {key: val for key, val in zip(good_subs_df.columns, row)}, ignore_index=True
    )
    with open(good_csv_path, "w") as f:
        good_subs_df.to_csv(f)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    toml_string = toml.dumps(args_dict)
    with open(args.config, "w") as toml_file:
        toml.dump(args_dict, toml_file)

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

    data_filepath = os.path.join(
        args.data_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.datatype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )

    Parallel(n_jobs=-1)(
        delayed(prepare_data)(
            sub,
            args.data_path,
            args.save_path,
            s_freq=args.sfreq,
            epoched=args.epoched,
            datatype=args.datatype,
        )
        for sub in os.listdir(data_filepath)
    )
