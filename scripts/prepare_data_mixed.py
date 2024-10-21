"""This script is intended to work for the camcan MEG dataset with maxfilter
and transform to default common space (mf_transdef) data in BIDS format.

The Camcan dataset is not open access and you need to send a request on the
websitde in order to get access (https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/).

This script assumes a copy of the cc700 and dataman folders to a data path parsed
through the argparser.

example on how to run the script:
python prepare_data_mixed.py --config="config.ini" --raw-path="/home/user/data/camcan/" --save-path="/home/user/data"

This script (prepare_data_mixed) will look for passive oddball data as well as resting-state data in order to
create a mixed Dataset.
"""

import os
import logging
import mne
import warnings
import pandas as pd
import numpy as np
from meegnet.parsing import parser, save_config

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

LOG = logging.getLogger("meegnet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)


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
    LOG.info(message)
    row = [sub, info]
    with open(df_path, "r") as f:
        df = pd.read_csv(f, index_col=0)
    df = df._append({key: val for key, val in zip(df.columns, row)}, ignore_index=True)
    with open(df_path, "w") as f:
        df.to_csv(f)


def process_data(data, sfreq, datatype, n_epochs=None):
    # TODO HARD-CODED = bad !!
    length = 400
    ignore = 30  # to ignore the first 30s of resting-state signals
    # TODO end ugly hard-coded values

    if data is not None:
        data = data.resample(sfreq=sfreq)
        data = np.array(
            [
                data.get_data(picks="mag"),
                data.get_data(picks="planar1"),
                data.get_data(picks="planar2"),
            ]
        )
        # Ugly but, this means that we will select n_epochs trials only when data is not passive
        if datatype == "passive":
            data = data.swapaxes(0, 1)
        elif n_epochs is not None:
            # Generating fake epochs from continuous data like resting-state:
            rest_indexes = range(ignore * length, data.shape[-1] - length, length)
            random_indexes = np.random.choice(rest_indexes, n_epochs, replace=False)
            data = np.array([data[:, :, i : i + length] for i in random_indexes])
    return data


def load_and_process(
    sub_folder: str,
    sfreq: int,
    data_path: str,
    save_path: str,
    label: str,
):
    bad_csv_path = os.path.join(save_path, f"bad_participants_info.csv")
    if os.path.exists(bad_csv_path):
        with open(bad_csv_path, "r") as f:
            bad_subs_df = pd.read_csv(f, index_col=0)
    else:
        bad_subs_df = pd.DataFrame({}, columns=["sub", "error"])
        with open(bad_csv_path, "w") as f:
            bad_subs_df.to_csv(f)

    good_csv_path = os.path.join(save_path, f"participants_info.csv")
    columns = ["sub", "age", "stuff", "hand", "Coil", "MT_TR", "label"]
    if os.path.exists(good_csv_path):
        with open(good_csv_path, "r") as f:
            good_subs_df = pd.read_csv(f, index_col=0)
    else:
        good_subs_df = pd.DataFrame({}, columns=columns)
        with open(good_csv_path, "w") as f:
            good_subs_df.to_csv(f)

    sub = sub_folder.split("-")[1]
    filename = f"{sub}_epoched_mixed.npy"
    out_path = os.path.join(save_path, f"downsampled_{args.sfreq}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    filepath = os.path.join(out_path, filename)
    if sub in bad_subs_df["sub"].tolist():
        return None, None
    elif sub in good_subs_df["sub"].tolist() and os.path.exists(filepath):
        return None, None

    datas = []
    # Important : we start with passive because we will match number of epochs from passive to rest
    for dataset in ("passive", "rest"):
        row = None
        data_filepath = os.path.join(
            data_path,
            "cc700/meg/pipeline/release005/BIDSsep/",
            f"derivatives_{dataset}",
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
        dir = os.path.join(data_filepath, sub_folder)
        if os.path.exists(dir):
            file_list = os.listdir(dir)
            while not fif_file.endswith(".fif"):
                fif_file = os.path.join(data_filepath, sub_folder, file_list.pop())
        else:
            bad_subj_found(
                sub=sub,
                info="couldnt find all files for subject",
                message=f"{dir} could not be found",
                df_path=bad_csv_path,
            )
            return None, None

        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        bads = raw.info["bads"]
        if bads == []:
            if dataset in ("passive", "smt"):  # dataset != "rest"
                try:
                    events = mne.find_events(raw)
                except ValueError as e:
                    bad_subj_found(
                        sub=sub,
                        info="wrong event timings",
                        message=f"{sub} could not be used because of {e}",
                        df_path=bad_csv_path,
                    )
                    return None, None
                unique_events = set(events[:, -1])
                if unique_events == {6, 7, 8, 9}:
                    event_dict = {
                        6: "auditory",
                        7: "auditory",
                        8: "auditory",
                        9: "visual",
                    }
                    labels = np.array([event_dict[event] for event in events[:, -1]])
                else:
                    bad_subj_found(
                        sub=sub,
                        info=f"wrong event found: {unique_events}",
                        message=f"a different event has been found in {sub}: {unique_events}",
                        df_path=bad_csv_path,
                    )
                    return None, None
                # mixing up the timing of the stimulus
                tmin = (np.random.random_sample() / 5) + 0.01
                data = mne.Epochs(raw, events, tmin=-tmin, tmax=0.8 - tmin, preload=True)
                if len(events) != len(data):
                    bad_subj_found(
                        sub=sub,
                        info=f"Size missmatch between events and data.",
                        message=f"Size missmatch between data ({len(data)}) and events ({len(events)}).",
                        df_path=bad_csv_path,
                    )
                    return None, None
                lab_idx = np.where(labels == label)[0]
                labels = list(labels[lab_idx])
                data = data[lab_idx]
            else:
                data = raw

            # doing this, we match the number of resting-state pseudo-epochs to the number of passive epochs:
            if dataset in ("passive", "smt"):
                n_epochs = len(labels)
            # send n_epochs to the function in order to select trials when using resting-state data
            datas.append(process_data(data, sfreq, dataset, n_epochs))
        else:
            bad_subj_found(
                sub=sub,
                info="bad channels",
                message=f"{sub} was dropped because of bad channels {bads}",
                df_path=bad_csv_path,
            )
            return None, None

    row = df[df["CCID"] == sub].values.tolist()[0]
    rest_labels = ["rest"] * n_epochs
    row.append(labels + rest_labels)
    if not sub in good_subs_df["sub"].tolist():
        good_subs_df = good_subs_df._append(
            {key: val for key, val in zip(good_subs_df.columns, row)},
            ignore_index=True,
        )
        with open(good_csv_path, "w") as f:
            good_subs_df.to_csv(f)
    return np.concatenate(datas, axis=0), filepath


if __name__ == "__main__":
    args = parser.parse_args()
    save_config(vars(args), args.config)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ######################
    ### LOGGING CONFIG ###
    ######################

    if args.log:
        log_file = os.path.join(args.save_path, "prepare_data.log")
        logging.basicConfig(filename=log_file, filemode="a")
        LOG.info(f"Starting logging in {log_file}")

    ########################
    ### ASSERTION CHECKS ###
    ########################

    assert args.raw_path is not None, "The --raw-path parameter has to be set."
    assert os.path.exists(
        args.raw_path
    ), f'The --raw-path "{args.raw_path}" parameter doesnt exist.'
    check_path = os.listdir(args.raw_path)
    assert (
        "cc700" in check_path and "dataman" in check_path
    ), "The --raw-path must contain the cc700 and dataman folders in order for this script to work properly."

    #######################
    ### FIXING UP PATHS ###
    #######################

    # This path will be used to get the subject list but in fine, we will be using passive and rest data
    data_filepath = os.path.join(
        args.raw_path,
        "cc700/meg/pipeline/release005/BIDSsep/",
        f"derivatives_{args.datatype}",
        "aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/",
    )

    ##################################
    ### LOADING AND PREPARING DATA ###
    ##################################

    for label in ("auditory", "visual"):
        np.random.seed(args.seed)
        save_path = os.path.join(args.save_path, f"mixed_{label}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for sub in os.listdir(data_filepath):
            data, filepath = load_and_process(sub, args.sfreq, args.raw_path, save_path, label)
            if data is None:
                continue

            np.save(filepath, data)
