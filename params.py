"""Load parameters for all scripts."""
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
paths = dict(config["DEFAULT"])
SAVE_PATH = paths["save"]
DATA_PATH = paths["data"]
RAW_PATH = paths["raw"]
subject_list = paths["subject_list"]
channels_list = paths["channels_list"]


def group1(age):
    if 18 <= age <= 28:
        return 0
    elif 28 <= age <= 38:
        return 1
    elif 38 <= age <= 48:
        return 2
    elif 48 <= age <= 58:
        return 3
    elif 58 <= age <= 68:
        return 4
    elif 68 <= age <= 78:
        return 5
    elif 78 <= age <= 88:
        return 6


FREQ_DICT = {
    "Delta": (1, 4),
    "Theta": (5, 8),
    "Alpha": (9, 15),
    "Beta": (16, 30),
    "Gamma1": (31, 60),
    "Gamma2": (61, 90),
}

SF = 1000  # sampling frequency
N_ELEC = 306  # number electrodes

# Pour calculs PSD
WINDOW = 3000  # fenetre
OVERLAP = 0.25  # overlap (entre 0 et 1)

SUB_DF = pd.read_csv(subject_list, index_col=0)
SUBJECT_LIST = SUB_DF["participant_id"].tolist()

CHAN_DF = pd.read_csv(channels_list, index_col=0)
CHANNEL_NAMES = CHAN_DF["ch_name"].tolist()

AGES = SUB_DF["age"].tolist()

LABELS = {}
LABELS["gender"] = SUB_DF["gender"].tolist()
LABELS["age"] = [group1(i) for i in AGES]
LABELS["age_all"] = AGES
LABELS["subject"] = list(range(len(SUBJECT_LIST)))
