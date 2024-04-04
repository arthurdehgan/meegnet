"""Load parameters for all scripts."""

# mis apres examination des donnees, on ignore les 10 premieres secondes (donnees samplee a 200Hz)
OFFSET = 2000
# mis en fonction de Van puten qui utilise des trials de 2s samples a 200Hz (utilise les donnees ds200)
NBINS = 129


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
    "Theta": (4, 8),
    "Alpha": (8, 15),
    "Beta": (15, 30),
    "Gamma1": (30, 60),
    "Gamma2": (60, 90),
    "Gamma3": (90, 120),
}

SF = 1000  # sampling frequency
N_ELEC = 306  # number electrodes

# Pour calculs PSD
WINDOW = 3000  # fenetre
OVERLAP = 0.25  # overlap (entre 0 et 1)

# Commenting this whole part as it is no longer necessary to have those here.
# keeping it in case we want to go back.

# SUB_DF = pd.read_csv("./clean_participant_new.csv", index_col=0)
# SUBJECT_LIST = SUB_DF["participant_id"].tolist()
#
# CHAN_DF = pd.read_csv("./channel_names.csv", index_col=0)
# CHANNEL_NAMES = CHAN_DF["ch_name"].tolist()
#
# AGES = SUB_DF["age"].tolist()
#
# LABELS = {}
# LABELS["sex"] = SUB_DF["sex"].tolist()
# LABELS["age"] = [group1(i) for i in AGES]
# LABELS["age_all"] = AGES
# LABELS["subject"] = list(range(len(SUBJECT_LIST)))
