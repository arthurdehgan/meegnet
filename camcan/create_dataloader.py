"""This file is used once, takes the data path and files to create a csv containing the position of each possible trial from the dataset"""

import numpy as np
import pandas as pd
from params import DATA_PATH, SUB_DF, OFFSET, TIME_TRIAL_LENGTH


def create_sub_info_timestamps(dataframe, dpath=DATA_PATH):
    k = 0
    data = []
    for row in dataframe.iterrows():
        print(f"loading subject {k+1}/{len(dataframe)}", end="\r")
        sub, lab = row[1]["participant_id"], row[1]["sex"]
        try:
            # On prend les donnees ds200 (downsampled 200Hz), voir params.py
            # On prend uniquement l'electrode 0 vu qu' on va juste compter, ce sera plus rapide.
            sub_data = np.load(dpath + f"{sub}_ICA_transdef_mfds200.npy")[0]
        except:
            # si les donnees sont corrompues, on affiche le sujet et on passe au suivant.
            print("\nThere was a problem loading subject ", sub)
            continue

        for i in range(OFFSET, sub_data.shape[-1], TIME_TRIAL_LENGTH):
            if i + TIME_TRIAL_LENGTH < sub_data.shape[-1]:
                dat = (sub, lab, i, i + TIME_TRIAL_LENGTH)
                data.append(dat)
        k += 1
    return pd.DataFrame(data)


if __name__ == "__main__":
    data_df = SUB_DF[["participant_id", "sex"]]
    infos = create_sub_info_timestamps(data_df)
    infos.to_csv("trials_df.csv")
