import os
import argparse
import numpy as np
from mne.io import read_raw_fif

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_type",
    choices=["task", "rest", "passive"],
    default="rest",
    help="The type of data to use for classification",
)
parser.add_argument(
    "--clean_type",
    choices=["mf", "transdef_mf", "raw"],
    default="transdef_mf",
    help="The type of preprocessing step to use for classification",
)
args = parser.parse_args()


def fif2npy(in_path, out_path):
    data = read_raw_fif(in_path, verbose=False, preload=True).pick_types(meg=True)[:][0]
    np.save(out_path, data)


if __name__ == "__main__":
    DATA_PATH = f"/home/arthur/data/camcan/data/meg_{args.data_type}_{args.clean_type}/"
    SAVE_PATH = (
        f"/home/arthur/data/camcan/npdata/meg_{args.data_type}_{args.clean_type}/"
    )
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    file_list = os.listdir(DATA_PATH)
    for sub in file_list:
        if sub.startswith("sub"):
            file_name = (
                f"{sub}/{sub}_ses-{args.data_type}_task-{args.data_type}_proc-sss.fif"
            )
            new_file_name = f"{sub}_{args.data_type}_{args.clean_type}"
            fif2npy(DATA_PATH + file_name, SAVE_PATH + new_file_name)
