"""Load parameters for all scripts."""
import pandas as pd
from path import Path as path

DATA_PATH = path('/home/arthur/Documents/camcan/')
SAVE_PATH = path('/home/arthur/Documents/camcan/')
SUB_INFO_PATH = path('/home/arthur/Documents/camcan/clean_camcan_participant_data.csv')

FREQ_DICT = {'Delta': (1, 4),
             'Theta': (4, 8),
             'Alpha': (8, 15),
             'Beta': (15, 30),
             'Gamma1': (30, 60),
             'Gamma2': (60, 90)}

SF = 1000  # sampling frequency
N_ELEC = 306  # number electrodes

# Pour calculs PSD
WINDOW = 3000  # fenetre
OVERLAP = 0.25  # overlap (entre 0 et 1)

SUBJECT_LIST = pd.read_csv(SUB_INFO_PATH)['Observations'].tolist()
