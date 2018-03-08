'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
from joblib import Parallel, delayed
import numpy as np
from numpy.random import permutation
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups
from params import SAVE_PATH, N_ELEC, LABEL_PATH, path,\
                   WINDOW, OVERLAP, STATE_LIST  # , FREQ_DICT

N_PERMUTATIONS = 1000
SAVE_PATH = SAVE_PATH / 'riemann'
WINDOW = WINDOW
# FREQ_DICT = FREQ_DICT
OVERLAP = OVERLAP


def classification(state, elec):
    global OVERLAP, WINDOW, SAVE_PATH, N_ELEC, N_PERMUTATIONS # , FREQ_DICT
    labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
    labels, groups = create_groups(labels)

    #for key in FREQ_DICT:
    key = "Beta"

    results_file_path = SAVE_PATH / 'results' /\
                        'perm_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                            state, key, elec, WINDOW, OVERLAP)
    if not path(results_file_path).isfile():
        # print('\nloading PSD for {} frequencies'.format(key))
        data_file_path = SAVE_PATH /\
                'PSD_{}_{}_{}_{}_{}.mat'.format(
                    state, key, elec, WINDOW, OVERLAP)
        if path(data_file_path).isfile():
            temp = loadmat(data_file_path)['data'].ravel()
            data = temp[0].ravel()
            for submat in temp[1:]:
                data = np.concatenate((submat.ravel(), data))
            data = data.reshape(len(data), 1)
            del temp
        else:
            print(path(data_file_path).name + ' Not found')
            print('please run "computePSD.py" and\
                  "group_PSD_per_subjects.py" before\
                  running this script')

        # print('classification...')
        sl2go = StratifiedLeave2GroupsOut()
        clf = LDA()
        scores = []
        pvalue = 0
        good_score = cross_val_score(cv=sl2go,
                                     estimator=clf,
                                     X=data, y=labels,
                                     groups=groups).mean()
        for _ in range(N_PERMUTATIONS):
            clf = LDA()
            perm_set = permutation(len(labels))
            labels_perm = labels[perm_set]
            groups_perm = groups[perm_set]
            scores.append(cross_val_score(cv=sl2go,
                                          estimator=clf,
                                          X=data, y=labels_perm,
                                          groups=groups_perm,
                                          n_jobs=-1).mean())
        for score in scores:
            if good_score <= score:
                pvalue += 1/N_PERMUTATIONS

        data = {'score': good_score,
                'pscore': scores,
                'pvalue': pvalue}
        print('{} : {:.2f} significatif a p={:.4f}'.format(
        key, good_score, pvalue))

        savemat(results_file_path, data)


if __name__ == '__main__':
    T0 = time()

    Parallel(n_jobs=-1)(delayed(classification)(state,
                                                elec)
                        for state in STATE_LIST for elec in range(N_ELEC))
    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
