'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from joblib import Parallel, delayed
import numpy as np
from numpy.random import permutation
from scipy.io import savemat
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils import load_psd_cc_subjects
from params import SAVE_PATH, N_ELEC, WINDOW, OVERLAP, FREQ_DICT,\
                   SUB_INFO_PATH, path

N_PERMUTATIONS = 1000
SAVE_PATH = SAVE_PATH / 'psd'
PERM = True
PREFIX = 'perm' if PERM else 'classif'


def permutation_score(cv, clf, X, y):
    clf = clone(clf)
    perm_set = permutation(len(y))
    y_perm = y[perm_set]
    score = cross_val_score(cv=cv,
                            estimator=clf,
                            X=X, y=y_perm,
                            n_jobs=1).mean()
    return score


def classification(elec):

    results_file_path = SAVE_PATH / 'results' /\
                        '{}_PSD_SEX_{}_{}_{:.2f}.mat'.format(
                            PREFIX, elec, WINDOW, OVERLAP)

    if not path(results_file_path).isfile():
        data, labels = load_psd_cc_subjects(SAVE_PATH, SUB_INFO_PATH,
                                            WINDOW, OVERLAP)
        f, psd = np.array(data[:, 0].tolist()), data[:, 1]
        psd = np.array(list(psd))
        f = f.reshape(f.shape[0], f.shape[-1])[0]

        for key in FREQ_DICT:
            print(elec, key)
            fmin, fmax = FREQ_DICT[key]
            data = np.mean(psd[:, elec, (f >= fmin) * (f <= fmax)],
                           axis=-1).reshape(-1, 1)
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            clf = LDA()
            scores = []
            pvalue = 0
            good_score = cross_val_score(cv=cv,
                                         estimator=clf,
                                         X=data, y=labels,
                                         n_jobs=-1).mean()
            if PERM:
                scores = Parallel(n_jobs=-1)(
                    delayed(permutation_score)(cv=cv,
                                               clf=LDA(),
                                               X=data,
                                               y=labels)
                    for _ in range(N_PERMUTATIONS))

                for score in scores:
                    if good_score <= score:
                        pvalue += 1/N_PERMUTATIONS
                data = {'score': good_score,
                        'pscore': scores,
                        'pvalue': pvalue}
                print('{} : {:.2f} significatif a p={:.4f}'.format(
                    key, good_score, pvalue))
                savemat(results_file_path, data)
            else:
                data = {'score': good_score}
                print('{} : {:.2f}'.format(
                    key, good_score))
                savemat(results_file_path, data)


if __name__ == '__main__':
    for elec in range(N_ELEC):
        classification(elec)
