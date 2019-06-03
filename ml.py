import mne
import scipy as sp

# from mne.viz import plot_topomap
# from mne.io import read_raw_fif
# import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit as SSS, cross_val_score

# from sklearn.ensemble import AdaBoostClassifier as ADABoost, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

# from path import Path as path
from params import DATA_PATH, SAVE_PATH, CHAN_DF, SUB_DF

typ = 2  # MAG
if typ == 2:
    N_ELEC = 102
else:
    N_ELEC = 204


def extract_bands(data):
    f = np.asarray([float(i / 3) for i in range(data.shape[-1])])
    # data = data[:, :, (f >= 8) * (f <= 12)].mean(axis=2)
    data = [
        data[:, :, (f >= .5) * (f <= 4)].mean(axis=-1)[..., None],
        data[:, :, (f >= 8) * (f <= 12)].mean(axis=-1)[..., None],
        data[:, :, (f >= 12) * (f <= 30)].mean(axis=-1)[..., None],
        data[:, :, (f >= 30) * (f <= 120)].mean(axis=-1)[..., None],
    ]
    data = np.concatenate(data, axis=2)
    return data


def load_freq_data(dataframe, path=DATA_PATH):
    shape = (0, N_ELEC, 4)
    X = np.array([]).reshape(shape)
    y = []
    i = 0
    mag_index = [i for i in range(typ, 306, 3)]
    for row in dataframe[: int(len(dataframe))].iterrows():
        sub = row[1]["Observations"]
        # lab = row[1]["gender_code"]
        sub_data = sp.stats.zscore(
            extract_bands(np.load(path + "{}_rest_psd.npy".format(sub)))
        )
        # sub_data = np.load(path + "{}_rest_psd.npy".format(sub))
        # sub_data = sub_data - sub_data.mean(axis=0)[None, :]
        try:
            sub_data2 = sp.stats.zscore(
                extract_bands(np.load(path + "{}_task_psd.npy".format(sub)))
            )
            # sub_data2 = np.load(path + "{}_task_psd.npy".format(sub))
            # sub_data2 = sub_data2 - sub_data2.mean(axis=0)[None, :]
        except:
            sub_data2 = np.array([]).reshape(shape)
        # y += [lab] * len(sub_data)
        y += [0] * len(sub_data)
        y += [1] * len(sub_data2)
        sub_data = sub_data[:, mag_index]
        sub_data2 = sub_data2[:, mag_index]
        sub_data = np.concatenate((sub_data, sub_data2), axis=0)
        # y += [i] * len(sub_data)
        X = np.concatenate((X, sub_data), axis=0)
        i += 1
    return np.array(X), np.array(y)


if __name__ == "__main__":

    # from sklearn.datasets import load_breast_cancer

    # data = load_breast_cancer()
    # X, y = data["data"][:100], data["target"][:100]
    # idx = np.random.RandomState(0).permutation(range(len(X)))
    # X, y = X[idx], y[idx]
    # X_test_og, y_test = X[60:], y[60:]
    # X_og, y = X[:60], y[:60]

    data_df = SUB_DF[["Observations", "gender_code"]]
    idx = np.random.RandomState(0).permutation(range(len(data_df)))
    data_df = data_df.iloc[idx]
    n_train_subs = int(0.6 * len(data_df))
    train_df = data_df[:n_train_subs]
    test_df = data_df[n_train_subs:]
    X_og, y = load_freq_data(train_df)
    X_test_og, y_test = load_freq_data(test_df)

    idx = np.random.RandomState(0).permutation(range(len(X_og)))
    X_og = X_og[idx]
    y = y[idx]
    print(X_og[[0, 33, 166]], y[[0, 33, 166]])

    idx = np.random.RandomState(0).permutation(range(len(X_test_og)))
    X_test_og = X_test_og[idx]
    y_test = y_test[idx]

    cv = SSS(5)

    all_scores = []
    # for C in [0.1, 1.0, 10.0, 100.0]:
    param_distributions = {
        "C": sp.stats.expon(scale=10),
        "gamma": sp.stats.expon(scale=0.1),
    }
    for elec in range(N_ELEC):
        # X = X_og
        # X_test = X_test_og
        X = X_og[:, elec]
        X_test = X_test_og[:, elec]
        # if len(X.shape) < 2:
        #     X = X[..., None]
        #     X_test = X_test[..., None]

        randsearch = RandomizedSearchCV(
            SVC(),
            param_distributions=param_distributions,
            n_iter=150,
            cv=cv,
            random_state=420,
            n_jobs=-1,
        )
        randsearch.fit(X, y)
        param = randsearch.best_params_
        train = randsearch.best_score_
        clf = SVC(**randsearch.best_params_)
        test = np.mean(cross_val_score(clf, X_test, y_test, cv=cv, n_jobs=-1))

        print(train, test, param)
        np.save(SAVE_PATH + f"SVM_train_scores_elec{elec}", train)
        np.save(SAVE_PATH + f"SVM_test_scores_elec{elec}", test)
        np.save(SAVE_PATH + f"SVM_params_elec{elec}", param)

    # all_scores = np.asarray(all_scores)
    # final = all_scores[np.argmax(np.max(all_scores, axis=0))]
    # DATA_PATH = path("/home/arthur/data/raw_camcan/data/data/")
    # sub = DATA_PATH.dirs()[0]
    # dtype = "task"
    # file_path = sub / f"{dtype}/{dtype}_raw.fif"
    # a = read_raw_fif(file_path, preload=True).pick_types(meg=True)
    # ch_names = a.info["ch_names"]
    # mask_params = dict(
    # marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
    # )

    # CHANNEL_NAMES = [ch_names[i] for i in range(typ, 306, 3)]
    # pval_corr = np.asarray([pval_corr[i] for i in range(typ, 306, 3)])

    # tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
    # tt_mask[pval_corr < 0.01] = True

    # plot_topomap(
    # final,
    # a.pick_types(meg="mag").info,
    # res=128,
    # cmap="Spectral_r",
    # vmin=.5,
    # show=False,
    # names=CHANNEL_NAMES,
    # show_names=False,
    # mask=tt_mask,
    # mask_params=mask_params,
    # contours=0,
    # )

    # mne.viz.tight_layout()
    # plt.show()
