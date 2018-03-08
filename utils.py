"""Functions used to compute and analyse EEG/MEG data with pyriemann."""
from scipy.io import loadmat, savemat
from time import time
import numpy as np
import pandas as pd
from path import Path as path
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from sklearn.utils import indexable
from sklearn.utils.validation import check_array, _num_samples
from itertools import combinations
from pyriemann.classification import MDM, TSclassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from scipy.signal import welch


def load_psd_cc_subjects(PSD_PATH, sub_info_path, window, overlap):
    df = pd.read_csv(sub_info_path)
    sub_list = df['Observations'].tolist()
    labels = list(df['gender_code'] - 1)
    psd = []
    for sub in sub_list:
        file_path = path(PSD_PATH) / 'PSD_{}_{}_{}'.format(
            sub, window, overlap)
        try:
            psd.append(loadmat(file_path)['data'].ravel())
        except IOError:
            print(sub, 'Not Found')
    return np.array(psd), np.array(labels)


def gen_dif_index(groups, test_group):
    """Generate an index to identify which group is in test_group."""
    a = list(set(test_group))
    index = []
    for e in a:
        index.append(np.where(groups == e)[0][0])
    return index


def is_strat(y, groups=None, test_group=None):
    """Tell if a label vector is stratified."""
    if groups is not None:
        indx = gen_dif_index(groups, test_group)
        check = len(indx)
    else:
        check = len(y)
    labels = []
    for i in indx:
        labels.append(y[i])
    labels = np.asarray(labels)
    if len(np.where(labels == 1)[0]) == check/2:
        return True
    else:
        return False


def computePSD(signal, window, overlap, fmin, fmax, fs):
    """Compute PSD."""
    f, psd = welch(signal, fs=fs, window='hamming', nperseg=window,
                   noverlap=overlap, nfft=None)
    psd = np.mean(psd[(f >= fmin)*(f <= fmax)])
    return psd


def classif_choice(classif):
    """Select the classifier and parameters."""
    params = {}
    if classif == 'MDM':
        clf = MDM()
    elif classif == 'Logistic Regression':
        clf = TSclassifier()
    elif classif == 'SVM':
        svm = SVC()
        # params = {'clf__C': expon(scale=100), 'clf__gamma': expon(scale=.1),
        #           'clf__kernel': ['rbf'], 'clf__class_weight': ['balanced']}
        clf = TSclassifier(clf=svm)
    elif classif == 'LDA':
        lda = LDA()
        clf = TSclassifier(clf=lda)
    elif classif == 'Random Forest':
        rf = RF()
        clf = TSclassifier(clf=rf)
    return clf, params


def set_parameters(clf, classif, params):
    """Set the right parameters."""
    if classif == 'SVM':
        clf.set_params(clf__C=params['clf__C'])
        clf.set_params(clf__gamma=params['clf__gamma'])
        clf.set_params(clf__kernel=str(params['clf__kernel']))
        clf.set_params(clf__class_weight=params['clf__class_weight'])
    elif classif == 'Random Forest':
        rf = RF()
        clf = TSclassifier(clf=rf)
    return clf


def prepare_params(save_path, classif, X, y, sleep_state, clf_choice, params, cross_val, groups=None):
    """Load or Optimize best params for given clf, cv and params."""
    file_path = save_path / '%s_%s_best_parameters.mat' % (classif, sleep_state)
    if not file_path.isfile():
        t2 = time()
        print('Hyperparameter optimization...')
        RS = RandomizedSearchCV(estimator=clf_choice,
                                param_distributions=params,
                                n_iter=50,
                                n_jobs=-1,
                                cv=cross_val).fit(X, y, groups=groups)
        best_params = RS.best_params_
        savemat(file_path, best_params)
        print('Optimization done in %s' % elapsed_time(time(), t2))
    else:
        best_params = loadmat(file_path)
    params = {}
    for param in best_params.keys():
        if not param.startswith('__'):
            if len(best_params[param].shape) == 1:
                params[param] = best_params[param][0]
            elif len(best_params[param].shape) == 2:
                params[param] = best_params[param][0, 0]
    return params


def create_groups(y):
    """Generate groups from labels of shape (subject x labels)."""
    k = 0
    groups = []
    for i in range(len(y)):
        for j in range(y[i].shape[1]):
            groups.append(k)
        k += 1
    groups = np.asarray(groups).ravel()
    y = np.concatenate(y[range(len(y))], axis=1).ravel()
    return y, groups


def elapsed_time(t0, t1):
    """Time lapsed between t0 and t1.

    Returns the time (from time.time()) between t0 and t1 in a
    more readable fashion.

    Parameters
    ----------
    t0: float,
        time.time() initial measure of time
        (eg. at the begining of the script)
    t1: float,
        time.time() time at the end of the script
        or the execution of a function.

    """
    lapsed = abs(t1-t0)
    m, h, j = 60, 3600, 24*3600
    nbj = lapsed // j
    nbh = (lapsed - j * nbj) // h
    nbm = (lapsed - j * nbj - h * nbh) // m
    nbs = lapsed - j * nbj - h * nbh - m * nbm
    if lapsed > m:
        if lapsed > h:
            if lapsed > j:
                Time = "%ij, %ih:%im:%is" % (nbj, nbh, nbm, nbs)
            else:
                Time = "%ih:%im:%is" % (nbh, nbm, nbs)
        else:
            Time = "%im:%is" % (nbm, nbs)
    else:
        Time = "%is" % nbs
    return Time


def load_samples(data_path, subjectNumber, sleep_state, elec=None):
    """Load the samples of a subject for a sleepstate."""
    tempFileName = data_path / "%s_s%i.mat" % (sleep_state, subjectNumber)
    if elec == None:
        dataset = loadmat(tempFileName)[sleep_state][:19]
    else:
        dataset = loadmat(tempFileName)[sleep_state][elec]
    dataset = dataset.swapaxes(0, 2)
    dataset = dataset.swapaxes(1, 2)
    return dataset


def import_data(data_path, sleep_state, subject_list,
                label_path=None, full_trial=False):
    """Transform the data and generate labels.

    Takes the original files and put them in a matrix of
    shape (Trials x 19 x 30000)

    """
    X = []
    print("Loading data...")
    for i in range(len(subject_list)):
        # Loading of the trials of the selected sleepstate
        dataset = load_samples(data_path, subject_list[i], sleep_state)

        if full_trial:
            dataset = np.concatenate((dataset[range(len(dataset))]), axis=1)  # use if you want full trial
            dataset = dataset.reshape(1, dataset.shape[0], dataset.shape[1])
        X.append(dataset)

        del dataset

    if label_path is not None:
        y = loadmat(label_path / sleep_state + '_labels.mat')['y'].ravel()
    return X, np.asarray(y)


def is_signif(pvalue, p=0.05):
    """Tell if condition with classifier is significative.

    Returns a boolean : True if the condition is significativeat given p
    """
    answer = False
    if pvalue <= p:
            answer = True
    return answer


class BaseStratCrossValidator(with_metaclass(ABCMeta)):
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def __init__(self):
        # We need this for the build_repr to work properly in py2.7
        # see #6304
        pass

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            if y[test_index[0]] != y[test_index[-1]]:
                yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""

    def __repr__(self):
        return _build_repr(self)


class StratLeavePGroupsOut(BaseStratCrossValidator):
    """Leave P Group(s) Out cross-validator.

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.
    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    The difference between LeavePGroupsOut and LeaveOneGroupOut is that
    the former builds the test sets with all the samples assigned to
    ``p`` different values of the groups while the latter uses samples
    all assigned the same groups.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_groups : int
        Number of groups (``p``) to leave out in the test split.
    Examples
    --------
    >>> from sklearn.model_selection import LeavePGroupsOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1])
    >>> groups = np.array([1, 2, 3])
    >>> lpgo = LeavePGroupsOut(n_groups=2)
    >>> lpgo.get_n_splits(X, y, groups)
    3
    >>> print(lpgo)
    LeavePGroupsOut(n_groups=2)
    >>> for train_index, test_index in lpgo.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [2] TEST: [0 1]
    [[5 6]] [[1 2]
     [3 4]] [1] [1 2]
    TRAIN: [1] TEST: [0 2]
    [[3 4]] [[1 2]
     [5 6]] [2] [1 1]
    TRAIN: [0] TEST: [1 2]
    [[1 2]] [[3 4]
     [5 6]] [1] [2 1]
    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The groups parameter should not be None")
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups = np.unique(groups)
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1))
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            test_index = np.zeros(_num_samples(X), dtype=np.bool)
            for l in unique_groups[np.array(indices)]:
                test_index[groups == l] = True
            yield test_index

    def get_n_splits(self, X, y, groups):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The groups parameter should not be None")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        X, y, groups = indexable(X, y, groups)
        return int(comb(len(np.unique(groups)), self.n_groups, exact=True))


def StratifiedLeave2GroupsOut():
    return StratLeavePGroupsOut(n_groups=2)

def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))
