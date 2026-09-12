import os
import numpy as np
import torch

from meegnet.dataloaders import ContinuousDataset


SAVE_PATH = '/home/arthur/camcan/rest'
SFREQ = 200
WINDOW = 0.8
OVERLAP = 0.0
TRAIN_SIZE = 0.7
SEED = 42
TARGET_COL = 'label'


def n_windows(shape):
    step = int(WINDOW * SFREQ * (1 - OVERLAP))
    start = int(10 * SFREQ)
    n = 0
    for i in range(start, shape[-1], step):
        if i + step <= shape[-1]:
            n += 1
    return n


def split_train_valid(n_subjects, seed):
    total = TRAIN_SIZE + (0.9 - TRAIN_SIZE)
    sizes = (TRAIN_SIZE / total, (0.9 - TRAIN_SIZE) / total)
    generator = torch.Generator().manual_seed(seed)
    s1, s2 = torch.utils.data.random_split(
        np.arange(n_subjects), [round(n_subjects * sizes[0]), n_subjects - round(n_subjects * sizes[0])], generator
    )
    n_train = round(n_subjects * sizes[0])
    splits = torch.utils.data.random_split(np.arange(n_subjects), [n_train, n_subjects - n_train], generator)
    return list(splits[0]), list(splits[1])


def count_files(sub_list):
    counts = {}
    missing = []
    for sub in sub_list:
        fp = os.path.join(SAVE_PATH, 'downsampled_200', f'{sub}_rest.npy')
        if not os.path.exists(fp):
            missing.append(sub)
            continue
        shp = np.load(fp, mmap_mode='r').shape
        counts[sub] = n_windows(shp)
    return counts, missing


def run(max_subj):
    ds = ContinuousDataset(
        window=WINDOW,
        overlap=OVERLAP,
        sfreq=SFREQ,
        n_subjects=max_subj,
        split_sizes=TRAIN_SIZE,
        sensortype=None,
        lso=True,
        random_state=SEED,
        target_col=TARGET_COL,
    )
    df = ds.preload(SAVE_PATH, target_col=TARGET_COL)

    test_subs = ds.test_subjects
    pool_subs = df['sub'].tolist()

    labels = {}
    for _, row in df.iterrows():
        labels[row['sub']] = row[TARGET_COL]
    test_labels = {}
    for _, row in ds.test_dataframe.iterrows():
        test_labels[row['sub']] = row[TARGET_COL]

    train_idx, valid_idx = split_train_valid(len(pool_subs), SEED)
    train_subs = [pool_subs[i] for i in train_idx]
    valid_subs = [pool_subs[i] for i in valid_idx]

    train_counts, train_miss = count_files(train_subs)
    valid_counts, valid_miss = count_files(valid_subs)
    test_counts, test_miss = count_files(test_subs)

    print(f'\n=== max-subj={max_subj} ===')
    print(f'total pool subjects: {len(pool_subs)}  test holdout: {len(test_subs)}')
    print(f'train: {len(train_subs)}  valid: {len(valid_subs)}')
    print(f'missing npy files: train {len(train_miss)} valid {len(valid_miss)} test {len(test_miss)}')

    classes = sorted(set(list(labels.values()) + list(test_labels.values())))
    print(f'classes: {classes} (n={len(classes)})')

    hdr = f'{"class":<10} {"train(subj)":<12} {"valid(subj)":<12} {"test(subj)":<12}'
    print(hdr)
    for c in classes:
        tr = sum(1 for s in train_subs if labels[s] == c)
        va = sum(1 for s in valid_subs if labels[s] == c)
        te = sum(1 for s in test_subs if test_labels[s] == c)
        print(f'{str(c):<10} {tr:<12} {va:<12} {te:<12}')

    print('\ntrial-weighted (windows from downsampled_200):')
    for c in classes:
        tr = sum(train_counts.get(s, 0) for s in train_subs if labels[s] == c)
        va = sum(valid_counts.get(s, 0) for s in valid_subs if labels[s] == c)
        te = sum(test_counts.get(s, 0) for s in test_subs if test_labels[s] == c)
        print(f'{str(c):<10} {tr:<12} {va:<12} {te:<12}')

    print(f'valid subjects: {valid_subs}')
    print(f'train subjects: {train_subs}')


if __name__ == '__main__':
    for m in [25, 50, 100, 450, 1000]:
        run(m)