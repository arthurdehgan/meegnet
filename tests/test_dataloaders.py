import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from meegnet.dataloaders import (
    _InfiniteSampler,
    InfiniteDataLoader,
    EpochedDataset,
    ContinuousDataset,
    _split_holdout,
)


class MockDataset(Dataset):
    """A mock dataset for testing purposes."""

    def __init__(self, size):
        self.data = torch.arange(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestInfiniteSampler(unittest.TestCase):
    def test_infinite_sampler(self):
        dataset = MockDataset(10)
        sampler = _InfiniteSampler(torch.utils.data.RandomSampler(dataset))
        iterator = iter(sampler)

        # Check that the sampler produces an infinite stream
        for _ in range(20):  # Check the first 20 samples
            self.assertIn(next(iterator), range(len(dataset)))


class TestInfiniteDataLoader(unittest.TestCase):
    def test_infinite_dataloader(self):
        dataset = MockDataset(10)
        dataloader = InfiniteDataLoader(dataset, batch_size=2, num_workers=0)

        # Check that the dataloader produces batches of the correct size
        iterator = iter(dataloader)
        for _ in range(6):  # Check the first 6 batches, 6th batch is created because infinite
            batch = next(iterator)
            self.assertEqual(len(batch), 2)


class TestEpochedDataset(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 10, 500)  # 100 samples, 10 channels, 500 time points
        self.targets = np.random.randint(0, 2, size=100)  # Binary targets
        self.groups = np.random.randint(0, 5, size=100)  # 5 groups

    def test_epoched_dataset_initialization(self):
        dataset = EpochedDataset(sfreq=500, n_subjects=5, n_samples=50)
        dataset.set_data(self.data, self.targets, self.groups)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(len(dataset.subject_list), 5)

    def test_epoched_dataset_split(self):
        dataset = EpochedDataset(sfreq=500, n_subjects=5, n_samples=50)
        dataset.set_data(self.data, self.targets, self.groups)

        train_idx, valid_idx, test_idx = dataset.split_data(0.6, 0.2, 0.2)
        self.assertEqual(len(train_idx) + len(valid_idx) + len(test_idx), len(dataset))


class TestContinuousDataset(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 10, 500)  # 100 samples, 10 channels, 500 time points
        self.targets = np.random.randint(0, 2, size=100)  # Binary targets
        self.groups = np.random.randint(0, 5, size=100)  # 5 groups

    def test_continuous_dataset_initialization(self):
        dataset = ContinuousDataset(
            window=2, overlap=0.5, sfreq=500, n_subjects=5, n_samples=50
        )
        dataset.set_data(self.data, self.targets, self.groups)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(len(dataset.subject_list), 5)

    def test_continuous_dataset_windowing(self):
        dataset = ContinuousDataset(
            window=2, overlap=0.5, sfreq=500, n_subjects=5, n_samples=50
        )
        dataset.set_data(self.data, self.targets, self.groups)

        # Check that the windowing works correctly
        self.assertEqual(dataset.window, 2)
        self.assertEqual(dataset.overlap, 0.5)


class TestSplitHoldout(unittest.TestCase):
    def _make_df(self, n=30):
        return pd.DataFrame({"sub": [f"sub{i:03d}" for i in range(n)], "label": ["aud"] * n})

    def test_holdout_deterministic(self):
        df = self._make_df()
        test1, pool1 = _split_holdout(df, 0.1, 42)
        test2, pool2 = _split_holdout(df, 0.1, 42)
        pd.testing.assert_frame_equal(test1, test2)
        pd.testing.assert_frame_equal(pool1, pool2)

    def test_holdout_fraction(self):
        df = self._make_df(30)
        test, pool = _split_holdout(df, 0.1, 42)
        self.assertEqual(len(test), 3)
        self.assertEqual(len(pool), 27)

    def test_holdout_disjoint_and_covers_pool(self):
        df = self._make_df(50)
        test, pool = _split_holdout(df, 0.1, 42)
        self.assertEqual(set(test["sub"]) & set(pool["sub"]), set())
        self.assertEqual(len(test) + len(pool), len(df))

    def test_holdout_zero_is_noop(self):
        df = self._make_df(10)
        test, pool = _split_holdout(df, 0, 42)
        self.assertEqual(len(test), 0)
        self.assertEqual(len(pool), len(df))

    def test_holdout_stable_regardless_of_max_subj(self):
        # Holdout is selected from the total pool before the max_subj cut, so the
        # pool cut to any size never overlaps the test subjects.
        df = self._make_df(100)
        test, pool = _split_holdout(df, 0.1, 42)
        cut_pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)[:25]
        self.assertEqual(set(test["sub"]) & set(cut_pool["sub"]), set())


class TestSplitDataCV(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(100, 10, 500)  # 100 samples, 10 channels, 500 time points
        self.targets = np.random.randint(0, 2, size=100)  # Binary targets
        self.groups = np.random.randint(0, 5, size=100)  # 5 groups

    def test_cv_fold_is_valid_and_train_is_rest(self):
        dataset = EpochedDataset(sfreq=500, n_samples=50, lso=True)
        dataset.set_data(self.data, self.targets, self.groups)
        for fold in range(5):
            train_idx, valid_idx, test = dataset.split_data_cv(fold, n_folds=5)
            self.assertIsNone(test)
            train_subjects = {dataset.groups[i].item() for i in train_idx}
            valid_subjects = {dataset.groups[i].item() for i in valid_idx}
            self.assertEqual(len(train_subjects & valid_subjects), 0)
            self.assertEqual(train_subjects | valid_subjects, set(range(5)))

    def test_default_split_has_no_test(self):
        dataset = EpochedDataset(sfreq=500, n_samples=50, lso=True)
        dataset.set_data(self.data, self.targets, self.groups)
        train_idx, valid_idx, test = dataset.split_data()
        self.assertIsNone(test)
        self.assertEqual(len(train_idx) + len(valid_idx), len(dataset))

    def test_cv_folds_stratified_by_label(self):
        # 6 subjects, each with 10 trials: 5 visual + 5 auditory (eventclf-like)
        data = np.random.rand(60, 10, 500)
        targets = np.array([0, 1] * 30)
        groups = np.repeat(np.arange(6), 10)
        dataset = EpochedDataset(sfreq=500, lso=True)
        dataset.set_data(data, targets, groups)
        seen_valid_subjects = set()
        for fold in range(3):
            train_idx, valid_idx, test = dataset.split_data_cv(fold, n_folds=3)
            self.assertIsNone(test)
            valid_subjects = {dataset.groups[i].item() for i in valid_idx}
            self.assertEqual(len(valid_subjects), 2)  # whole subjects per fold
            self.assertEqual(len(set(train_idx) & set(valid_idx)), 0)
            self.assertEqual(len(train_idx) + len(valid_idx), len(dataset))
            seen_valid_subjects |= valid_subjects
        self.assertEqual(seen_valid_subjects, set(range(6)))

    def test_cv_fallback_on_singleton_classes(self):
        # subject-classification-like labels: each subject is its own class with 1 trial
        data = np.random.rand(20, 10, 500)
        targets = np.arange(20)
        groups = np.arange(20)
        dataset = EpochedDataset(sfreq=500, lso=True)
        dataset.set_data(data, targets, groups)
        train_idx, valid_idx, test = dataset.split_data_cv(0, n_folds=5)
        self.assertIsNone(test)
        train_subjects = {dataset.groups[i].item() for i in train_idx}
        valid_subjects = {dataset.groups[i].item() for i in valid_idx}
        self.assertEqual(len(train_subjects & valid_subjects), 0)
        self.assertEqual(train_subjects | valid_subjects, set(range(20)))

    def test_cv_folds_stratified_single_label_subjects(self):
        # sexclf-like: each subject holds one label across all its trials
        data = np.random.rand(40 * 10, 10, 500)
        targets = np.repeat(np.arange(40) % 2, 10)
        groups = np.repeat(np.arange(40), 10)
        dataset = EpochedDataset(sfreq=500, lso=True)
        dataset.set_data(data, targets, groups)
        seen_valid_subjects = set()
        for fold in range(5):
            train_idx, valid_idx, test = dataset.split_data_cv(fold, n_folds=5)
            self.assertIsNone(test)
            valid_labels = {dataset.targets[i].item() for i in valid_idx}
            self.assertEqual(valid_labels, {0, 1})  # both classes in every fold
            valid_subjects = {dataset.groups[i].item() for i in valid_idx}
            train_subjects = {dataset.groups[i].item() for i in train_idx}
            self.assertEqual(len(train_subjects & valid_subjects), 0)
            seen_valid_subjects |= valid_subjects
        self.assertEqual(seen_valid_subjects, set(range(40)))

    def test_cv_fallback_single_label_subjects_as_classes(self):
        # age-like degenerate: one class per subject, many trials each;
        # StratifiedGroupKFold cannot spread each class over 5 folds -> random subject folds
        data = np.random.rand(20 * 10, 10, 500)
        targets = np.repeat(np.arange(20), 10)
        groups = np.repeat(np.arange(20), 10)
        dataset = EpochedDataset(sfreq=500, lso=True)
        dataset.set_data(data, targets, groups)
        for fold in range(5):
            train_idx, valid_idx, test = dataset.split_data_cv(fold, n_folds=5)
            self.assertIsNone(test)
            valid_subjects = {dataset.groups[i].item() for i in valid_idx}
            train_subjects = {dataset.groups[i].item() for i in train_idx}
            self.assertEqual(len(train_subjects & valid_subjects), 0)
            self.assertEqual(train_subjects | valid_subjects, set(range(20)))
            self.assertEqual(len(valid_subjects), 4)  # whole subjects per fold (20/5)

    def test_testdataset_empty_without_holdout(self):
        dataset = EpochedDataset(sfreq=500, n_samples=50)
        dataset.set_data(self.data, self.targets, self.groups)
        self.assertEqual(len(dataset.testDataset()), 0)


class TestWithinSubjectHoldout(unittest.TestCase):
    """lso=False mode (subject classification): per-subject trial-level test set."""

    def _make(self, n_sub=20, n_trials=10):
        np.random.seed(0)
        data = np.random.rand(n_sub * n_trials, 10, 500)
        targets = np.random.randint(0, 2, n_sub * n_trials)
        groups = np.repeat(np.arange(n_sub), n_trials)
        dataset = EpochedDataset(sfreq=500, lso=False, split_sizes=(0.7, 0.2, 0.1))
        dataset.set_data(data, targets, groups)
        return dataset

    def test_every_subject_in_test_set(self):
        dataset = self._make()
        dataset._carve_trial_holdout()
        self.assertEqual(len(dataset.testDataset()), 20)  # 1 trial per subject (10 trials, 10%)
        self.assertEqual(len(set(dataset.test_groups.tolist())), 20)
        # every subject still in the training pool
        self.assertEqual(len(np.unique(dataset.groups.numpy())), 20)
        self.assertEqual(len(dataset), 200 - 20)

    def test_min_one_test_trial(self):
        dataset = self._make(n_sub=1, n_trials=12)
        dataset._carve_trial_holdout()
        self.assertEqual(len(dataset.testDataset()), 1)  # round(12 * 0.1) = 1

    def test_no_holdout_when_test_size_zero(self):
        dataset = EpochedDataset(sfreq=500, lso=False, split_sizes=(0.9, 0.1, 0))
        dataset.set_data(np.random.rand(100, 10, 500), np.random.randint(0, 2, 100), np.repeat(np.arange(10), 10))
        dataset._carve_trial_holdout()
        self.assertEqual(len(dataset.testDataset()), 0)
        self.assertEqual(len(dataset), 100)

    def test_test_trials_stable_with_seed(self):
        dataset_a = self._make()
        dataset_a._carve_trial_holdout()
        dataset_b = self._make()
        dataset_b._carve_trial_holdout()
        self.assertTrue(torch.equal(dataset_a.test_data, dataset_b.test_data))

    def test_default_split_covers_remaining(self):
        dataset = self._make()
        dataset._carve_trial_holdout()
        train_idx, valid_idx, test = dataset.split_data()
        self.assertIsNone(test)
        self.assertEqual(len(set(train_idx) & set(valid_idx)), 0)
        self.assertEqual(len(train_idx) + len(valid_idx), len(dataset))

    def test_cv_within_subject(self):
        dataset = self._make()
        dataset._carve_trial_holdout()
        for fold in range(5):
            train_idx, valid_idx, test = dataset.split_data_cv(fold, n_folds=5)
            self.assertIsNone(test)
            self.assertEqual(len(set(train_idx) & set(valid_idx)), 0)
            self.assertEqual(len(train_idx) + len(valid_idx), len(dataset))
            # each subject contributes trials to both sides
            train_subjects = set(dataset.groups[train_idx].tolist())
            valid_subjects = set(dataset.groups[valid_idx].tolist())
            self.assertEqual(train_subjects, valid_subjects)
            self.assertEqual(train_subjects, set(range(20)))


class TestStratifiedHoldout(unittest.TestCase):
    """Stratified holdout via sklearn StratifiedGroupKFold (lso=True, parseable labels)."""

    def test_scalar_labels_proportional(self):
        df = pd.DataFrame(
            {"sub": [f"s{i:03d}" for i in range(130)], "label": ["a"] * 100 + ["b"] * 30}
        )
        test, pool = _split_holdout(df, 0.1, 42, target_col="label")
        self.assertEqual(len(test), 13)
        counts = test["label"].value_counts()
        self.assertEqual(counts["a"], 10)
        self.assertEqual(counts["b"], 3)
        self.assertEqual(set(test["sub"]) & set(pool["sub"]), set())
        self.assertEqual(len(test) + len(pool), 130)

    def test_list_labels_keep_subjects_whole(self):
        # eventclf-style: per-trial label lists, subjects internally balanced
        rows = [{"sub": f"sub{i:03d}", "label": str(["visual", "auditory"] * 10)} for i in range(20)]
        df = pd.DataFrame(rows)
        test, pool = _split_holdout(df, 0.1, 42, target_col="label")
        self.assertEqual(len(test) + len(pool), 20)
        self.assertGreaterEqual(len(test), 1)
        self.assertEqual(set(test["sub"]) & set(pool["sub"]), set())
        # every held-out subject keeps all its trials and both classes
        for label in test["label"]:
            trials = label.strip("[]").split(", ")
            self.assertEqual(len(trials), 20)

    def test_fallback_on_singleton_classes(self):
        df = pd.DataFrame(
            {"sub": [f"s{i:03d}" for i in range(20)], "label": [f"s{i:03d}" for i in range(20)]}
        )
        test, pool = _split_holdout(df, 0.1, 42, target_col="label")
        # sklearn raises on singleton classes -> plain seeded slice fallback
        self.assertEqual(len(test), 2)
        self.assertEqual(len(test) + len(pool), 20)

    def test_fallback_on_unparseable_labels(self):
        df = pd.DataFrame({"sub": [f"s{i:03d}" for i in range(20)], "label": ["[]"] * 20})
        test, _ = _split_holdout(df, 0.1, 42, target_col="label")
        self.assertEqual(len(test), 2)

    def test_deterministic_and_stable_under_pool_cut(self):
        df = pd.DataFrame(
            {"sub": [f"s{i:03d}" for i in range(100)], "label": ["a"] * 70 + ["b"] * 30}
        )
        test_a, pool_a = _split_holdout(df, 0.1, 42, target_col="label")
        test_b, _ = _split_holdout(df, 0.1, 42, target_col="label")
        pd.testing.assert_frame_equal(test_a, test_b)
        cut_pool = pool_a.sample(frac=1, random_state=42).reset_index(drop=True)[:25]
        self.assertEqual(set(test_a["sub"]) & set(cut_pool["sub"]), set())

    def test_target_col_without_column_falls_back(self):
        df = pd.DataFrame({"sub": [f"s{i:03d}" for i in range(20)], "label": ["a"] * 20})
        test, _ = _split_holdout(df, 0.1, 42, target_col="sex")
        self.assertEqual(len(test), 2)  # 'sex' not in columns -> random slice


class TestSplitSizes(unittest.TestCase):
    def test_float_mapping_default(self):
        dataset = EpochedDataset(split_sizes=0.7)
        self.assertAlmostEqual(dataset.split_sizes[0] / 0.9, 7 / 9)
        self.assertAlmostEqual(dataset.split_sizes[1] / 0.9, 2 / 9)
        self.assertEqual(dataset.test_size, 0.1)

    def test_float_mapping_legacy(self):
        dataset = EpochedDataset(split_sizes=0.8)
        self.assertAlmostEqual(dataset.split_sizes[0], 0.8)
        self.assertAlmostEqual(dataset.split_sizes[1], 0.1)
        self.assertAlmostEqual(dataset.split_sizes[2], 0.1)

    def test_float_mapping_out_of_range(self):
        with self.assertRaises(ValueError):
            EpochedDataset(split_sizes=0.95)

    def test_default_subject_split_is_7_2_of_9(self):
        data = np.random.rand(90, 10, 500)  # 90 samples, 10 channels, 500 time points
        targets = np.random.randint(0, 2, size=90)  # Binary targets
        groups = np.repeat(np.arange(9), 10)  # 9 subjects, 10 trials each
        dataset = EpochedDataset(sfreq=500, lso=True)
        dataset.set_data(data, targets, groups)
        train_idx, valid_idx, test = dataset.split_data()
        train_subjects = {dataset.groups[i].item() for i in train_idx}
        valid_subjects = {dataset.groups[i].item() for i in valid_idx}
        self.assertIsNone(test)
        self.assertEqual(len(train_subjects), 7)
        self.assertEqual(len(valid_subjects), 2)
        self.assertEqual(train_subjects | valid_subjects, set(range(9)))


class TestStratifiedSubjectSplit(unittest.TestCase):
    def _dataset(self, n_subjects, n_trials=10, n_classes=2, random_state=0):
        data = np.random.rand(n_subjects * n_trials, 10, 500)
        # Subjects keep a single class label across all their trials.
        targets = np.repeat(np.arange(n_subjects) % n_classes, n_trials)
        groups = np.repeat(np.arange(n_subjects), n_trials)
        dataset = EpochedDataset(sfreq=500, lso=True, random_state=random_state)
        dataset.set_data(data, targets, groups)
        return dataset

    def test_stratified_keeps_all_classes_in_valid(self):
        dataset = self._dataset(n_subjects=30, n_classes=2)
        train_idx, valid_idx, test = dataset.split_data()
        train_labels = {dataset.targets[i].item() for i in train_idx}
        valid_labels = {dataset.targets[i].item() for i in valid_idx}
        self.assertIsNone(test)
        self.assertEqual(train_labels, {0, 1})
        self.assertEqual(valid_labels, {0, 1})

    def test_stratified_legacy_three_way_keeps_classes(self):
        dataset = self._dataset(n_subjects=30, n_classes=2)
        train_idx, valid_idx, test_idx = dataset.split_data(0.6, 0.2, 0.2)
        for idx in (train_idx, valid_idx, test_idx):
            self.assertEqual({dataset.targets[i].item() for i in idx}, {0, 1})
        self.assertEqual(len(train_idx) + len(valid_idx) + len(test_idx), len(dataset))

    def test_degenerate_pool_falls_back_to_random(self):
        # 7 classes over 25 subjects: the validation set (6 subjects) cannot hold 7
        # classes, so sklearn raises and the split must fall back without crashing.
        dataset = self._dataset(n_subjects=25, n_classes=7)
        train_idx, valid_idx, test = dataset.split_data()
        self.assertIsNone(test)
        self.assertEqual(len(train_idx) + len(valid_idx), len(dataset))
        train_subjects = {dataset.groups[i].item() for i in train_idx}
        valid_subjects = {dataset.groups[i].item() for i in valid_idx}
        self.assertEqual(len(train_subjects & valid_subjects), 0)
        self.assertEqual(train_subjects | valid_subjects, set(range(25)))

    def test_stratified_split_deterministic(self):
        ds1 = self._dataset(n_subjects=30, random_state=42)
        ds2 = self._dataset(n_subjects=30, random_state=42)
        train1, valid1, _ = ds1.split_data()
        train2, valid2, _ = ds2.split_data()
        self.assertEqual({ds1.groups[i].item() for i in train1}, {ds2.groups[i].item() for i in train2})
        self.assertEqual({ds1.groups[i].item() for i in valid1}, {ds2.groups[i].item() for i in valid2})


if __name__ == "__main__":
    unittest.main()
