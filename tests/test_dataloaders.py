import unittest
import torch
import numpy as np
from torch.utils.data import Dataset
from meegnet.dataloaders import (
    _InfiniteSampler,
    InfiniteDataLoader,
    EpochedDataset,
    ContinuousDataset,
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


if __name__ == "__main__":
    unittest.main()
