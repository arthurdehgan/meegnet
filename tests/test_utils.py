import unittest
import torch
import numpy as np
from meegnet.utils import stratified_sampling


class TestStratifiedSampling(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.data = torch.randn(330, 10)  # 100 samples, 10 features each
        self.targets = torch.tensor(
            [0] * 50 + [1] * 60 + [0] * 50 + [1] * 60 + [0] * 50 + [1] * 60
        )  # 50 samples of class 0, 60 of class 1 for each group
        self.groups = torch.tensor([0] * 110 + [1] * 110 + [2] * 110)  # 3 groups

    def test_stratified_sampling_with_groups(self):
        # Test stratified sampling for a specific group
        sampled_indices = stratified_sampling(
            self.data, self.targets, n_samples=20, subject=0, groups=self.groups
        )
        sampled_targets = self.targets[sampled_indices]

        # Check that the number of samples is correct
        self.assertEqual(len(sampled_indices), 20)

        # Check that the class distribution is stratified
        unique, counts = torch.unique(sampled_targets, return_counts=True)
        self.assertTrue(
            torch.allclose(counts.float() / counts.sum(), torch.tensor([0.5, 0.5]))
        )

    def test_stratified_sampling_without_groups(self):
        # Test stratified sampling without specifying groups
        sampled_indices = stratified_sampling(self.data, self.targets, n_samples=20)
        sampled_targets = self.targets[sampled_indices]

        # Check that the number of samples is correct
        self.assertEqual(len(sampled_indices), 20)

        # Check that the class distribution is stratified
        unique, counts = torch.unique(sampled_targets, return_counts=True)
        self.assertTrue(
            torch.allclose(counts.float() / counts.sum(), torch.tensor([0.5, 0.5]))
        )

    def test_invalid_inputs(self):
        # Test invalid inputs
        with self.assertRaises(AssertionError):
            stratified_sampling(self.data, self.targets, n_samples=20, subject=0)
            stratified_sampling(self.data, self.targets, n_samples=20, groups=self.groups)


if __name__ == "__main__":
    unittest.main()
