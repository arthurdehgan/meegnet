import unittest
import torch
import numpy as np
from meegnet.utils import stratified_sampling
from meegnet.utils import extract_bands
from meegnet.utils import compute_psd
from meegnet.utils import string_to_int


class TestStringToInt(unittest.TestCase):
    def test_string_to_int_with_unique_labels(self):
        # Test with unique labels inferred from the array
        array = ["cat", "dog", "cat", "bird"]
        int_array, labels = string_to_int(array)
        self.assertTrue(np.array_equal(int_array, [1, 2, 1, 0]))
        self.assertEqual(labels.tolist(), ["bird", "cat", "dog"])

    def test_string_to_int_with_predefined_labels(self):
        # Test with predefined target labels
        array = ["cat", "dog", "cat", "bird"]
        target_labels = ["dog", "cat", "bird"]
        int_array, labels = string_to_int(array, target_labels=target_labels)
        self.assertTrue(np.array_equal(int_array, [1, 0, 1, 2]))
        self.assertEqual(labels, target_labels)

    def test_string_to_int_empty_array(self):
        # Test with an empty array
        array = []
        with self.assertRaises(ValueError) as context:
            string_to_int(array)
        self.assertEqual(str(context.exception), "Input array is empty.")

    def test_string_to_int_duplicate_target_labels(self):
        # Test with duplicate target labels
        array = ["cat", "dog", "cat", "bird"]
        target_labels = ["dog", "cat", "cat"]
        with self.assertRaises(ValueError) as context:
            string_to_int(array, target_labels=target_labels)
        self.assertEqual(str(context.exception), "`target_labels` must contain unique values.")

    def test_string_to_int_with_numpy_array(self):
        # Test with a numpy array as input
        array = np.array(["cat", "dog", "cat", "bird"])
        int_array, labels = string_to_int(array)
        self.assertTrue(np.array_equal(int_array, [1, 2, 1, 0]))
        self.assertEqual(labels.tolist(), ["bird", "cat", "dog"])


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
            from meegnet.utils import extract_bands


class TestExtractBands(unittest.TestCase):
    def test_extract_bands_with_default_bins(self):
        # Create mock data with shape (n_samples, n_channels, n_bins)
        data = np.random.rand(5, 3, 240)  # 5 samples, 3 channels, 240 bins
        result = extract_bands(data)

        # Check the output shape
        self.assertEqual(result.shape, (5, 3, 7))  # 7 frequency bands

    def test_extract_bands_with_custom_bins(self):
        # Create mock data with shape (n_channels, n_bins)
        data = np.random.rand(3, 240)  # 3 channels, 240 bins
        custom_bins = np.linspace(0, 120, 240)  # Custom frequency bins
        result = extract_bands(data, f=custom_bins)

        # Check the output shape
        self.assertEqual(result.shape, (3, 7))  # 7 frequency bands


class TestComputePSD(unittest.TestCase):
    def test_compute_psd_multitaper(self):
        # Create mock data with shape (n_channels, n_samples)
        data = np.random.rand(3, 500)  # 3 channels, 500 samples
        fs = 250  # Sampling frequency
        result = compute_psd(data, fs, option="multitaper")

        # Check the output shape
        self.assertEqual(result.shape, (3, 7))  # 7 frequency bands

    def test_compute_psd_welch(self):
        # Create mock data with shape (n_channels, n_samples)
        data = np.random.rand(3, 500)  # 3 channels, 500 samples
        fs = 250  # Sampling frequency
        result = compute_psd(data, fs, option="welch")

        # Check the output shape
        self.assertEqual(result.shape, (3, 7))  # 7 frequency bands

    def test_compute_psd_invalid_option(self):
        # Create mock data with shape (n_channels, n_samples)
        data = np.random.rand(3, 500)  # 3 channels, 500 samples
        fs = 250  # Sampling frequency

        # Check that an invalid option raises an error
        with self.assertRaises(Exception):
            compute_psd(data, fs, option="invalid_option")


if __name__ == "__main__":
    unittest.main()
