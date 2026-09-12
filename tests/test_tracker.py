import os
import shutil
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat
from torch import nn, optim

from meegnet.network import Model, TrainingTracker


class TrackerTestMixin:
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()
		self.save_path = os.path.join(self.tmpdir, 'save')
		os.makedirs(self.save_path, exist_ok=True)

	def tearDown(self):
		shutil.rmtree(self.tmpdir)

	def read_mat(self, name):
		mat_path = os.path.join(self.save_path, name, name + '.mat')
		self.assertTrue(os.path.exists(mat_path), f'{mat_path} should exist')
		return loadmat(mat_path)

	def run_epochs(self, tracker, net, optimizer, losses, early_stop='loss'):
		"""Simulate epochs: validation loss improves at epoch 1 then degrades."""
		for epoch, (tloss, tacc, vloss, vacc) in enumerate(losses):
			tracker.start_epoch()
			tracker.stop_epoch()
			tracker.update(epoch, tloss, tacc, vloss, vacc, net, optimizer, early_stop=early_stop)


class TestTrainingTracker(TrackerTestMixin, unittest.TestCase):
	def setUp(self):
		super().setUp()
		self.name = 'test_tracker'
		# Model appends the model name to save_path before creating the tracker
		tracker_path = os.path.join(self.save_path, self.name)
		os.makedirs(tracker_path, exist_ok=True)
		self.tracker = TrainingTracker(tracker_path, self.name)
		self.net = nn.Linear(4, 2)
		self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

	def test_mat_written_every_epoch(self):
		losses = [
			(1.0, 0.5, 0.9, 0.6),
			(0.8, 0.6, 0.7, 0.7),  # improvement
			(0.7, 0.7, 0.8, 0.6),
			(0.6, 0.8, 0.9, 0.5),  # patience tail, no improvement
		]
		self.run_epochs(self.tracker, self.net, self.optimizer, losses)

		data = self.read_mat(self.name)
		self.assertEqual(len(np.atleast_1d(data['train_losses']).squeeze()), 4)
		self.assertEqual(len(np.atleast_1d(data['validation_losses']).squeeze()), 4)
		self.assertEqual(len(np.atleast_1d(data['train_accuracies']).squeeze()), 4)
		self.assertEqual(len(np.atleast_1d(data['validation_accuracies']).squeeze()), 4)
		self.assertTrue(np.isclose(data['train_losses'].squeeze()[-1], 0.6))
		self.assertTrue(np.isclose(data['validation_losses'].squeeze()[-1], 0.9))

	def test_mat_has_best_epoch_and_metrics(self):
		losses = [
			(1.0, 0.5, 0.9, 0.6),
			(0.8, 0.6, 0.7, 0.7),  # best epoch
			(0.7, 0.8, 0.8, 0.6),
		]
		self.run_epochs(self.tracker, self.net, self.optimizer, losses)

		data = self.read_mat(self.name)
		self.assertEqual(int(data['epoch'].squeeze()), 1)
		self.assertTrue(np.isclose(float(data['validation_loss'].squeeze()), 0.7))
		self.assertTrue(np.isclose(float(data['validation_accuracy'].squeeze()), 0.7))
		self.assertTrue(np.isclose(float(data['train_loss'].squeeze()), 0.8))
		self.assertTrue(np.isclose(float(data['train_accuracy'].squeeze()), 0.6))

	def test_checkpoint_only_on_improvement(self):
		pt_path = os.path.join(self.save_path, self.name, self.name + '.pt')
		losses = [
			(1.0, 0.5, 0.9, 0.6),
			(0.8, 0.6, 0.7, 0.7),  # improvement -> .pt written
			(0.7, 0.8, 0.8, 0.6),  # no improvement -> .pt untouched
		]
		self.run_epochs(self.tracker, self.net, self.optimizer, losses)
		self.assertTrue(os.path.exists(pt_path))
		mtime = os.path.getmtime(pt_path)
		self.run_epochs(self.tracker, self.net, self.optimizer, [(0.5, 0.9, 0.8, 0.5)])
		self.assertEqual(os.path.getmtime(pt_path), mtime)

	def test_early_stop_criterion_saved(self):
		losses = [(1.0, 0.5, 0.9, 0.6), (0.8, 0.6, 0.8, 0.7)]
		self.run_epochs(self.tracker, self.net, self.optimizer, losses, early_stop='accuracy')

		data = self.read_mat(self.name)
		self.assertEqual(str(data['early_stop'].squeeze()), 'accuracy')
		self.assertEqual(int(data['epoch'].squeeze()), 1)

	def test_load_roundtrip(self):
		losses = [
			(1.0, 0.5, 0.9, 0.6),
			(0.8, 0.6, 0.7, 0.7),
			(0.7, 0.8, 0.8, 0.6),
		]
		self.run_epochs(self.tracker, self.net, self.optimizer, losses)

		loaded = TrainingTracker(self.save_path, self.name)
		mat_path = os.path.join(self.save_path, self.name, self.name + '.mat')
		loaded.load(mat_path)

		self.assertEqual(len(loaded.progress['train_losses']), 3)
		self.assertEqual(loaded.best['epoch'], 1)
		self.assertEqual(loaded.early_stop, 'loss')
		self.assertTrue(np.isclose(loaded.progress['validation_losses'][-1], 0.8))


class TestModelTrainingSummary(TrackerTestMixin, unittest.TestCase):
	def setUp(self):
		super().setUp()
		self.name = 'test_model'

	def _make_model(self):
		return Model(
			self.name,
			'mlp',
			(2, 2, 2),
			2,
			save_path=self.save_path,
			learning_rate=0.01,
			optimizer=optim.Adam,
			criterion=nn.CrossEntropyLoss(),
			device='cpu',
			net_params={'linear': 4, 'hlayers': 2, 'dropout': 0.5},
		)

	def _make_dataset(self, n=16):
		class DummyDataset:
			def __init__(self, n):
				data = np.random.randn(n, 2, 2, 2).astype(np.float32)
				targets = np.arange(n) % 2
				groups = np.arange(n) % 4
				self.data = [np.squeeze(d) for d in data]
				self.targets = targets
				self.groups = groups

			def __len__(self):
				return len(self.data)

			def split_data(self):
				# deterministic split of indices
				idx = np.arange(len(self.data))
				return idx[: len(idx) // 2], idx[len(idx) // 2 :], []

			def torchDataset(self, index):
				import torch

				class DS(torch.utils.data.Dataset):
					def __init__(self, data, targets, groups, index):
						self.data = [data[i] for i in index]
						self.targets = [targets[i] for i in index]
						self.groups = [groups[i] for i in index]

					def __len__(self):
						return len(self.data)

					def __getitem__(self, i):
						return self.data[i], self.targets[i], self.groups[i]

				return DS(self.data, self.targets, self.groups, index)

		return DummyDataset(n)

	def test_train_writes_full_history(self):
		np.random.seed(42)
		model = self._make_model()
		dataset = self._make_dataset()
		model.train(dataset, batch_size=4, patience=2, min_epoch=5, max_epoch=8, num_workers=0, verbose=0)

		data = self.read_mat(self.name)
		n_epochs = len(np.atleast_1d(data['train_losses']).squeeze())
		self.assertGreaterEqual(n_epochs, 5)
		self.assertLessEqual(n_epochs, 8)
		self.assertTrue(np.isfinite(float(data['validation_loss'].squeeze())))
		self.assertTrue(np.isfinite(float(data['train_loss'].squeeze())))
		self.assertIn(str(data['early_stop'].squeeze()), ('loss', 'accuracy'))
		# best epoch must be within trained epochs
		self.assertLess(int(data['epoch'].squeeze()), n_epochs)


if __name__ == '__main__':
	unittest.main()
