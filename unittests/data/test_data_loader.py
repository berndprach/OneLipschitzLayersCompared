import math
import os
import time
import unittest

from torch.utils.data import TensorDataset as TDs
from torch import zeros

from src.data import datasets, DataLoader


BATCH_SIZE = 4
IMG_SIZE = (3, 32, 32)

DUMMY_SIZES = (48, 17, 64)


class DummyDataset:
    def __init__(self, img_size):
        s = DUMMY_SIZES
        self.train = TDs(zeros((s[0], *img_size)), zeros((s[0],)))
        self.val = TDs(zeros((s[1], *img_size)), zeros((s[1],)))
        self.test = TDs(zeros((s[2], *img_size)), zeros((s[2],)))


class TestGetDataLoaders(unittest.TestCase):
    # Set num_workers=0. With >0 workers, test takes about 12s to run on CPU.

    @classmethod
    def setUpClass(cls):
        cls.cifar10 = datasets.CIFAR10().prepare_data(val_proportion=0.2)
        cls.dummy_ds = DummyDataset(IMG_SIZE)

    def test_is_iterable(self):
        iter(DataLoader(self.dummy_ds, batch_size=BATCH_SIZE, num_workers=0))
        iter(DataLoader(self.cifar10, batch_size=BATCH_SIZE, num_workers=0))

    def test_batch_shapes(self):
        self._check_batch_shapes(self.dummy_ds)
        self._check_batch_shapes(self.cifar10)

    def test_loader_lengths(self):
        self._check_loader_length(self.dummy_ds, DUMMY_SIZES)

    def test_loading_times(self):
        self._check_loading_time(self.dummy_ds)
        self._check_loading_time(self.cifar10)

    def _check_batch_shapes(self, ds):
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
        for partition_loader in dl:
            first_batch = next(iter(partition_loader))
            self._check_cifar10_shaped(first_batch)

    def _check_loader_length(self, ds, ds_lengths):
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)

        for partition_loader, ds_length in zip(dl, ds_lengths):
            loader_length = math.ceil(ds_length / BATCH_SIZE)
            self.assertEqual(len(partition_loader), loader_length)

    def _check_cifar10_shaped(self, batch):
        self.assertEqual(batch[0].shape, (BATCH_SIZE, *IMG_SIZE))
        self.assertEqual(batch[1].shape, (BATCH_SIZE,))

    def _check_loading_time(self, ds):
        start_time = time.time()

        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
        self.assertLess(time.time() - start_time, 0.01)

        _ = next(iter(dl.train))
        self.assertLess(time.time() - start_time, 0.1)
