
import time
import unittest

import torch
from torchvision import transforms

from src import data
from src.data import datasets


from src.data import DataPreprocessor


BATCH_SIZE = 4
IMG_SIZE = (3, 32, 32)


class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cifar10 = datasets.CIFAR10().prepare_data(val_proportion=0.2)
        cls.cifar10_dl = data.DataLoader(
            cifar10, batch_size=BATCH_SIZE, num_workers=0
        )

    def test_is_iterable(self):
        dp = DataPreprocessor(self.cifar10_dl)
        for partition_processor in dp.all:
            iter(partition_processor)

    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_data_to_gpu(self):
        self._check_data_to_device(self.cifar10_dl, "cuda")

    def test_data_to_cpu(self):
        self._check_data_to_device(self.cifar10_dl, "cpu")

    def _check_data_to_device(self, dl, device):
        dp = DataPreprocessor(dl)
        dp.data_to(device)
        for partition_dl in dp.all:
            # print("Testing next partition")
            batch_x, batch_y = next(iter(partition_dl))
            # print(f"Got batches of shape {batch_x.shape}, {batch_y.shape}")
            self.assertTrue(device in str(batch_x.device))
            self.assertTrue(device in str(batch_y.device))

    def test_length_stays_same(self):
        dp = DataPreprocessor(self.cifar10_dl)
        self.assertEqual(len(dp.train), len(self.cifar10_dl.train))

    def test_padding_train(self):
        dp = DataPreprocessor(self.cifar10_dl)
        dp.train.apply_to_x(transforms.Pad(4))

        self._check_batch_shapes(dp.train, (BATCH_SIZE, 3, 40, 40))
        self._check_batch_shapes(dp.val, (BATCH_SIZE, 3, 32, 32))
        self._check_batch_shapes(dp.test, (BATCH_SIZE, 3, 32, 32))

    def _check_batch_shapes(self, partition_loader, goal_shape):
        x, y = next(iter(partition_loader))
        self.assertEqual(x.shape, goal_shape)
        self.assertEqual(y.shape, (BATCH_SIZE,))

    def test_padding_all(self):
        dp = DataPreprocessor(self.cifar10_dl)
        dp.apply_to_all_xs(transforms.Pad(2))

        for partition_loader in dp.all:
            self._check_batch_shapes(partition_loader, (BATCH_SIZE, 3, 36, 36))

    def test_multiple_transforms(self):
        dp = DataPreprocessor(self.cifar10_dl)
        dp.train.apply_to_x(transforms.RandomCrop(28))
        dp.train.apply_to_x(transforms.Pad(3))

        self._check_batch_shapes(dp.train, (BATCH_SIZE, 3, 34, 34))

    def test_transformation_time(self):
        dp = DataPreprocessor(self.cifar10_dl)
        start_time = time.time()
        dp.train.apply_to_x(transforms.RandomCrop(28))
        dp.train.apply_to_x(transforms.Pad(3))
        self.assertLess(time.time() - start_time, 0.01)

        _ = next(iter(dp.train))
        self.assertLess(time.time() - start_time, 0.01)
