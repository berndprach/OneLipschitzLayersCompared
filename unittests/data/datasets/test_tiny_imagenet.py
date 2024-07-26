import unittest

import torch
from torchvision import transforms

from src import data
from src.data.datasets import TinyImageNet
from src.data.datasets.transform_dataset import transform_dataset


class TestTinyImagenet(unittest.TestCase):
    def test_plain_data_loading(self):
        dataset = TinyImageNet()
        dataset.prepare_data()
        self.assertEqual(len(dataset.train), 90_000)
        self.assertEqual(len(dataset.val), 10_000)
        self.assertEqual(len(dataset.test), 10_000)

    def test_image_size(self):
        dataset = TinyImageNet()
        dataset.prepare_data()
        x, y = dataset.train[0]
        self.assertEqual(x.shape, (3, 64, 64))
        self.assertIsInstance(y, int)

    def test_batching(self):
        dataset = TinyImageNet()
        dataset.prepare_data()
        dl = data.DataLoader(dataset, batch_size=32)
        x, y = next(iter(dl.train))
        self.assertEqual(x.shape, (32, 3, 64, 64))
        self.assertEqual(y.shape, (32,))

    def test_random_augmentation(self):
        rand_aug = transforms.RandAugment(2, 9)

        ds = TinyImageNet(train_augmentation=rand_aug)
        ds.prepare_data()
        dp = data.get_dp(ds, 32, "cpu")

        x_batch, y_batch = next(iter(dp.train))
        self.assertEqual(x_batch.shape, (32, 3, 64, 64))
        self.assertEqual(y_batch.shape, (32,))

        x_batch, y_batch = next(iter(dp.val))
        self.assertEqual(x_batch.shape, (32, 3, 64, 64))
        self.assertEqual(y_batch.shape, (32,))



