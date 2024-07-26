import unittest

from src import data
from src.data.datasets import Imagenette


class TestTinyImagenet(unittest.TestCase):
    def test_plain_data_loading(self):
        dataset = Imagenette()
        dataset.prepare_data()
        self.assertEqual(len(dataset.train), 8_523)
        self.assertEqual(len(dataset.val), 946)
        self.assertEqual(len(dataset.test), 3_925)

    def test_image_size(self):
        dataset = Imagenette()
        dataset.prepare_data()
        x, y = dataset.train[0]
        self.assertEqual(x.shape, (3, 256, 256))
        self.assertIsInstance(y, int)

    def test_batching(self):
        dataset = Imagenette()
        dataset.prepare_data()
        dl = data.DataLoader(dataset, batch_size=32)
        x, y = next(iter(dl.train))
        self.assertEqual(x.shape, (32, 3, 256, 256))
        self.assertEqual(y.shape, (32,))



