import sys
import unittest

from src.data import get_subset, datasets


class TestDatasets(unittest.TestCase):
    """
    Expected to be run from parent folder of src (so data_root is correct).
    """
    def test_subset(self):
        ds = datasets.CIFAR10().prepare_data(val_proportion=0.2)

        subset = get_subset(
            ds, train_size=None, val_size=100, test_size=100_000
        )

        self.assertEqual(len(subset.train), 40_000)
        self.assertEqual(len(subset.val), 100)
        self.assertEqual(len(subset.test), 10_000)


