import time
import unittest

from src.data import datasets

from . import dataset_tests

ALREADY_DOWNLOADED_OUT = "Files already downloaded and verified"


class TestDatasets(unittest.TestCase):
    """
    Expected to be run from parent folder of src (so data_root is correct).
    """
    @classmethod
    def setUpClass(cls):
        ds = datasets.CIFAR10()
        try:
            ds.prepare_data(val_proportion=0.2)
        except RuntimeError:
            print("\nDownloading CIFAR10 data ...")
            ds.prepare_data(download=True, val_proportion=0.2)
        cls.ds = ds

    def test_channel_means_in_01(self):
        dataset_tests.test_channel_means_in_01(self, self.ds)

    def test_channel_means(self):
        ds = datasets.CIFAR10()

        self.assertEqual(len(ds.channel_means), 3)
        for mean in ds.channel_means:
            self.assertTrue(0. <= mean <= 1.)

    # @patch("sys.stdout", new_callable=StringIO)
    # def test_download(self, mock_stdout):
    #     ds = datasets.CIFAR10()
    #     ds.prepare_data(download=True)
    #
    #     stdout_value = mock_stdout.getvalue()
    #     just_downloaded = "Downloading" in stdout_value
    #     previously_downloaded = ALREADY_DOWNLOADED_OUT in stdout_value
    #     self.assertTrue(just_downloaded or previously_downloaded)

    def test_number_of_partitions(self):
        self.assertEqual(len(self.ds.partitions), 3)

    def test_partition_lengths(self):
        self.assertEqual(len(self.ds.train), 40_000)
        self.assertEqual(len(self.ds.val), 10_000)
        self.assertEqual(len(self.ds.test), 10_000)

    def test_shapes(self):
        dataset_tests.test_shapes(self, self.ds)
        # for partition in self.ds.partitions:
        #     x0, y0 = partition[0]
        #     self.assertEqual(x0.shape, (3, 32, 32))
        #     self.assertIsInstance(y0, int)

    def test_scaling(self):
        for partition in self.ds.partitions:
            x0, _ = partition[0]
            self.assertTrue(0. <= x0.min() <= 0.5)
            self.assertTrue(0.5 <= x0.max() <= 1.)

    def test_initialization_time(self):
        start_time = time.time()

        datasets.CIFAR10()

        initialization_seconds = time.time() - start_time
        self.assertLess(initialization_seconds, 0.01)

    def test_preparation_time(self):
        start_time = time.time()

        ds = datasets.CIFAR10()
        ds.prepare_data(val_proportion=0.2)

        preparation_seconds = time.time() - start_time
        self.assertTrue(0.1 < preparation_seconds < 2.)
