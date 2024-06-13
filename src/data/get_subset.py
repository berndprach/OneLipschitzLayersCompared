from typing import Optional

from torch.utils.data import Subset

from src.data.datasets.dataset import Dataset


def get_subset(dataset: Dataset,
               train_size=None,
               val_size=None,
               test_size=None
               ) -> Dataset:
    dataset.train = get_sub_partition(dataset.train, train_size)
    dataset.val = get_sub_partition(dataset.val, val_size)
    dataset.test = get_sub_partition(dataset.test, test_size)
    return dataset


def get_sub_partition(data_partition, size: Optional[int] = None):
    if size is None:
        return data_partition
    size = min(size, len(data_partition))  # Otherwise .__len__() lies!
    return Subset(data_partition, range(size))
