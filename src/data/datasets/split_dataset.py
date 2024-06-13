import numpy as np

from torch.utils.data import Subset

from .dataset import ListLike


def split_dataset(ds: ListLike, proportion=0.9, shuffle=True):
    total_size = len(ds)
    indices = list(range(total_size))
    split = int(np.floor(proportion * total_size))

    np.random.seed(1111)
    if shuffle:
        np.random.shuffle(indices)

    part1_indices, part2_indices = indices[:split], indices[split:]
    subset1 = Subset(ds, part1_indices)
    subset2 = Subset(ds, part2_indices)
    return subset1, subset2
