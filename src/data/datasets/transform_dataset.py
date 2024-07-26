from torch.utils.data import Dataset as TorchDataset


def none(*args):
    return args


def apply_x_transform(dataset, transform):
    """ Add transformations to the data directly (before batching). """
    return TransformedDataset(dataset, transform)


class TransformedDataset(TorchDataset):
    """
    Add transformations or augmentations to the data directly.
    E.g. useful to get all images to the same size before batching.
    """

    def __init__(self, base_dataset, x_transform):
        self.base_dataset = base_dataset
        self.x_transform = x_transform

    def __getitem__(self, index):
        x, *rest = self.base_dataset[index]
        return self.x_transform(x), *rest

    def __len__(self):
        return len(self.base_dataset)


def transform_dataset(dataset, transform=none, x_transform=none):
    """ Add transformations to the data directly (before batching). """
    return AugmentedDataset(dataset, transform, x_transform)


class AugmentedDataset(TorchDataset):
    """
    Add transformations or augmentations to the data directly.
    E.g. useful to get all images to the same size before batching.
    """

    def __init__(self, base_dataset, transform, x_transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.x_transform = x_transform

    def __getitem__(self, index):
        x, *rest = self.transform(*self.base_dataset[index])
        return self.x_transform(x), *rest

    def __len__(self):
        return len(self.base_dataset)
