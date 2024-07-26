import torch
from torchvision.transforms import Normalize

from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from .. import data


def get_dp(dataset, batch_size, device, val_proportion=0.1):
    dataset.prepare_data(download=True, val_proportion=val_proportion)
    dl = DataLoader(dataset, batch_size)
    dp = DataPreprocessor(dl)
    dp.data_to(device)
    return dp


def get_data_preprocessor(dataset_name, batch_size, val_proportion=0.1):
    dataset = getattr(data.datasets, dataset_name)()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return data.get_dp(dataset, batch_size, device, val_proportion)


def get_augmented_dp(dataset_name, batch_size, val_proportion=0.1):
    ds = getattr(data.datasets, dataset_name)()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dp = get_dp(ds, batch_size, device, val_proportion)

    dp.apply_to_all_xs(Normalize(mean=ds.channel_means, std=[1., 1., 1.]))
    dp.train.apply_to_x(ds.autmentation)

    return dp
