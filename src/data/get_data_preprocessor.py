import torch

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
