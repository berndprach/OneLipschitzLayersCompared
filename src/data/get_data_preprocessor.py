
from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor


def get_dp(dataset, batch_size, device, val_proportion=0.1):
    # Standard way of getting a data preprocessor.
    dataset.prepare_data(download=True, val_proportion=val_proportion)
    dl = DataLoader(dataset, batch_size)
    dp = DataPreprocessor(dl)
    dp.data_to(device)
    return dp
