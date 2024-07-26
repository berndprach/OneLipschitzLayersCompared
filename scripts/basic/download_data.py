

from src.data import datasets


def main(dataset_name: str = "CIFAR10"):
    ds_cls = getattr(datasets, dataset_name)
    ds: datasets.Dataset = ds_cls()
    ds.download_data()
