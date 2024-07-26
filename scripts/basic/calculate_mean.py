import torch
from tqdm import tqdm

from src.data import datasets


def main(dataset_name: str = "CIFAR10"):
    ds_cls = getattr(datasets, dataset_name)
    ds: datasets.Dataset = ds_cls()
    ds.prepare_data(val_proportion=0.)

    print(f"Calculating mean for {dataset_name}:")
    image_means = []
    for x, _ in tqdm(ds.train, leave=True):
        image_means.append(x.mean(dim=(1, 2)))

    mean = torch.stack(image_means).mean(dim=0)
    print(mean.tolist())
