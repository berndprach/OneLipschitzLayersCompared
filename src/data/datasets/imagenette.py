import os
import tarfile
import zipfile
from urllib.request import urlretrieve

from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset

from .dataset import Dataset


TRAIN_VAL_SUBFOLDER = os.path.join("imagenette2-320", "train")
TEST_SUBFOLDER = os.path.join("imagenette2-320", "val")

# Approximate mean, due to random crop augmentation:
IMAGENETTE_MEAN = [0.46, 0.45, 0.42]


train_transform = transforms.Compose([
    transforms.RandAugment(2, 9),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENETTE_MEAN, [1., 1., 1.]),
    # transforms.RandomHorizontalFlip(),
])

test_transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENETTE_MEAN, [1., 1., 1.]),
])


class AugmentedImagenette(Dataset):
    """
    Data from https://github.com/fastai/imagenette
    """
    channel_means = [0., 0., 0.]  # means post augmentation
    # augmentation = lambda *args: args  # no further augmentation needed
    augmentation = transforms.Compose([])  # no further augmentation needed

    def prepare_data(self, val_proportion=0.1, **kwargs):
        self.train, self.val = get_imagenette_train_val(
            self.data_dir,
            val_proportion,
        )
        self.test = get_imagenette_test(self.data_dir)

        return self

    def download_data(self):
        # raise NotImplementedError(
        #     f"Please download Imagenette data manually.\n "
        #     f"Source: https://github.com/fastai/imagenette"
        # )

        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        filename = os.path.join(self.data_dir, "imagenette2-320.tgz")
        if not os.path.exists(filename):
            print(f"Downloading {url} to {filename}...")
            urlretrieve(url, filename)
            print("Done.")

        # Unzip the dataset
        dataset_folder = os.path.join(self.data_dir, "imagenette2-320")
        if not os.path.exists(dataset_folder):
            print(f"Unzipping {filename} to {dataset_folder}...")
            with tarfile.open(filename, 'r') as tar_ref:
                tar_ref.extractall(self.data_dir)
            print("Done.")


def get_imagenette_train_val(data_dir, val_proportion):
    train_val_path = os.path.join(data_dir, TRAIN_VAL_SUBFOLDER)
    train_val_data = ImageFolder(root=train_val_path, transform=None)

    val_size = int(val_proportion * len(train_val_data))
    train_size = len(train_val_data) - val_size
    train_ds, val_ds = random_split(train_val_data, [train_size, val_size])

    train_ds_tf = AugmentedDataset(train_ds, train_transform)
    val_ds_tf = AugmentedDataset(val_ds, test_transform)
    return train_ds_tf, val_ds_tf


def get_imagenette_test(data_dir):
    test_path = os.path.join(data_dir, TEST_SUBFOLDER)
    test_data = ImageFolder(root=test_path, transform=None)
    test_ds_tf = AugmentedDataset(test_data, test_transform)
    return test_ds_tf


class AugmentedDataset(TorchDataset):
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
