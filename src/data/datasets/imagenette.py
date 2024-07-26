import os
import tarfile
from functools import partial
from urllib.request import urlretrieve

from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

from .dataset import Dataset
from .transform_dataset import apply_x_transform

TRAIN_VAL_SUBFOLDER = os.path.join("imagenette2-320", "train")
TEST_SUBFOLDER = os.path.join("imagenette2-320", "val")

# Approximate mean, due to random crop augmentation:
IMAGENETTE_MEAN = [0.465, 0.452, 0.423]


dtf_train = transforms.Compose([
    transforms.RandAugment(2, 9),
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    # transforms.Normalize(IMAGENETTE_MEAN, [1., 1., 1.]),
    # transforms.RandomHorizontalFlip(),
])

dtf_test = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    # transforms.Normalize(IMAGENETTE_MEAN, [1., 1., 1.]),
])


class Imagenette(Dataset):
    """ Data from https://github.com/fastai/imagenette. """
    # channel_means = [0., 0., 0.]  # means post augmentation
    channel_means = IMAGENETTE_MEAN
    augmentation = transforms.Compose([])  # all augmentation done on dataset.

    def __init__(self,
                 train_transform=dtf_train,
                 test_transform=dtf_test,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self, val_proportion=0.1, **kwargs):
        self.train, self.val = get_imagenette_train_val(
            self.data_dir,
            val_proportion,
            self.train_transform,
            self.test_transform
        )
        self.test = get_imagenette_test(
            self.data_dir,
            self.test_transform
        )

        return self

    def download_data(self):
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


def get_imagenette_train_val(data_dir, val_proportion, train_tf, val_tf):
    train_val_path = os.path.join(data_dir, TRAIN_VAL_SUBFOLDER)
    train_val_data = ImageFolder(root=train_val_path, transform=None)

    val_size = int(val_proportion * len(train_val_data))
    train_size = len(train_val_data) - val_size
    train_ds, val_ds = random_split(train_val_data, [train_size, val_size])

    train_ds_tf = apply_x_transform(train_ds, train_tf)
    val_ds_tf = apply_x_transform(val_ds, val_tf)
    return train_ds_tf, val_ds_tf


def get_imagenette_test(data_dir, test_transform):
    test_path = os.path.join(data_dir, TEST_SUBFOLDER)
    test_data = ImageFolder(root=test_path, transform=None)
    test_ds_tf = apply_x_transform(test_data, test_transform)
    return test_ds_tf


NoAugImagenette = partial(Imagenette, train_transform=dtf_test)

