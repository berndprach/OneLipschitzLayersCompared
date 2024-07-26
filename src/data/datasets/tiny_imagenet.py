import os
import zipfile

from typing import Any, Callable, Optional
from urllib.request import urlretrieve
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import split_dataset
from .dataset import Dataset


DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATASET_FOLDER = 'tiny-imagenet-200'
DATASET_ZIP_FN = 'tiny-imagenet-200.zip'
VAL_ANNOTATION_FN = 'val_annotations.txt'


TINY_IMAGENET_MEAN = [0.4802, 0.4481, 0.3975]

DEFAULT_TRAIN_AUGMENTATION = transforms.Compose([
    transforms.RandAugment(2, 9),
    # transforms.RandomHorizontalFlip(),
])


class TinyImageNet(Dataset):
    channel_means = TINY_IMAGENET_MEAN
    augmentation = DEFAULT_TRAIN_AUGMENTATION

    def prepare_data(self, val_proportion=0.1, transform=None) -> Dataset:
        if transform is None:
            transform = transforms.ToTensor()

        train_val_root = os.path.join(self.data_dir, DATASET_FOLDER, "train")
        train_val_data = ImageFolder(train_val_root, transform=transform)
        print(train_val_data)

        # self.val, self.train = split_dataset(train_val, val_proportion)
        val_size = int(val_proportion * len(train_val_data))
        train_size = len(train_val_data) - val_size
        self.train, self.val = random_split(
            train_val_data, [train_size, val_size]
        )

        test_root = os.path.join(self.data_dir, DATASET_FOLDER, "val")
        self.test = ImageFolder(test_root, transform=transform)

        return self

    def download_data(self):
        url = DATASET_URL
        filename = os.path.join(self.data_dir, DATASET_ZIP_FN)
        if not os.path.exists(filename):
            print(f"Downloading {url} to {filename}...")
            urlretrieve(url, filename)
            print("Done.")

        # Unzip the dataset
        dataset_folder = os.path.join(self.data_dir, DATASET_FOLDER)
        if not os.path.exists(dataset_folder):
            print(f"Unzipping {filename} to {dataset_folder}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Done.")

            # Move the validation images to sub-folders
            val_dir = os.path.join(dataset_folder, "val")
            val_annotation_filename = os.path.join(val_dir, VAL_ANNOTATION_FN)
            print("Moving validation images to sub-folders...")
            with open(val_annotation_filename) as f:
                for line in f:
                    img_filename, label, *_ = line.split()
                    label_dir = os.path.join(val_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    # move image to subfolder
                    old_path = os.path.join(val_dir, "images", img_filename)
                    new_path = os.path.join(label_dir, img_filename)
                    os.rename(old_path, new_path)
                # remove empty image folder
                os.rmdir(os.path.join(val_dir, "images"))
            print("Done.")
