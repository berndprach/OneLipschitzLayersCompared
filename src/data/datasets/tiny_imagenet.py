import os
import zipfile
from functools import partial

from urllib.request import urlretrieve
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from .dataset import Dataset
from .transform_dataset import apply_x_transform

DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATASET_FOLDER = 'tiny-imagenet-200'
DATASET_ZIP_FN = 'tiny-imagenet-200.zip'
VAL_ANNOTATION_FN = 'val_annotations.txt'

TINY_IMAGENET_MEAN = [0.480, 0.448, 0.398]
# [0.43549561500549316, 0.4132375121116638, 0.3745059370994568] ?

# DEFAULT_TRAIN_AUGMENTATION = transforms.Compose([
#     transforms.RandAugment(2, 9),
#     # transforms.RandomHorizontalFlip(),
# ])


def no_aug(x):
    return x


dtf_train = transforms.RandAugment(2, 9)


class TinyImageNet(Dataset):
    channel_means = TINY_IMAGENET_MEAN
    augmentation = staticmethod(no_aug)

    def __init__(self, train_augmentation=dtf_train, **kwargs):
        super().__init__(**kwargs)
        self.ds_augmentation = train_augmentation

    def prepare_data(self, val_proportion=0.1) -> Dataset:
        train_val_root = os.path.join(self.data_dir, DATASET_FOLDER, "train")
        train_val_data = ImageFolder(train_val_root, transform=None)

        # self.val, self.train = split_dataset(train_val, val_proportion)
        val_size = int(val_proportion * len(train_val_data))
        train_size = len(train_val_data) - val_size
        train, val = random_split(train_val_data, [train_size, val_size])

        train_tf = transforms.Compose([self.ds_augmentation, ToTensor()])
        self.train = apply_x_transform(train, train_tf)
        self.val = apply_x_transform(val, ToTensor())

        test_root = os.path.join(self.data_dir, DATASET_FOLDER, "val")
        self.test = ImageFolder(test_root, transform=transforms.ToTensor())

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


# TRAIN_DS_AUGMENTATION = transforms.RandAugment(2, 9)
# AugTinyImageNet = partial(TinyImageNet, TRAIN_DS_AUGMENTATION)

NoAugTinyImageNet = partial(TinyImageNet, train_augmentation=no_aug)
