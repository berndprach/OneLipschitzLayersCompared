
from torchvision import transforms

from torchvision import datasets as torch_datasets

from .dataset import Dataset
from .split_dataset import split_dataset

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]


class CIFAR10(Dataset):
    channel_means = CIFAR10_MEAN

    def prepare_data(self, download=False, val_proportion=0.1, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train_val = torch_datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=download,
            transform=transform,
        )
        self.val, self.train = split_dataset(train_val, val_proportion)

        self.test = torch_datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=download,
            transform=transform,
        )

        return self


DEFAULT_TRAIN_AUGMENTATION = transforms.Compose([
    transforms.Normalize(mean=CIFAR10_MEAN, std=[1., 1., 1.]),
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
])
