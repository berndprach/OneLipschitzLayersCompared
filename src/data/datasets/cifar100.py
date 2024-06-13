
import torchvision
from torchvision import transforms
from torchvision import datasets as torch_datasets

from .dataset import Dataset
from .split_dataset import split_dataset

Transform = torchvision.transforms.Compose

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, [1., 1., 1.]),
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
])

DEFAULT_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, [1., 1., 1.]),
])


class CIFAR100(Dataset):
    channel_means = CIFAR100_MEAN

    def prepare_data(self, download=False, val_proportion=0.1, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        train_val = torch_datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=download,
            transform=transform,
        )
        self.val, self.train = split_dataset(train_val, val_proportion)

        self.test = torch_datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=download,
            transform=transform,
        )

        return self
