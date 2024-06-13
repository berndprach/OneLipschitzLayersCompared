import torch

from torchvision import transforms as tfs


class GaussianNoise:
    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        noised_x = x + noise
        return torch.clip_(noised_x, 0., 1.)


cifar_basic_augmentation = tfs.Compose([
    tfs.RandomCrop(32, 4),
    # tfs.Lambda(IndividualRandomCrop(32, 4)),
    tfs.RandomHorizontalFlip(),
])
