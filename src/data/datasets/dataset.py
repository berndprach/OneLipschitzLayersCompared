
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional

DATA_DIR = os.path.join("data", "datasets")


class ListLike(Protocol):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


@dataclass
class SimpleDataset:
    train: ListLike
    val: ListLike
    test: ListLike

    @property
    def partitions(self):
        return [self.train, self.val, self.test]


class Dataset(ABC):
    metadata = {}

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.train: Optional[ListLike] = None
        self.val: Optional[ListLike] = None
        self.test: Optional[ListLike] = None

    @abstractmethod
    def prepare_data(self, download=False, **kwargs) -> Dataset:
        raise NotImplementedError()

    @property
    def partitions(self) -> list[ListLike]:
        return [self.train, self.val, self.test]



