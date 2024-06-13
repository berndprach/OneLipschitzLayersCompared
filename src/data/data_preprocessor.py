
from typing import Collection  # Collection: sized iterable container

from src.data import DataLoader


class DataPreprocessor:
    def __init__(self, data_loader: DataLoader):
        self.train = CollectionTransformer(data_loader.train)
        self.val = CollectionTransformer(data_loader.val)
        self.test = CollectionTransformer(data_loader.test)

    @property
    def all(self):
        return [self.train, self.val, self.test]

    def data_to(self, device):
        def to_device(*tensors):
            return [t.to(device) for t in tensors]

        self.apply_to_all(to_device)

    def apply_to_all(self, transform):
        for ct in self.all:
            ct.apply(transform)

    def apply_to_all_xs(self, x_transform):
        for ct in self.all:
            ct.apply_to_x(x_transform)


class CollectionTransformer:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.transforms = []
        self.collection_iter = None

    def __iter__(self):
        return CollectionTransformerIterator(
            iter(self.collection),
            self.transforms,
        )

    def __len__(self):
        return len(self.collection)

    def add(self, transform):
        self.transforms.append(transform)

    def apply(self, transform):
        self.add(transform)

    def apply_to_x(self, transform):
        self.add(lambda x, *rest: (transform(x), *rest))


class CollectionTransformerIterator:
    def __init__(self, base_iterator, transforms):
        self.base_iterator = base_iterator
        self.transforms = transforms

    # def __iter__(self):
    #     return self

    def __next__(self):
        batch = next(self.base_iterator)
        for transform in self.transforms:
            batch = transform(*batch)
        return batch
