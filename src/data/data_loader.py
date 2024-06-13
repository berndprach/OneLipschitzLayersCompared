
from torch.utils.data import DataLoader as TDLoader


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle_train=True, num_workers=0):
        ds, bs, st, nw = dataset, batch_size, shuffle_train, num_workers
        self.train = TDLoader(ds.train, bs, shuffle=st, num_workers=nw)
        self.val = TDLoader(ds.val, bs, shuffle=False, num_workers=nw)
        self.test = TDLoader(ds.test, bs, shuffle=False, num_workers=nw)

    def __iter__(self):
        return iter([self.train, self.val, self.test])
