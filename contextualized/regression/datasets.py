from abc import abstractmethod
import torch
from torch.utils.data import IterableDataset


class Dataset:
    """
    Superclass for datastreams (iterators) used to train contextualized.regression models
    """
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        self.n = C.shape[0]
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype

    def __iter__(self):
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class MultivariateDataset(Dataset):
    def __next__(self):
        if self.n_i >= self.C.shape[0]:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1),
            self.Y[self.n_i].unsqueeze(-1),
            self.n_i,
        )
        self.n_i += 1
        return ret

    def __len__(self):
        return self.n


class UnivariateDataset(Dataset):
    def __next__(self):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1).unsqueeze(-1),
            self.Y[self.n_i].expand(self.x_dim, -1).T.unsqueeze(-1),
            self.n_i,
        )
        self.n_i += 1
        return ret

    def __len__(self):
        return self.n


class MultitaskMultivariateDataset(Dataset):
    def __next__(self):
        if self.y_i >= self.y_dim:
            self.n_i += 1
            self.y_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = torch.zeros(self.y_dim)
        t[self.y_i] = 1
        ret = (
            self.C[self.n_i],
            t,
            self.X[self.n_i],
            self.Y[self.n_i, self.y_i].unsqueeze(0),
            self.n_i,
            self.y_i,
        )
        self.y_i += 1
        return ret

    def __len__(self):
        return self.n * self.y_dim


class MultitaskUnivariateDataset(Dataset):
    def __next__(self):
        if self.y_i >= self.y_dim:
            self.x_i += 1
            self.y_i = 0
        if self.x_i >= self.x_dim:
            self.n_i += 1
            self.x_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = torch.zeros(self.x_dim + self.y_dim)
        t[self.x_i] = 1
        t[self.x_dim + self.y_i] = 1
        ret = (
            self.C[self.n_i],
            t,
            self.X[self.n_i, self.x_i].unsqueeze(0),
            self.Y[self.n_i, self.y_i].unsqueeze(0),
            self.n_i,
            self.x_i,
            self.y_i,
        )
        self.y_i += 1
        return ret

    def __len__(self):
        return self.n * self.x_dim * self.y_dim


class DataIterable(IterableDataset):
    """
    Dataset wrapper, required by PyTorch
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)


def singletask_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    split_size = len(dataset.dataset.C) // worker_info.num_workers  # divide equally among workers
    start = worker_id*split_size
    end = (worker_id+1)*split_size
    dataset.dataset.C = dataset.dataset.C[start: end]
    dataset.dataset.X = dataset.dataset.X[start: end]
    dataset.dataset.Y = dataset.dataset.Y[start: end]
