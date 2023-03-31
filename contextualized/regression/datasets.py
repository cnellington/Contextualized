"""
Datasets used for Contextualized regression training.
"""
from abc import abstractmethod
import torch


class RegressionDatasetBase:
    """
    Superclass for map-based datasets used to train contextualized.regression models
    """

    def __init__(self, C, X, Y, dtype=torch.float):

        self.dtype = dtype
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)

        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.n = self.C.shape[0]

        assert len(set([self.C.shape[0], self.X.shape[0], self.Y.shape[0]])) == 1

        self.n_i = 0
        self.y_i = 0
        self.x_i = 0
        self.sample_ids = []

        for i in range(len(self)):
            self.sample_ids.append(self._get_next_id(i))

    @abstractmethod
    def _total_len(self):
        pass

    @abstractmethod
    def _get_next_id(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass


class UnivariateDataset(RegressionDatasetBase):
    """
    Simple univariate dataset with context, predictors, and one outcome.
    """

    def _get_next_id(self, i):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = self.n_i
        self.n_i += 1
        return ret

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        n_i = self.sample_ids[idx]
        ret = (
            self.C[n_i],
            self.X[n_i].expand(self.y_dim, -1).unsqueeze(-1),
            self.Y[n_i].expand(self.x_dim, -1).T.unsqueeze(-1),
            n_i,
        )
        return ret


class MultivariateDataset(RegressionDatasetBase):
    """
    Simple multivariate dataset with context, predictors, and outcomes.
    """

    def _get_next_id(self, i):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = self.n_i
        self.n_i += 1
        return ret

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # one vs multiple
        n_i = self.sample_ids[idx]
        ret = (
            self.C[n_i],
            self.X[n_i].expand(self.y_dim, -1),
            self.Y[n_i].unsqueeze(-1),
            n_i,
        )
        return ret


class MultitaskUnivariateDataset(RegressionDatasetBase):
    """
    Multitask Univariate Dataset
    """

    def _get_next_id(self, i):
        if self.y_i >= self.y_dim:
            self.x_i += 1
            self.y_i = 0
        if self.x_i >= self.x_dim:
            self.n_i += 1
            self.x_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration

        t = [0] * (self.x_dim + self.y_dim)
        t[self.x_i] = 1
        t[self.x_dim + self.y_i] = 1

        ret = (
            t,
            self.n_i,
            self.x_i,
            self.y_i,
        )

        self.y_i += 1
        return ret

    def __len__(self):
        return self.n * self.y_dim * self.x_dim

    def __getitem__(self, idx):
        t, n_i, x_i, y_i = self.sample_ids[idx]
        t = torch.zeros(self.x_dim + self.y_dim)
        ret = (
            self.C[n_i],
            torch.tensor(t, dtype=torch.float),
            self.X[n_i, x_i].unsqueeze(0),
            self.Y[n_i, y_i].unsqueeze(0),
            n_i,
            x_i,
            y_i,
        )
        return ret


class MultitaskMultivariateDataset(RegressionDatasetBase):
    """
    Multi-task Multivariate Dataset.
    """

    def _get_next_id(self, i):
        if self.y_i >= self.y_dim:
            self.n_i += 1
            self.y_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = [0] * (self.y_dim)
        t[self.y_i] = 1

        ret = (t, self.n_i, self.y_i)
        self.y_i += 1
        return ret

    def __len__(self):
        return self.n * self.y_dim

    def __getitem__(self, idx):
        t, n_i, y_i = self.sample_ids[idx]
        ret = (
            self.C[n_i],
            torch.tensor(t, dtype=torch.float),
            self.X[n_i],
            self.Y[n_i, y_i].unsqueeze(0),
            n_i,
            y_i,
        )
        return ret


DATASETS = {
    "multivariate": MultivariateDataset,
    "univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}
