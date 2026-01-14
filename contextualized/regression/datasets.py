"""
Data generators used for Contextualized regression training.
"""

from abc import abstractmethod
import torch
from torch.utils.data import Dataset


class MultivariateDataset(Dataset):
    """
    Simple multivariate dataset with context, predictors, and outcomes.
    """
    def __init__(self, C, X, Y, orig_idx=None, dtype=torch.float):
        self.C = torch.as_tensor(C, dtype=dtype)
        self.X = torch.as_tensor(X, dtype=dtype)
        self.Y = torch.as_tensor(Y, dtype=dtype)

        if orig_idx is None:
            self.orig_idx = torch.arange(len(self.C), dtype=torch.long)
        else:
            self.orig_idx = torch.as_tensor(orig_idx, dtype=torch.long).view(-1)

        self.c_dim = self.C.shape[-1]
        self.x_dim = self.X.shape[-1]
        self.y_dim = self.Y.shape[-1]
        self.dtype = dtype

    def __len__(self):
        return len(self.C)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "orig_idx": self.orig_idx[idx],
            "contexts": self.C[idx],
            "predictors": self.X[idx].expand(self.y_dim, -1),
            "outcomes": self.Y[idx].unsqueeze(-1),
        }


class UnivariateDataset(Dataset):
    """
    Simple univariate dataset with context, predictors, and one outcome.
    """
    def __init__(self, C, X, Y, orig_idx=None, dtype=torch.float):
        self.C = torch.as_tensor(C, dtype=dtype)
        self.X = torch.as_tensor(X, dtype=dtype)
        self.Y = torch.as_tensor(Y, dtype=dtype)

        if orig_idx is None:
            self.orig_idx = torch.arange(len(self.C), dtype=torch.long)
        else:
            self.orig_idx = torch.as_tensor(orig_idx, dtype=torch.long).view(-1)

        self.c_dim = self.C.shape[-1]
        self.x_dim = self.X.shape[-1]
        self.y_dim = self.Y.shape[-1]
        self.dtype = dtype

    def __len__(self):
        return len(self.C)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "orig_idx": self.orig_idx[idx],
            "contexts": self.C[idx],
            "predictors": self.X[idx].expand(self.y_dim, -1).unsqueeze(-1),
            "outcomes": self.Y[idx].expand(self.x_dim, -1).T.unsqueeze(-1),
        }


class MultitaskMultivariateDataset(Dataset):
    """
    Multi-task Multivariate Dataset.
    """
    def __init__(self, C, X, Y, orig_idx=None, dtype=torch.float):
        self.C = C.to(dtype) if isinstance(C, torch.Tensor) else torch.as_tensor(C, dtype=dtype)
        self.X = X.to(dtype) if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=dtype)
        self.Y = Y.to(dtype) if isinstance(Y, torch.Tensor) else torch.as_tensor(Y, dtype=dtype)

        if orig_idx is None:
            self.orig_idx = torch.arange(len(self.C), dtype=torch.long)
        else:
            self.orig_idx = torch.as_tensor(orig_idx, dtype=torch.long).view(-1)

        self.c_dim = self.C.shape[-1]
        self.x_dim = self.X.shape[-1]
        self.y_dim = self.Y.shape[-1]
        self.dtype = dtype

    def __len__(self):
        return len(self.C) * self.y_dim

    def __getitem__(self, idx):
        # Get task-split sample indices
        n_i = idx // self.y_dim
        y_i = idx % self.y_dim

        # Create a one-hot encoding for the task
        t = torch.zeros(self.y_dim, dtype=self.dtype)
        t[y_i] = 1

        return {
            "idx": idx,
            "orig_idx": self.orig_idx[n_i],
            "contexts": self.C[n_i],
            "task": t,
            "predictors": self.X[n_i],
            "outcomes": self.Y[n_i, y_i].unsqueeze(0),
            "sample_idx": n_i,
            "outcome_idx": y_i,
        }


class MultitaskUnivariateDataset(Dataset):
    """
    Multitask Univariate Dataset.
    Splits each sample into univariate X and Y feature pairs for univariate regression tasks.
    """
    def __init__(self, C, X, Y, orig_idx=None, dtype=torch.float):
        self.C = torch.as_tensor(C, dtype=dtype)
        self.X = torch.as_tensor(X, dtype=dtype)
        self.Y = torch.as_tensor(Y, dtype=dtype)

        if orig_idx is None:
            self.orig_idx = torch.arange(len(self.C), dtype=torch.long)
        else:
            self.orig_idx = torch.as_tensor(orig_idx, dtype=torch.long).view(-1)

        self.c_dim = self.C.shape[-1]
        self.x_dim = self.X.shape[-1]
        self.y_dim = self.Y.shape[-1]
        self.dtype = dtype

    def __len__(self):
        return len(self.C) * self.x_dim * self.y_dim

    def __getitem__(self, idx):
        # Get task-split sample indices
        n_i = idx // (self.x_dim * self.y_dim)
        x_i = (idx // self.y_dim) % self.x_dim
        y_i = idx % self.y_dim

        # Create a one-hot encoding for the task
        t = torch.zeros(self.x_dim + self.y_dim, dtype=self.dtype)
        t[x_i] = 1
        t[self.x_dim + y_i] = 1

        return {
            "idx": idx,
            "orig_idx": self.orig_idx[n_i],
            "contexts": self.C[n_i],
            "task": t,
            "predictors": self.X[n_i, x_i].unsqueeze(0),
            "outcomes": self.Y[n_i, y_i].unsqueeze(0),
            "sample_idx": n_i,
            "predictor_idx": x_i,
            "outcome_idx": y_i,
        }
