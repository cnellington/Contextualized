import numpy as np
from sklearn.decomposition import PCA
import torch


def onehot_task(t_i, k):
    task_rep = np.zeros(k)
    task_rep[t_i] = 1
    return task_rep


def to_pairwise(C, X, Y, task_fn=onehot_task, dtype=torch.float, device=torch.device('cpu')):
    """
    Load a pairwise (X x Y) dataset of C, T, X, Y
    """
    n, p_x = X.shape
    _, p_y = Y.shape
    p_t = max(p_x, p_y)
    N = n * p_x * p_y
    C_pairwise = np.repeat(C, p_x * p_y, axis=0)
    T_pairwise = np.zeros((N, p_t * 2))
    X_pairwise = np.zeros(N)
    Y_pairwise = np.zeros(N)
    for n in range(N):
        t_i = (n // p_y) % p_x
        t_j = n % p_y
        m = n // (p_x * p_y)
        # k = n // (k_n * self.p ** 2)
        x_i = X[m, t_i]
        y_j = Y[m, t_j]
        X_pairwise[n] = x_i
        Y_pairwise[n] = y_j
        task_x = task_fn(t_i, p_t)
        task_y = task_fn(t_j, p_t)
        taskpair = np.concatenate((task_x, task_y))
        T_pairwise[n] = taskpair
    C_pairwise = torch.tensor(C_pairwise, dtype=dtype, device=device)
    T_pairwise = torch.tensor(T_pairwise, dtype=dtype, device=device)
    X_pairwise = torch.tensor(X_pairwise, dtype=dtype, device=device)
    Y_pairwise = torch.tensor(Y_pairwise, dtype=dtype, device=device)
    return C_pairwise, T_pairwise, X_pairwise, Y_pairwise


class Dataset:
    """
    Dataset
    """
    def __init__(self, C, X, Y, task_fn=onehot_task, seed=None, dtype=torch.float, device=torch.device('cpu')):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        self.dtype = dtype
        self.device = device
        self.task_fn = task_fn
        self.C, self.X, self.Y = C, X, Y
        self.N, self.p_x = X.shape
        _, self.p_y = Y.shape
        self.c = C.shape[-1] 
        # Train/test split
        self.train_idx = np.random.permutation(self.N)

    def load_data(self, batch_size=None, batch_start=None):
        """
        Return a batch from the test set
        batch_size = None gives the full test set
        batch_start = None gives a random subset of batch_size elements
        batch_start = i gives the elements specified by test_idx[batch_start:batch_start+batch_size]
        """
        batch_idx = None
        if batch_size is None:
            batch_idx = self.train_idx
        elif batch_start is None:
            batch_idx = np.random.choice(self.train_idx, size=batch_size, replace=False)
        else:
            if batch_start < 0:
                batch_start += self.N
            batch_end = min(self.N, batch_start + batch_size)
            batch_idx = self.train_idx[batch_start:batch_end]
        return to_pairwise(self.C[batch_idx], self.X[batch_idx], self.Y[batch_idx], device=self.device)

