import numpy as np
from sklearn.decomposition import PCA
import torch


class GaussianSimulator:
    """
    Generate samples with known correlation
    """
    def __init__(self, p, k, c, seed=None):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        # Distribution generation parameters
        self.p = p
        self.k = k
        self.c = c
        # Distribution parameters
        self.sigmas = None
        self.mus = None
        self.contexts = None
        self._build()

    def _build(self):
        """
        Generate parameters for k p-variate gaussians with context
        """
        self.mus = np.zeros((self.k, self.p))
        self.sigmas = np.zeros((self.k, self.p, self.p))
        self.contexts = np.zeros((self.k, self.c))
        for i in range(self.k):
            self.mus[i] = np.zeros(self.p)
            self.contexts[i] = np.random.random((self.c,))
            # TODO: generate sigma using eigen decomposition
            sigma = np.random.random((self.p, self.p)) * 2 - 1
            sigma = sigma @ sigma.T
            self.sigmas[i] = sigma

    def gen_samples(self, k_n):
        """
        Generate full datasets of samples
        """
        # Sample each distribution
        n = self.k * k_n
        C = np.zeros((n, self.c))
        X = np.zeros((n, self.p))
        for i in range(self.k):
            mu, sigma, context = self.mus[i], self.sigmas[i], self.contexts[i]
            sample = np.random.multivariate_normal(mu, sigma, k_n)
            C[i * k_n:(i + 1) * k_n] = context
            X[i * k_n:(i + 1) * k_n] = sample
        return C, X


class Dataset:
    """
    Dataset
    """
    def __init__(self, C, X, testsplit=0.2, seed=1, dtype=torch.float):
        self.seed = seed
        np.random.seed(self.seed)
        self.dtype = dtype
        self.n, self.p = X.shape
        self.c = C.shape[-1] 
        self.N = self.n * self.p ** 2
        # Train/test split
        split = int(self.N * testsplit)
        idx = torch.randperm(self.N)
        self.train_idx = idx[:-split]
        self.test_idx = idx[-split:]
        self.batch_i = 0
        self.epoch = 0
        # Transform into task pair dataset
        self._build(C, X)

    def _build(self, C, X):
        """
        Build the task pairs
        """
#         # TODO: Normalize C and X
#         X_train = X[self.train_idx]
#         C_train = C[self.train_idx]
#         X = (X - torch.mean(X_train, 0)) / torch.std(X_train, 0)
#         C = (C - torch.mean(C_train, 0)) / torch.std(C_train, 0)
        self.C = np.repeat(C, self.p ** 2, axis=0)
        self.T = np.zeros((self.N, 2 * self.p))
        self.X = np.zeros((self.N, 2))
        for n in range(self.N):
            t_i = (n // self.p) % self.p
            t_j = n % self.p
            m = n // self.p ** 2
            # k = n // (k_n * self.p ** 2)
            x_i = X[m, t_i]
            x_j = X[m, t_j]
            self.X[n] = [x_i, x_j]
            taskpair = np.zeros(self.p * 2)
            taskpair[t_i] = 1
            taskpair[self.p + t_j] = 1
            self.T[n] = taskpair
        self.C = torch.tensor(self.C, dtype=self.dtype)
        self.T = torch.tensor(self.T, dtype=self.dtype)
        self.X = torch.tensor(self.X, dtype=self.dtype)

    def get_test(self):
        """
        Return the test set from train_test_split
        """
        return self.C[self.test_idx], self.T[self.test_idx], self.X[self.test_idx]
    
    def load_data(self, batch_size=32, device=None):
        """
        Load batch_size samples from the training set
        A single epoch should see training samples exactly once
        """
        batch_end = min(self.N, self.batch_i + batch_size)
        batch_idx = self.train_idx[self.batch_i:batch_end]
        if batch_end == self.N:
            self.batch_i = 0
            self.epoch += 1
        else:
            self.batch_i += batch_size
        C_batch = self.C[batch_idx]
        T_batch = self.T[batch_idx]
        X_batch = self.X[batch_idx]
        if device is None:
            return C_batch.detach(), T_batch.detach(), X_batch.detach()
        return C_batch.to(device), T_batch.to(device), X_batch.to(device)

