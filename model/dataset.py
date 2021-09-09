import numpy as np
from sklearn.decomposition import PCA
import torch


class Dataset(object):
    """
    Superclass for experiment datasets
    """
    def __init__(self, p, k, c, seed=1, dtype=torch.float):
        self.seed = seed
        self.dtype = dtype
        self.p = p
        self.k = k
        self.c = c
        self.sigmas = None
        self.mus = None
        self.contexts = None 
        self.X = None
        self.C = None
        self.T = None
        np.random.seed(self.seed)
        self.build()

    def build(self):
        """
        Set up the data generation constants
        """
        self.sigmas = np.zeros(self.k, self.p, self.p)
        self.mus = np.zeros(self.k, self.p)
        self.contexts = np.zeros(self.k, self.c)

    def gen_samples(self, k_n):
        """
        Sample each of the k archetypes k_n times
        """
        self.X = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.T = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.C = torch.zeros(self.k * k_n * self.p**2, self.c, dtype=self.dtype)
        self.B = torch.zeros(self.k * k_n * self.p**2, 1, dtype=self.dtype)
        return self.X, self.T, self.C, self.B

    def train_test_split(test_size):
        """
        Split the samples into train and test
        """
        # todo: add idx and increment for batching, add get_test
    
    def load_batch_data(self, batch_size, shuffle=True, device=None):
        """
        Batched data loading for standard training flows
        """
        n = self.X.shape[0]
        X = self.X
        while True:
            if shuffle:
                idx = torch.randperm(n)
                X = X[idx]
            for i in range(0, n, batch_size):
                num_samples = min(n - i, batch_size)
                X_batch = X[i:i + num_samples]
                T_batch = T[i:i + num_samples]
                C_batch = C[i:i + num_samples]
                if device is None:
                    yield X_batch.detach(), T_batch.detach(), C_batch.detach()
                else:
                    yield X_batch.to(device), T_batch.to(device), C_batch.to(device)

    def load_data(self, batch_size=32, device=None):
        """
        Load batch_size samples chosen at random from the samples
        """
        idx = torch.randperm(self.X.shape[0])[:batch_size]
        if device is None:
            return self.X[idx].detach(), self.T[idx].detach(), self.C[idx].detach()
        return self.X[idx].to(device), self.T[idx].to(device), self.C[idx].to(device)


class SimulationDataset(Dataset):
    """
    Simulation dataset
    """
    def build(self):
        """
        Generate parameters for k p-variate gaussians with context
        """
        self.mus = np.zeros((self.k, self.p))
        self.sigmas = np.zeros((self.k, self.p, self.p))
        self.contexts = np.zeros((self.k, self.c))
        for i in range(self.k):
            self.mus[i] = np.zeros(self.p)
            sigma = np.random.random((self.p, self.p)) * 2 - 1
            sigma = sigma @ sigma.T
            self.sigmas[i] = sigma
        pca = PCA(n_components=self.c)
        self.contexts = pca.fit_transform(self.sigmas.reshape((self.k, self.p ** 2)))

    def gen_samples(self, k_n):
        """
        Generate full datasets of samples (X), tasks (T), contexts (C) and true regression coefficients (B)
        """
        # Sample each distribution
        M = self.k * k_n
        X_sampled = np.zeros((M, self.p))
        for i in range(self.k):
            mu, sigma = self.mus[i], self.sigmas[i]
            sample = np.random.multivariate_normal(mu, sigma, k_n)
            X_sampled[i * k_n:(i + 1) * k_n] = sample

        # Take the cartesian product of the samples and tasks to generate a full dataset
        N = M * self.p ** 2
        self.X = np.zeros((N, 2))
        self.T = np.zeros((N, 2))
        self.C = np.repeat(self.contexts, k_n * self.p ** 2, axis=0)  # (N x self.c)
        self.B = np.zeros((N, 1))
        for n in range(N):
            t_i = (n // self.p) % self.p
            t_j = n % self.p
            m = n // self.p ** 2
            k = n // (k_n * self.p ** 2)
            x_i = X_sampled[m, t_i]
            x_j = X_sampled[m, t_j]
            self.X[n] = [x_i, x_j]
            self.T[n] = [t_i, t_j]
            self.B[n] = self.sigmas[k, t_i, t_j] / self.sigmas[k, t_i, t_i]  # Cov(i, j) / Var(i)
        self.X = torch.tensor(self.X, dtype=self.dtype)
        self.T = torch.tensor(self.T, dtype=self.dtype)
        self.C = torch.tensor(self.C, dtype=self.dtype)
        self.B = torch.tensor(self.B, dtype=self.dtype)
        return self.X, self.T, self.C, self.B

