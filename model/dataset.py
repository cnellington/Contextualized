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
            # TODO: generate sigma w eigen decomposition
            sigma = np.random.random((self.p, self.p)) * 2 - 1
            sigma = sigma @ sigma.T
            self.sigmas[i] = sigma

    def gen_samples(self, k_n):
        """
        Generate full datasets of samples
        """
        # Sample each distribution
        n = self.k * k_n
        X = np.zeros((n, self.p))
        C = np.zeros((n, self.c))
        for i in range(self.k):
            mu, sigma, context = self.mus[i], self.sigmas[i], self.contexts[i]
            sample = np.random.multivariate_normal(mu, sigma, k_n)
            X[i * k_n:(i + 1) * k_n] = sample
            C[i * k_n:(i + 1) * k_n] = context
        return X, C


class Dataset(object):
    """
    General dataset
    TODO: create T from X[train]
    """
    def __init__(self, C, X, testsplit=0.2, taskdims='full', seed=None, dtype=torch.float):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        self.dtype = dtype
        self.testsplit = testsplit
        self.n, self.p = self.X.shape
        self.c = self.C.shape[-1] 
        self.t = self.p if taskdims == 'full' else taskdims
        self.batch_i = 0
        # Build full dataset
        self._build(X, C)

    def _build(self, X, C):
        """
        Build the dataset
        """
        # Train/test split
        self.N = self.n * self.p ** 2
        testsize = int(self.N * self.testsplit)
        shuffle_idx = torch.randperm(self.N)
        self.train_idx = shuffle_idx[:-testsize]
        self.test_idx = shuffle_idx[-testsize:]
       
        # Get task representations
        self.task_transformer = PCA(self.t)
        self.reverse_idx = np.zeros((self.N, 3))  # (i, j, k)
        self.X = np.zeros((self.N, 2))  # (Xi[j], Xi[k])
        self.T = np.zeros((self.N, 2 * t))  # (Ti[j], Ti[k])

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

    def gen_samples(self, k_n):
        """
        Generate from k distributions or load pre-defined samples
        Sample each of the k archetypes k_n times
        """
        self.X = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.T = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.C = torch.zeros(self.k * k_n * self.p**2, self.c, dtype=self.dtype)
        self.B = torch.zeros(self.k * k_n * self.p**2, 1, dtype=self.dtype)
        return self.X, self.T, self.C
        # todo: add idx and increment for batching, add get_test
        # todo: normalize X? Generate context/tasks according to trainset

    def get_test(self):
        """
        Return the test set from train_test_split
        """
        return self.X[self.test_idx], self.T[self.test_idx], self.C[self.test_idx]
    
    def load_batch_data(self, batch_size, shuffle=True, device=None):
        """
        Batched data loading for standard training flows
        """
        X_train = self.X[self.train_idx]
        T_train = self.T[self.train_idx]
        C_train = self.C[self.train_idx]
        X_epoch = X_train[self.batch_idx]
        T_epoch = T_train[self.batch_idx]
        C_epoch = C_train[self.batch_idx]
        n = self.train_idx.shape[0]
        while True:
            for i in range(0, n, batch_size):
                num_samples = min(n - i, batch_size)
                X_batch = X_epoch[i:i + num_samples]
                T_batch = T_epoch[i:i + num_samples]
                C_batch = C_epoch[i:i + num_samples]
                if device is None:
                    yield X_batch.detach(), T_batch.detach(), C_batch.detach()
                else:
                    yield X_batch.to(device), T_batch.to(device), C_batch.to(device)

    def load_data(self, batch_size=32, device=None):
        """
        Load batch_size samples chosen at random from the samples
        """
        X_batch = (self.X[train_idx])[batch_idx]
        T_batch = (self.T[train_idx])[batch_idx]
        C_batch = (self.C[train_idx])[batch_idx]

        idx = torch.randperm(self.X.shape[0])[:batch_size]
        if device is None:
            return self.X[idx].detach(), self.T[idx].detach(), self.C[idx].detach()
        return self.X[idx].to(device), self.T[idx].to(device), self.C[idx].to(device)


class SimulationDataset:
    """
    Simulation dataset
    """
    def __init__(self, p, k, c, testsize=0.2, seed=None, dtype=torch.float):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        self.dtype = dtype
        # Distribution generation parameters
        self.p = p
        self.k = k
        self.c = c
        self.testsize = testsize
        self.train_idx = None
        self.test_idx = None
        # Distribution parameters
        self.sigmas = None
        self.mus = None
        self.contexts = None
        # Sampling
        self.X = None
        self.C = None
        self.T = None
        self.batch_i = 0
        self.build()

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

    def load_data(self, batch_size=32, device=None):
        """
        Load batch_size samples chosen at random from the samples
        """
        idx = torch.randperm(self.X.shape[0])[:batch_size]
        if device is None:
            return self.X[idx].detach(), self.T[idx].detach(), self.C[idx].detach()
        return self.X[idx].to(device), self.T[idx].to(device), self.C[idx].to(device)
    
    def load_batch_data(self, batch_size, shuffle=True, device=None):
        """
        Batched data loading for standard training flows
        """
        X_train = self.X[self.train_idx]
        T_train = self.T[self.train_idx]
        C_train = self.C[self.train_idx]
        X_epoch = X_train[self.batch_idx]
        T_epoch = T_train[self.batch_idx]
        C_epoch = C_train[self.batch_idx]
        n = self.train_idx.shape[0]
        while True:
            for i in range(0, n, batch_size):
                num_samples = min(n - i, batch_size)
                X_batch = X_epoch[i:i + num_samples]
                T_batch = T_epoch[i:i + num_samples]
                C_batch = C_epoch[i:i + num_samples]
                if device is None:
                    yield X_batch.detach(), T_batch.detach(), C_batch.detach()
                else:
                    yield X_batch.to(device), T_batch.to(device), C_batch.to(device)

