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
        self.X = None
        self.C = None
        self.T = None
        np.random.seed(self.seed)

    def gen_samples(self, k_n):
        """
        Generate n samples for each of the k archetypes
        """
        self.X = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.T = torch.zeros(self.k * k_n * self.p**2, 2, dtype=self.dtype)
        self.C = torch.zeros(self.k * k_n * self.p**2, self.c, dtype=self.dtype)
        return self.X, self.T, self.C
    
    def load_batch_data(self, batch_size, shuffle=True, device=None):
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
        idx = torch.randperm(self.X.shape[0])[:batch_size]
        if device is None:
            return self.X[idx].detach(), self.T[idx].detach(), self.C[idx].detach()
        return self.X[idx].to(device), self.T[idx].to(device), self.C[idx].to(device)


class SimulationDataset(Dataset):
    """
    Simulation dataset
    """
    def gen_samples(self, k_n):
        n = self.k * k_n
        # Generate k p-variate gaussians
        means = []
        covs = []
        for _ in range(self.k):
            mean = np.zeros(self.p)
            cov = np.random.rand(self.p, self.p)
            cov = cov @ cov.T
            covs.append(cov)
            means.append(mean)
        context_full = np.copy(covs).reshape(self.k, self.p ** 2)
        pca = PCA(n_components=self.c)
        pca.fit(context_full)

        samples = np.zeros((n, self.p))
        contexts = np.zeros((n, self.c))
        cov_labels = np.zeros((n, self.p, self.p))
        distribution_ids = np.zeros(n)
        for distribution_id, (mean, cov) in enumerate(zip(means, covs)):
            samples[distribution_id*k_n:(distribution_id+1)*k_n] = np.random.multivariate_normal(mean, cov, k_n)
            contexts[distribution_id*k_n:(distribution_id+1)*k_n]= np.repeat(pca.transform([cov.flatten()]), k_n, axis=0)
            cov_labels[distribution_id*k_n:(distribution_id+1)*k_n] = np.repeat([cov], k_n, axis=0)
            distribution_ids[distribution_id*k_n:(distribution_id+1)*k_n] = np.ones(k_n) * distribution_id

#         return contexts, samples, cov_labels, distribution_ids
        C = contexts
        X = samples
        N = n * self.p ** 2
        C_all = np.repeat(C, self.p ** 2, axis=0)
        self.C = torch.tensor(C_all, dtype=self.dtype)
        sample_ids = np.repeat(np.arange(C.shape[0]).astype(int), self.p ** 2)
        self.sample_ids = sample_ids
        Xi = np.zeros((N, 1))
        Xj = np.zeros((N, 1))
        Ti = np.zeros((N, 1))
        Tj = np.zeros((N, 1))
        m = 0
        for k in range(n):
            for i in range(self.p):
                for j in range(self.p):
                    Xi[m, 0] = X[self.k, i]
                    Xj[m, 0] = X[self.k, j]
                    Ti[m, 0] = i
                    Tj[m, 0] = j
                    m += 1
        X = np.hstack((Xi, Xj))
        T = np.hstack((Ti, Tj))
        self.X = torch.tensor(X, dtype=self.dtype)
        self.T = torch.tensor(T, dtype=self.dtype)
        return self.X, self.T, self.C

