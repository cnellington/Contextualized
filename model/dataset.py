import numpy as np
import torch


class Dataset(object):
    """
    Superclass for experiment datasets
    """
    def __init__(self, p, k, c, seed=1):
        self.seed = seed
        np.random.seed(self.seed)
        self.p = p
        self.k = k
        self.c = c
        self.X = None
        self.C = None
        self.T = None

    def gen_samples(self, k_n):
        """
        Generate n samples for each of the k archetypes
        """
        self.X = torch.zeros(self.k * k_n * self.p**2, 2)
        self.T = torch.zeros(self.k * k_n * self.p**2, 2)
        self.C = torch.zeros(self.k * k_n * self.p**2, self.c)
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
                samples = X[i:i + num_samples]
                if device is None:
                    yield samples.detach()
                else:
                    yield samples.to(device)

    def load_data(self, batch_size=32, device=None):
        idx = torch.randperm(self.X.shape[0])[:batch_size]
        if device is None:
            return self.X[idx]
        return self.X[idx].to(device)


class SimulationDataset(Dataset):
    """
    Simulation dataset
    """
    def gen_samples(self, k_n):
        # todo
        raise NotImplementedError
        
