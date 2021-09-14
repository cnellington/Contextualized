import unittest
import numpy as np
import torch
from model.dataset import Dataset, GaussianSimulator


class TestGaussianSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGaussianSimulator, self).__init__(*args, **kwargs)
        self.p = 10
        self.c = 5
        self.k = 5
        self.sim = GaussianSimulator(self.p, self.k, self.c)
    
    def test_empirical_cov(self):
        k_n = int(1e5)
        C, X = self.sim.gen_samples(k_n)
        for i in range(self.k):
            X_sample = X[i * k_n:(i+1) * k_n]
            empirical_cov = 1 / (k_n - 1) * X_sample.T @ X_sample
            assert np.allclose(empirical_cov, self.sim.sigmas[i], atol=1e-1)


class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDataset, self).__init__(*args, **kwargs)
        self.dtype = torch.float
        self.p = 10
        self.c = 5
        self.k = 5
        self.k_n = 10
        self.sim = GaussianSimulator(self.p, self.k, self.c)
        self.C_full, self.X_full = self.sim.gen_samples(self.k_n)
        self.db = Dataset(self.C_full, self.X_full, dtype=self.dtype)

    def test_taskpairs(self):
        N = self.k * self.k_n * self.p ** 2
        assert self.db.X.shape == (N, 2)
        assert self.db.T.shape == (N, 2 * self.p)
        assert self.db.C.shape == (N, self.c)
        n = 0
        for k in range(self.k):
            for m in range(self.k_n):
                for i in range(self.p):
                    for j in range(self.p):
                        C_tensor = torch.tensor(self.sim.contexts[k], dtype=self.dtype)
                        X_full_tensor = torch.tensor(self.X_full[k * self.k_n + m], dtype=self.dtype)
                        assert (self.db.C[n] == C_tensor).all()
                        assert self.db.T[n, i] == 1
                        assert self.db.T[n, self.p + j] == 1
                        assert (self.db.X[n, 0] == X_full_tensor[i]).item(), (self.db.X[n], X_full_tensor)
                        assert (self.db.X[n, 1] == X_full_tensor[j]).item()
                        n += 1

    def test_batching(self):
        X_train = self.db.X[self.db.train_idx].detach().numpy()
        X_epoch = []
        while len(X_epoch) < len(X_train):
            _, _, X_batch = self.db.load_data()
            X_batch = X_batch.numpy().tolist()
            X_epoch += X_batch
        X_epoch = np.array(X_epoch)
        assert (X_train == X_epoch).all(), (X_train, X_epoch)
        

if __name__ == '__main__':
    unittest.main()
