import unittest
import pdb
import numpy as np
import torch

from correlator.dataset import Dataset, to_pairwise
from correlator.helpers.simulation import GaussianSimulator


class TestGaussianSimulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGaussianSimulator, self).__init__(*args, **kwargs)
        self.p = 10
        self.c = 5
        self.k = 5
        self.sim = GaussianSimulator(self.p, self.k, self.c)
    
    def test_empirical_cov(self):
        k_n = int(1e5)
        _, X = self.sim.gen_samples(k_n)
        for i in range(self.k):
            X_sample = X[i * k_n:(i+1) * k_n]
            empirical_cov = 1 / (k_n - 1) * X_sample.T @ X_sample
            assert np.allclose(empirical_cov, self.sim.sigmas[i], atol=1e-1)


class TestDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDataset, self).__init__(*args, **kwargs)
        self.dtype = torch.float
        self.p_x = 7
        self.p_y = 3
        self.c = 5
        self.k = 5
        self.k_n = 10
        self.sim = GaussianSimulator(self.p_x + self.p_y, self.k, self.c)
        self.C_full, XY_full = self.sim.gen_samples(self.k_n)
        self.X_full = XY_full[:,:self.p_x]
        self.Y_full = XY_full[:,self.p_x:]
        self.db = Dataset(self.C_full, self.X_full, self.Y_full, dtype=self.dtype)

    def test_taskpairs(self):
        N = self.k * self.k_n * self.p_x * self.p_y
        C, T, X, Y = to_pairwise(self.C_full, self.X_full, self.Y_full)
        assert X.shape == (N,)
        assert Y.shape == (N,)
        assert T.shape == (N, max(self.p_x, self.p_y) * 2)
        assert C.shape == (N, self.c)
        n = 0
        for k in range(self.k):
            for m in range(self.k_n):
                for i in range(self.p_x):
                    for j in range(self.p_y):
                        C_tensor = torch.tensor(self.sim.contexts[k], dtype=self.dtype)
                        m_full = k * self.k_n + m
                        X_full_tensor = torch.tensor(self.X_full[m_full], dtype=self.dtype)
                        Y_full_tensor = torch.tensor(self.Y_full[m_full], dtype=self.dtype)
                        assert (C[n] == C_tensor).all()
                        assert T[n, i] == 1
                        assert T[n, self.p_x + j] == 1
                        assert (X[n] == X_full_tensor[i]).item()
                        assert (Y[n] == Y_full_tensor[j]).item()
                        n += 1

    def test_load_data(self):
        _, _, X_full, _ = self.db.load_data()
        _, _, X_small, _ = self.db.load_data(batch_size=10)
        _, _, X_end, _ = self.db.load_data(batch_size=10, batch_start=-10)
        for x in X_small:
            assert x in X_full
        assert (X_full[-10 * self.p_x * self.p_y:].numpy() == X_end.numpy()).all()
        

if __name__ == '__main__':
    unittest.main()
