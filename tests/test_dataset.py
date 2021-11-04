import unittest
import pdb
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
        C, T, X, Y = self.db.pairwise(self.C_full, self.X_full, self.Y_full)
        assert X.shape == (N,)
        assert Y.shape == (N,)
        assert T.shape == (N, self.p_x + self.p_y)
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

    def test_batching(self):
        # TODO test batching over multiple epochs
        train_idx = self.db.train_idx
        C_train, T_train, X_train, Y_train = self.db.pairwise(self.C_full[train_idx], self.X_full[train_idx], self.Y_full[train_idx])
        C_train = C_train.detach().numpy()
        T_train = T_train.detach().numpy()
        X_train = X_train.detach().numpy()
        Y_train = Y_train.detach().numpy()
        C_epoch, T_epoch, X_epoch, Y_epoch = [], [], [], []
        while len(X_epoch) < len(X_train):
            C_batch, T_batch, X_batch, Y_batch = self.db.load_data()
            C_batch = C_batch.numpy().tolist()
            T_batch = T_batch.numpy().tolist()
            X_batch = X_batch.numpy().tolist()
            Y_batch = Y_batch.numpy().tolist()
            C_epoch += C_batch
            T_epoch += T_batch
            X_epoch += X_batch
            Y_epoch += Y_batch
        dtype = C_train.dtype
        C_epoch = np.array(C_epoch, dtype=dtype)
        T_epoch = np.array(T_epoch, dtype=dtype)
        X_epoch = np.array(X_epoch, dtype=dtype)
        Y_epoch = np.array(Y_epoch, dtype=dtype)
        assert (C_train == C_epoch).all()
        assert (T_train == T_epoch).all()
        assert (X_train == X_epoch).all()
        assert (Y_train == Y_epoch).all()

    def test_get_test(self):
        _, _, X_full, _ = self.db.get_test()
        _, _, X_small, _ = self.db.get_test(batch_size=10)
        _, _, X_end, _ = self.db.get_test(batch_size=10, batch_start=-10)
        for x in X_small:
            assert x in X_full
        for x in X_end:
            assert x in X_full
        

if __name__ == '__main__':
    unittest.main()
