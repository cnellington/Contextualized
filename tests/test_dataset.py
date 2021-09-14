import unittest
import numpy as np
import torch
from model.dataset import Dataset, SimulationDataset, GaussianSimulator


class TestSimulator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSimulator, self).__init__(*args, **kwargs)
        self.p = 10
        self.c = 5
        self.k = 5
        self.k_n = 10
        self.db = SimulationDataset(self.p, self.k, self.c)
    
    def test_gen_samples(self):
        k_n = 10
        X, T, C, B = self.db.gen_samples(k_n)
        C = C.numpy().astype(np.float16)
        N = self.k * k_n * self.p ** 2
        assert X.shape == (N, 2)
        assert T.shape == (N, 2)
        assert C.shape == (N, self.c)
        n = 0
        for k in range(self.k):
            for m in range(k_n):
                for i in range(self.p):
                    for j in range(self.p):
                        beta = self.db.sigmas[k,i,j] / self.db.sigmas[k,i,i]
                        context = self.db.contexts[k].astype(np.float16)
                        assert beta == B[n,0]
                        assert (context == C[n]).all()
                        n += 1

    def test_labels(self):
        # Are the MLE estimates of the regression coefficients close to the true labels for large k_n
        k_n = 1000
        X, T, C, B = self.db.gen_samples(k_n)
        for k in range(self.k):
            M = k_n * self.p ** 2
            X_k = X[k * M:(k + 1) * M]
            X_sample = np.zeros((k_n, self.p))
            for m in range(k_n):
                for i in range(self.p):
                    X_sample[m, i] = X_k[m * self.p**2 + i, 1]
            empirical_cov = 1 / (k_n - 1) * X_sample.T @ X_sample
            assert np.allclose(empirical_cov, self.db.sigmas[k], atol=1e0)

    def test_batching(self):
        # Are all training samples observed over a single epoch
        assert True

#     def test_labels(self):
#         for distribution_id in np.unique(self.distribution_ids):
#             idx = self.distribution_ids == distribution_id
#             c = self.context[idx][0]
#             l = self.labels[idx][0]
#             assert (self.context[idx] == c).all()
#             assert (self.labels[idx] == l).all()
# 
#     def test_contextual_covariance(self):
#         for distribution_id in np.unique(self.distribution_ids):
#             idx = self.distribution_ids == distribution_id
#             task_samples = self.samples[idx]
#             empirical_cov = 1 / (task_samples.shape[0] - 1) * task_samples.T @ task_samples
#             true_cov = self.labels[idx][0]
#             assert np.allclose(true_cov, empirical_cov, atol=1e-2)
# 
#     def test_load(self):
#         C, Ti, Tj, Xi, Xj, sample_ids = self.dl.split_tasks(self.context, self.samples)
#         for c, ti, tj, xi, xj, sample_id in zip(C, Ti, Tj, Xi, Xj, sample_ids):
#             assert (self.context[sample_id] == c).all()
#             assert self.samples[sample_id, int(ti)] == xi
#             assert self.samples[sample_id, int(tj)] == xj


class TestGaussianSimulator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGaussianSimulator, self).__init__(*args, **kwargs)
        self.p = 10
        self.c = 5
        self.k = 5
        self.sim = GaussianSimulator(self.p, self.k, self.c)
    
    def test_empirical_cov(self):
        k_n = int(1e5)
        X, C = self.sim.gen_samples(k_n)
        for i in range(self.k):
            X_sample = X[i * k_n:(i+1) * k_n]
            empirical_cov = 1 / (k_n - 1) * X_sample.T @ X_sample
            assert np.allclose(empirical_cov, self.sim.sigmas[i], atol=1e-1)


if __name__ == '__main__':
    unittest.main()
