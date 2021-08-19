import unittest
import dataloader
import numpy as np


class TestDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDataloader, self).__init__(*args, **kwargs)
        self.build()

    def build(self):
        self.dl = dataloader.Dataloader(2, 2, 2)
        self.context, self.samples, self.labels, self.distribution_ids = self.dl.simulate(10000)

    def test_labels(self):
        for distribution_id in np.unique(self.distribution_ids):
            idx = self.distribution_ids == distribution_id
            c = self.context[idx][0]
            l = self.labels[idx][0]
            assert (self.context[idx] == c).all()
            assert (self.labels[idx] == l).all()

    def test_contextual_covariance(self):
        for distribution_id in np.unique(self.distribution_ids):
            idx = self.distribution_ids == distribution_id
            task_samples = self.samples[idx]
            empirical_cov = 1 / (task_samples.shape[0] - 1) * task_samples.T @ task_samples
            true_cov = self.labels[idx][0]
            assert np.allclose(true_cov, empirical_cov, atol=1e-2)

    def test_load(self):
        C, Ti, Tj, Xi, Xj, sample_ids = self.dl.split_tasks(self.context, self.samples)
        for c, ti, tj, xi, xj, sample_id in zip(C, Ti, Tj, Xi, Xj, sample_ids):
            assert (self.context[sample_id] == c).all()
            assert self.samples[sample_id, int(ti)] == xi
            assert self.samples[sample_id, int(tj)] == xj


if __name__ == '__main__':
    unittest.main()
