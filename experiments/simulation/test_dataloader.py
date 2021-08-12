import unittest
import dataloader
import numpy as np


class TestDataloader(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDataloader, self).__init__(*args, **kwargs)
        self.build()

    def build(self):
        self.dl = dataloader.Dataloader(2, 2, 2)
        self.context, self.samples, self.labels, self.task_ids = self.dl.simulate(10000)

    def test_labels(self):
        for task_id in np.unique(self.task_ids):
            idx = self.task_ids == task_id
            c = self.context[idx][0]
            l = self.labels[idx][0]
            assert (self.context[idx] == c).all()
            assert (self.labels[idx] == l).all()

    def test_contextual_covariance(self):
        for task_id in np.unique(self.task_ids):
            idx = self.task_ids == task_id
            task_samples = self.samples[idx]
            empirical_cov = 1 / (task_samples.shape[0] - 1) * task_samples.T @ task_samples
            true_cov = self.labels[idx][0]
            assert np.allclose(true_cov, empirical_cov, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
