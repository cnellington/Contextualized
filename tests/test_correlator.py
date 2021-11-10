import time
import unittest
import numpy as np
import torch

from correlator.correlator import ContextualCorrelator, MSE, DTYPE, DEVICE
from correlator.dataset import Dataset
from correlator.helpers.simulation import GaussianSimulator


class TestCorrelator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelator, self).__init__(*args, **kwargs)

    def test_convergence(self):
        k, p, c, k_n = 4, 8, 4, 10
        k_arch = k * p ** 2
        epochs = 10
        runs = 10
        converges = np.zeros(runs)
        for i in range(10):
            sim = GaussianSimulator(p, k, c)
            C_train, X_train = sim.gen_samples(k_n)
            C_test, X_test = sim.gen_samples(k_n)
            task_shape = (X_train.shape[-1] * 2,)
            model = ContextualCorrelator(C_train.shape, task_shape, num_archetypes=k_arch)
            init_loss = model.get_mse(C_test, X_test, X_test)
            model.fit(C_train, X_train, X_train, epochs=50, batch_size=1)
            stop_loss = model.get_mse(C_test, X_test, X_test)
            converges[i] = stop_loss < init_loss
        assert converges.all()

    def test_predict(self):
        k, p, c, k_n = 4, 8, 4, 10
        k_arch = k * p ** 2
        sim = GaussianSimulator(p, k, c)
        C_train, X_train = sim.gen_samples(k_n)
        C_test, X_test = sim.gen_samples(k_n)
        task_shape = (X_train.shape[-1] * 2,)
        model = ContextualCorrelator(C_train.shape, task_shape, num_archetypes=k_arch)
        model.fit(C_train, X_train, X_train, epochs=50, batch_size=1)
        betas, mus = model.predict_regression(C_test)
        rhos = model.predict_correlation(C_test)
        assert (rhos == np.transpose(rhos, axes=(0, 2, 1))).all()


if __name__ == '__main__':
    unittest.main()
