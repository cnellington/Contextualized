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
        k, p, c, k_n = 10, 4, 4, 10
        k_arch = k * p ** 2
        epochs = 5
        runs = 5
        # Test with archetypes
        arch_converges = np.zeros(runs)
        noarch_converges = np.zeros(runs)
        for i in range(runs):
            sim = GaussianSimulator(p, k, c)
            C_train, X_train = sim.gen_samples(k_n)
            C_test, X_test = sim.gen_samples(k_n)
            test_set = (C_test, X_test, X_test)
            c_dim, x_dim, y_dim = C_train.shape[-1], X_train.shape[-1], X_train.shape[-1]
            # Test convergence with archetypes
            arch_model = ContextualCorrelator(c_dim, x_dim, y_dim, num_archetypes=k_arch)
            init_loss = arch_model.get_mse(C_test, X_test, X_test)
            arch_model.fit(C_train, X_train, X_train, epochs=epochs, batch_size=1, validation_set=test_set)
            stop_loss = arch_model.get_mse(C_test, X_test, X_test)
            arch_converges[i] = stop_loss < init_loss
            # Test convergence without archetypes
            noarch_model = ContextualCorrelator(c_dim, x_dim, y_dim, num_archetypes=None)
            init_loss = noarch_model.get_mse(C_test, X_test, X_test)
            noarch_model.fit(C_train, X_train, X_train, epochs=epochs, batch_size=1, validation_set=test_set)
            stop_loss = noarch_model.get_mse(C_test, X_test, X_test)
            noarch_converges[i] = stop_loss < init_loss
        assert arch_converges.all()
        assert noarch_converges.all()

    def test_predict(self):
        k, p, c, k_n = 1, 3, 1, 100
        k_arch = 2 * k * p ** 2
        sim = GaussianSimulator(p, k, c)
        true_sigma = sim.sigmas[0]
        true_vars_tiled = np.tile(true_sigma.diagonal(), (p, 1)).T
        true_betas = true_sigma / true_vars_tiled  # beta[i,j] = beta_{i-->j}
        true_rhos = np.power(true_sigma, 2) / (true_vars_tiled * true_vars_tiled.T)
        C_train, X_train = sim.gen_samples(k_n)
        C_test, X_test = sim.gen_samples(1)
        c_dim, x_dim, y_dim = C_train.shape[-1], X_train.shape[-1], X_train.shape[-1]
        model = ContextualCorrelator(c_dim, x_dim, y_dim, num_archetypes=k_arch)
        model.fit(C_train, X_train, X_train, epochs=100, batch_size=1, validation_set=(C_test, X_test, X_test), es_patience=100, es_epoch=0)
        betas, mus = model.predict_regression(C_test)
        rhos = model.predict_correlation(C_test)
#         print(rhos)
#         print()
#         print(true_rhos)
#         print()
#         print()
#         print(betas)
#         print()
#         print(true_betas)
        assert (rhos == np.transpose(rhos, axes=(0, 2, 1))).all()
        # todo: require exact convergence in the single archetype case
        # assert np.allclose(rhos[0] == true_rhos, atol=1e-3)

    def test_bootstrap_predict(self):
        k, p, c, k_n = 1, 3, 1, 100
        k_arch = 2 * k * p ** 2
        bootstraps = 3
        sim = GaussianSimulator(p, k, c)
        true_sigma = sim.sigmas[0]
        true_vars_tiled = np.tile(true_sigma.diagonal(), (p, 1)).T
        true_betas = true_sigma / true_vars_tiled  # beta[i,j] = beta_{i-->j}
        true_rhos = np.power(true_sigma, 2) / (true_vars_tiled * true_vars_tiled.T)
        C_train, X_train = sim.gen_samples(k_n)
        C_test, X_test = sim.gen_samples(1)
        c_dim, x_dim, y_dim = C_train.shape[-1], X_train.shape[-1], X_train.shape[-1]
        model = ContextualCorrelator(c_dim, x_dim, y_dim, num_archetypes=k_arch, bootstraps=bootstraps)
        model.fit(C_train, X_train, X_train, epochs=100, batch_size=1, validation_set=(C_test, X_test, X_test), es_patience=100, es_epoch=5)
        betas, mus = model.predict_regression(C_test)
        rhos = model.predict_correlation(C_test)
#         print(rhos)
#         print()
#         print(true_rhos)
#         print()
#         print()
#         print(betas)
#         print()
#         print(true_betas)
        betas, mus = model.predict_regression(C_test, all_bootstraps=True)
        rhos = model.predict_correlation(C_test, all_bootstraps=True)
        assert betas.shape == (len(C_test), x_dim, y_dim, bootstraps)
        assert mus.shape == (len(C_test), x_dim, y_dim, bootstraps)
        assert rhos.shape == (len(C_test), x_dim, y_dim, bootstraps)



if __name__ == '__main__':
    unittest.main()
