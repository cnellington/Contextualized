""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch
from contextualized.easy import (
    ContextualizedBayesianNetworks,
    ContextualizedCorrelationNetworks,
    ContextualizedMarkovNetworks,
)


class TestEasyNetworks(unittest.TestCase):
    """
    Test Easy Network models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _quicktest(self, model, C, X, **kwargs):
        print(f"{type(model)} quicktest")
        model.fit(C, X, max_epochs=0)
        err_init = model.measure_mses(C, X)
        model.fit(C, X, **kwargs)
        err_trained = model.measure_mses(C, X)
        W_pred = model.predict_networks(C)
        assert W_pred.shape == (C.shape[0], X.shape[1], X.shape[1])
        assert np.mean(err_trained) < np.mean(err_init)

    def setUp(self):
        """
        Shared unit test setup code.
        """
        self.n_samples = 100
        self.c_dim = 4
        self.x_dim = 5
        C = torch.rand((self.n_samples, self.c_dim)) - 0.5
        # TODO: Use graph utils to generate X from a network.
        X = torch.rand((self.n_samples, self.x_dim)) - 0.5
        self.C, self.X = C.numpy(), X.numpy()

    def test_markov(self):
        """Test Case for ContextualizedMarkovNetworks."""
        model = ContextualizedMarkovNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedMarkovNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedMarkovNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        omegas = model.predict_precisions(self.C, individual_preds=False)
        assert np.shape(omegas) == (self.n_samples, self.x_dim, self.x_dim)

    def test_bayesian(self):
        """ Test case for ContextualizedBayesianNetworks."""
        model = ContextualizedBayesianNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedBayesianNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)

    def test_correlation(self):
        """Test Case for ContextualizedCorrelationNetworks."""

        model = ContextualizedCorrelationNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedCorrelationNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedCorrelationNetworks(
            encoder_type="ngam", num_archetypes=16
        )
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        rho = model.predict_correlation(self.C, squared=False)
        assert np.min(rho) < 0
        rho_squared = model.predict_correlation(self.C, squared=True)
        assert np.min(rho_squared) >= 0


if __name__ == "__main__":
    unittest.main()
