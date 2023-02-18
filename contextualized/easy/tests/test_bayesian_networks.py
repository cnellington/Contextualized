""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch
from contextualized.easy import ContextualizedBayesianNetworks
from contextualized.easy.tests.test_networks import TestEasyNetworks


class TestContextualizedBayesianNetworks(TestEasyNetworks):
    """
    Test Contextualized Bayesian Network models.
    """

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

    def test_bayesian(self):
        """Test case for ContextualizedBayesianNetworks."""
        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16, num_factors=2)
        model.fit(self.C, self.X, max_epochs=10)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)
        networks = model.predict_networks(self.C, factors=True)
        assert np.shape(networks) == (self.n_samples, 2, 2)
        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16, num_factors=2)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedBayesianNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5)

        model = ContextualizedBayesianNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)


if __name__ == "__main__":
    unittest.main()
