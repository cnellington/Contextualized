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

    def test_bayesian_factors(self):
        """Test case for ContextualizedBayesianNetworks."""
        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16, num_factors=2)
        model.fit(self.C, self.X, max_epochs=10)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)
        networks = model.predict_networks(self.C, factors=True)
        assert np.shape(networks) == (self.n_samples, 2, 2)
        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16, num_factors=2)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

    def test_bayesian_default(self):
        model = ContextualizedBayesianNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

    def test_bayesian_val_split(self):
        model = ContextualizedBayesianNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5)

    def test_bayesian_archetypes(self):
        model = ContextualizedBayesianNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

    def test_bayesian_encoder(self):
        model = ContextualizedBayesianNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)

        model = ContextualizedBayesianNetworks(encoder_type="mlp", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)

    def test_bayesian_acyclicity(self):
        model = ContextualizedBayesianNetworks(archetype_dag_loss_type="DAGMA", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)

        model = ContextualizedBayesianNetworks(archetype_dag_loss_type="poly", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)


if __name__ == "__main__":
    unittest.main()
