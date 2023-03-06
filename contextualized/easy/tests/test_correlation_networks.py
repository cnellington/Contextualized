""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch
from contextualized.easy import ContextualizedCorrelationNetworks
from contextualized.easy.tests.test_networks import TestEasyNetworks


class TestContextualizedCorrelationNetworks(TestEasyNetworks):
    """
    Test Contextualized Correlation Network models.
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

    def test_correlation(self):
        """
        Test Case for ContextualizedCorrelationNetworks.
        """

        model = ContextualizedCorrelationNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5)

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
