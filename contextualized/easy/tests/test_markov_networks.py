""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch
from contextualized.easy.tests.test_networks import TestEasyNetworks
from contextualized.easy import ContextualizedMarkovNetworks


class TestContextualizedMarkovNetworks(TestEasyNetworks):
    """
    Test Contextualized Markov Network models.
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

    def test_markov(self):
        """Test Case for ContextualizedMarkovNetworks."""
        model = ContextualizedMarkovNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5)

        model = ContextualizedMarkovNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedMarkovNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        omegas = model.predict_precisions(self.C, individual_preds=False)
        assert np.shape(omegas) == (self.n_samples, self.x_dim, self.x_dim)


if __name__ == "__main__":
    unittest.main()
