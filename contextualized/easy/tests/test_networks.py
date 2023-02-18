""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch


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
        For subclasses, override this method to set up the test data.
        """
        self.n_samples = 100
        self.c_dim = 4
        self.x_dim = 5
        C = torch.rand((self.n_samples, self.c_dim)) - 0.5
        X = torch.rand((self.n_samples, self.x_dim)) - 0.5
        self.C, self.X = C.numpy(), X.numpy()


if __name__ == "__main__":
    unittest.main()
