""""
Unit tests for Easy Regressor.
"""

import unittest
import numpy as np
import torch

from contextualized.easy import ContextualizedRegressor
from contextualized.utils import DummyParamPredictor, DummyYPredictor


class TestEasyRegression(unittest.TestCase):
    """
    Test Easy Regression models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _quicktest(self, model, C, X, Y, **kwargs):
        print(f"{type(model)} quicktest")
        model.fit(C, X, Y, max_epochs=0)
        err_init = np.linalg.norm(Y - model.predict(C, X), ord=2)
        model.fit(C, X, Y, **kwargs)
        beta_preds, mu_preds = model.predict_params(C)
        assert beta_preds.shape == (X.shape[0], Y.shape[1], X.shape[1])
        assert mu_preds.shape == (X.shape[0], Y.shape[1])
        y_preds = model.predict(C, X)
        assert y_preds.shape == Y.shape
        err_trained = np.linalg.norm(Y - y_preds, ord=2)
        assert err_trained < err_init
        print(err_trained, err_init)

    def test_regressor(self):
        """ Test Case for ContextualizedRegressor.
        """
        n_samples = 1000
        c_dim = 2
        x_dim = 3
        y_dim = 2
        C = torch.rand((n_samples, c_dim)) - 0.5
        beta_1 = C.sum(axis=1).unsqueeze(-1) ** 2
        beta_2 = -C.sum(axis=1).unsqueeze(-1)
        b_1 = C[:, 0].unsqueeze(-1)
        b_2 = C[:, 1].unsqueeze(-1)
        X = torch.rand((n_samples, x_dim)) - 0.5
        outcome_1 = X[:, 0].unsqueeze(-1) * beta_1 + b_1
        outcome_2 = X[:, 1].unsqueeze(-1) * beta_2 + b_2
        Y = torch.cat((outcome_1, outcome_2), axis=1)

        C, X, Y = C.numpy(), X.numpy(), Y.numpy()

        # Naive Multivariate
        parambase = DummyParamPredictor((y_dim, x_dim), (y_dim, 1))
        ybase = DummyYPredictor((y_dim, 1))
        model = ContextualizedRegressor(
            base_param_predictor=parambase, base_y_predictor=ybase
        )
        self._quicktest(model, C, X, Y, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedRegressor(num_archetypes=0)
        self._quicktest(model, C, X, Y, max_epochs=10)

        model = ContextualizedRegressor(num_archetypes=4)
        self._quicktest(model, C, X, Y, max_epochs=10)

        # With regularization
        model = ContextualizedRegressor(
            num_archetypes=4, alpha=1e-1, l1_ratio=0.5, mu_ratio=0.1
        )
        self._quicktest(model, C, X, Y, max_epochs=10)

        # With bootstrap
        model = ContextualizedRegressor(
            num_archetypes=4, alpha=1e-1, l1_ratio=0.5, mu_ratio=0.1
        )
        self._quicktest(
            model, C, X, Y, max_epochs=10, n_bootstraps=2, learning_rate=1e-3
        )


if __name__ == "__main__":
    unittest.main()
