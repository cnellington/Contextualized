""""
Unit tests for Easy Classifier.
"""

import unittest
import numpy as np

from contextualized.easy import ContextualizedClassifier


class TestEasyClassifier(unittest.TestCase):
    """
    Test Easy Classifier models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _quicktest(self, model, C, X, Y, **kwargs):
        print(f"{type(model)} quicktest")
        model.fit(C, X, Y, max_epochs=0)
        err_init = (Y != model.predict(C, X)).sum()
        model.fit(C, X, Y, **kwargs)
        beta_preds, mu_preds = model.predict_params(C)
        assert beta_preds.shape == (X.shape[0], Y.shape[1], X.shape[1])
        assert mu_preds.shape == (X.shape[0], Y.shape[1])
        assert not np.any(np.isnan(beta_preds))
        assert not np.any(np.isnan(mu_preds))
        y_preds = model.predict(C, X)
        assert y_preds.shape == Y.shape
        err_trained = (Y != y_preds).sum()
        assert err_trained < err_init
        print(err_trained, err_init)

    def test_classifier(self):
        """Test Case for ContextualizedClassifier."""

        n_samples = 1000
        c_dim = 100
        x_dim = 3
        y_dim = 1
        C = np.random.uniform(-1, 1, size=(n_samples, c_dim))
        X = np.random.uniform(-1, 1, size=(n_samples, x_dim))
        Y = np.random.binomial(1, 0.5, size=(n_samples, y_dim))

        model = ContextualizedClassifier(alpha=1e-1, encoder_type="mlp")
        self._quicktest(model, C, X, Y, max_epochs=10, es_patience=float("inf"))


if __name__ == "__main__":
    unittest.main()
