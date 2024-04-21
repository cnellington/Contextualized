""""
Unit tests for Easy Networks.
"""

import unittest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from contextualized.easy import (
    ContextualizedMarkovNetworks,
    ContextualizedCorrelationNetworks,
    ContextualizedBayesianNetworks,
    ContextualizedRegressor,
    ContextualizedClassifier,
    ContextualGAMClassifier,
    ContextualGAMRegressor,
)
from contextualized.utils import DummyParamPredictor, DummyYPredictor


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


class TestContextualizedMarkovNetworks(TestEasyNetworks):
    """
    Test Contextualized Markov Network models.
    """

    def setUp(self):
        """
        Shared unit test setup code.
        """
        np.random.seed(0)
        torch.manual_seed(0)
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
        self._quicktest(
            model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5
        )

        model = ContextualizedMarkovNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedMarkovNetworks(encoder_type="ngam", num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        omegas = model.predict_precisions(self.C, individual_preds=False)
        assert np.shape(omegas) == (self.n_samples, self.x_dim, self.x_dim)
        omegas = model.predict_precisions(self.C, individual_preds=True)
        assert np.shape(omegas) == (1, self.n_samples, self.x_dim, self.x_dim)


class TestContextualizedCorrelationNetworks(TestEasyNetworks):
    """
    Test Contextualized Correlation Network models.
    """

    def setUp(self):
        """
        Shared unit test setup code.
        """
        np.random.seed(0)
        torch.manual_seed(0)
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
        self._quicktest(
            model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5
        )

        model = ContextualizedCorrelationNetworks(num_archetypes=16)
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

        model = ContextualizedCorrelationNetworks(
            encoder_type="ngam", num_archetypes=16
        )
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        rho = model.predict_correlation(self.C, squared=False)
        assert rho.shape == (1, self.n_samples, self.x_dim, self.x_dim)
        rho = model.predict_correlation(self.C, individual_preds=False, squared=False)
        assert rho.shape == (self.n_samples, self.x_dim, self.x_dim), rho.shape
        rho_squared = model.predict_correlation(self.C, squared=True)
        assert np.min(rho_squared) >= 0
        assert rho_squared.shape == (1, self.n_samples, self.x_dim, self.x_dim)


class TestContextualizedBayesianNetworks(TestEasyNetworks):
    """
    Test Contextualized Bayesian Network models.
    """

    def setUp(self):
        """
        Shared unit test setup code.
        """
        np.random.seed(0)
        torch.manual_seed(0)
        self.n_samples = 100
        self.c_dim = 4
        self.x_dim = 5
        C = torch.rand((self.n_samples, self.c_dim)) - 0.5
        # TODO: Use graph utils to generate X from a network.
        X = torch.rand((self.n_samples, self.x_dim)) - 0.5
        self.C, self.X = C.numpy(), X.numpy()

    def test_bayesian_factors(self):
        """Test case for ContextualizedBayesianNetworks."""
        model = ContextualizedBayesianNetworks(
            encoder_type="ngam", num_archetypes=16, num_factors=2
        )
        model.fit(self.C, self.X, max_epochs=10)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)
        networks = model.predict_networks(self.C, factors=True)
        assert np.shape(networks) == (self.n_samples, 2, 2)
        model = ContextualizedBayesianNetworks(
            encoder_type="ngam", num_archetypes=16, num_factors=2
        )
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

    def test_bayesian_default(self):
        model = ContextualizedBayesianNetworks()
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)

    def test_bayesian_val_split(self):
        model = ContextualizedBayesianNetworks()
        self._quicktest(
            model, self.C, self.X, max_epochs=10, learning_rate=1e-3, val_split=0.5
        )

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
        model = ContextualizedBayesianNetworks(
            archetype_dag_loss_type="DAGMA", num_archetypes=16
        )
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)

        model = ContextualizedBayesianNetworks(
            archetype_dag_loss_type="poly", num_archetypes=16
        )
        self._quicktest(model, self.C, self.X, max_epochs=10, learning_rate=1e-3)
        networks = model.predict_networks(self.C, individual_preds=False)
        assert np.shape(networks) == (self.n_samples, self.x_dim, self.x_dim)


class TestEasyClassifiers(unittest.TestCase):
    """
    Test Easy Classifier models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _quicktest(self, model, C, X, Y, **kwargs):
        print(f"{type(model)} quicktest")
        model.fit(C, X, Y, max_epochs=0)
        y_preds_init = model.predict(C, X)
        y_proba_preds_init = model.predict_proba(C, X)[:, :, 1]
        err_init = (Y != y_preds_init).sum()
        roc_init = roc_auc_score(Y, y_proba_preds_init)
        model.fit(C, X, Y, **kwargs)
        beta_preds, mu_preds = model.predict_params(C)
        assert beta_preds.shape == (X.shape[0], Y.shape[1], X.shape[1])
        assert mu_preds.shape == (X.shape[0], Y.shape[1])
        assert not np.any(np.isnan(beta_preds))
        assert not np.any(np.isnan(mu_preds))
        y_preds = model.predict(C, X)
        y_proba_preds = model.predict_proba(C, X)[:, :, 1]
        assert y_preds.shape == Y.shape
        assert y_proba_preds.shape == Y.shape
        err_trained = (Y != y_preds).sum()
        roc_trained = roc_auc_score(Y, y_proba_preds)
        assert err_trained < err_init
        assert roc_trained > roc_init
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

    def test_gam_classifier(self):
        """Test Case for ContextualGAMClassifier."""

        n_samples = 1000
        c_dim = 100
        x_dim = 3
        y_dim = 1
        C = np.random.uniform(-1, 1, size=(n_samples, c_dim))
        X = np.random.uniform(-1, 1, size=(n_samples, x_dim))
        Y = np.random.binomial(1, 0.5, size=(n_samples, y_dim))

        model = ContextualGAMClassifier(alpha=1e-1, encoder_type="mlp")
        self._quicktest(model, C, X, Y, max_epochs=10, es_patience=float("inf"))


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
        try:
            y_dim = Y.shape[1]
        except IndexError:
            y_dim = 1
        assert beta_preds.shape == (X.shape[0], y_dim, X.shape[1])
        assert mu_preds.shape == (X.shape[0], y_dim)
        y_preds = model.predict(C, X)
        assert y_preds.shape == (len(Y), y_dim)
        err_trained = np.linalg.norm(Y - np.squeeze(y_preds), ord=2)
        assert err_trained < err_init
        print(err_trained, err_init)

    def test_regressor(self):
        """Test Case for ContextualizedRegressor."""
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
        self._quicktest(
            model, C, X, Y, max_epochs=10, learning_rate=1e-3, es_patience=float("inf")
        )

        model = ContextualizedRegressor(num_archetypes=0)
        self._quicktest(model, C, X, Y, max_epochs=10, es_patience=float("inf"))

        model = ContextualizedRegressor(num_archetypes=4)
        self._quicktest(model, C, X, Y, max_epochs=10, es_patience=float("inf"))

        # With regularization
        model = ContextualizedRegressor(
            num_archetypes=4, alpha=1e-1, l1_ratio=0.5, mu_ratio=0.1
        )
        self._quicktest(model, C, X, Y, max_epochs=10, es_patience=float("inf"))

        # With bootstrap
        model = ContextualizedRegressor(
            num_archetypes=4, alpha=1e-1, l1_ratio=0.5, mu_ratio=0.1
        )
        self._quicktest(
            model,
            C,
            X,
            Y,
            max_epochs=10,
            n_bootstraps=2,
            learning_rate=1e-3,
            es_patience=float("inf"),
        )

        # Check smaller Y.
        model = ContextualizedRegressor(
            num_archetypes=4, alpha=1e-1, l1_ratio=0.5, mu_ratio=0.1
        )
        self._quicktest(
            model,
            C,
            X,
            Y[:, 0],
            max_epochs=10,
            n_bootstraps=2,
            learning_rate=1e-3,
            es_patience=float("inf"),
        )

    def test_gam_regressor(self):
        """Test Case for ContextualGAMRegressor."""
        n_samples = 100
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

        model = ContextualGAMRegressor()
        self._quicktest(
            model, C, X, Y, max_epochs=10, learning_rate=1e-3, es_patience=float("inf")
        )


if __name__ == "__main__":
    unittest.main()
