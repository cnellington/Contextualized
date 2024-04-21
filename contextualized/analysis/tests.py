"""
Unit tests for analysis utilities.
"""

import unittest
import copy
import torch
import numpy as np
import pandas as pd


from contextualized.analysis import (
    test_each_context,
    select_good_bootstraps,
    calc_heterogeneous_predictor_effects_pvals,
    calc_homogeneous_context_effects_pvals,
    calc_homogeneous_predictor_effects_pvals,
)

from contextualized.easy import ContextualizedRegressor


class TestTestEachContext(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Shared data setup.
        """
        torch.manual_seed(0)
        np.random.seed(0)
        n_samples = 1000
        C = np.random.uniform(0, 1, size=(n_samples, 2))
        X = np.random.uniform(0, 1, size=(n_samples, 2))
        beta = np.concatenate([np.ones((n_samples, 1)), C], axis=1)
        Y = np.sum(
            beta[:, :2] * X, axis=1
        )  # X1 changes effect under C0. C1 has no effect, X0 is constant

        self.C_train_df = pd.DataFrame(C, columns=["C0", "C1"])
        self.X_train_df = pd.DataFrame(X, columns=["X0", "X1"])
        self.Y_train_df = pd.DataFrame(Y, columns=["Y"])

    def test_test_each_context(self):
        """
        Test that the output shape of the test_each_context function is as expected.
        """
        pvals = test_each_context(
            ContextualizedRegressor,
            self.C_train_df,
            self.X_train_df,
            self.Y_train_df,
            model_kwargs={"encoder_type": "mlp", "layers": 1},
            fit_kwargs={"max_epochs": 1, "learning_rate": 1e-2, "n_bootstraps": 10},
        )

        expected_shape = (
            self.C_train_df.shape[1]
            * self.X_train_df.shape[1]
            * self.Y_train_df.shape[1],
            4,
        )
        self.assertEqual(pvals.shape, expected_shape)
        self.assertTrue(all(0 <= pval <= 1 for pval in pvals["Pvals"]))

        pval_c0_x1 = pvals.loc[1, "Pvals"]
        self.assertTrue(pval_c0_x1 < 0.2, "C0 X1 p-value is not significant.")

        other_pvals = pvals.drop(1)
        self.assertTrue(
            all(pval >= 0.2 for pval in other_pvals["Pvals"]),
            "Other p-values are significant.",
        )


class TestSelectGoodBootstraps(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.C = np.random.uniform(0, 1, size=(100, 2))
        self.X = np.random.uniform(0, 1, size=(100, 2))
        self.Y = np.random.uniform(0, 1, size=(100, 2))

    def test_model_has_fewer_bootstraps(self):
        """
        Test that the model has fewer bootstraps after calling select_good_bootstraps.
        """
        model = ContextualizedRegressor(n_bootstraps=3)
        model.fit(self.C, self.X, self.Y)
        Y_pred = model.predict(self.C, self.X, individual_preds=True)
        train_errs = np.zeros_like((self.Y - Y_pred) ** 2)
        train_errs[0] = 0.1
        train_errs[1] = 0.2
        train_errs[2] = 0.3
        model_copy = copy.deepcopy(model)
        select_good_bootstraps(model, train_errs)
        self.assertEqual(len(model.models), 1)
        self.assertEqual(len(model_copy.models), 3)
        self.assertLess(len(model.models), len(model_copy.models))


class TestPvals(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        # X1 is a heterogeneous predictor under C0, X0 is a homogeneous predictor
        # C0 is a homogeneous context predictor on Y0, C1 is a heterogeneous context predictor on Y1
        self.C = np.random.uniform(-1, 1, size=(100, 2))
        self.X = np.random.uniform(-1, 1, size=(100, 2))
        betas = np.concatenate([np.ones((100, 1)), self.C[:, 0, None]], axis=1)
        Y0 = np.sum(betas * self.X, axis=1) + self.C[:, 0]
        Y1 = np.sum(betas * self.X, axis=1) + self.C[:, 1]
        self.Y = np.column_stack([Y0, Y1])

    def test_homogeneous_context_effect_pvals(self):
        model = ContextualizedRegressor(n_bootstraps=10)
        model.fit(self.C, self.X, self.Y)
        pvals = calc_homogeneous_context_effects_pvals(model, self.C)
        assert pvals.shape == (self.C.shape[1], self.Y.shape[1])
        assert pvals[0, 0] < 0.2 and pvals[1, 1] < 0.2
        assert pvals[0, 1] > 0.2 and pvals[1, 0] > 0.2

    def test_homogeneous_predictor_effect_pvals(self):
        model = ContextualizedRegressor(n_bootstraps=10)
        model.fit(self.C, self.X, self.Y)
        pvals = calc_homogeneous_predictor_effects_pvals(model, self.X)
        assert pvals.shape == (self.X.shape[1], self.Y.shape[1])
        assert pvals[0, 0] < 0.2 and pvals[0, 1] < 0.2
        assert pvals[1, 0] > 0.2 and pvals[1, 1] > 0.2

    def test_heterogeneous_predictor_effect_pvals(self):
        model = ContextualizedRegressor(n_bootstraps=10)
        model.fit(self.C, self.X, self.Y)
        pvals = calc_heterogeneous_predictor_effects_pvals(model, self.C)
        assert pvals.shape == (self.C.shape[1], self.X.shape[1], self.Y.shape[1])
        assert pvals[0, 1, 0] < 0.2 and pvals[0, 1, 1] < 0.2
        pvals[0, 1, 0] = pvals[0, 1, 1] = 1
        assert (pvals > 0.2).all()


if __name__ == "__main__":
    unittest.main()
