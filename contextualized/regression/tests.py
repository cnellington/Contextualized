"""
Unit tests for Contextualized Regression.
"""
import unittest
import numpy as np
import torch

# from contextualized.modules import NGAM, MLP, SoftSelect, Explainer
from contextualized.regression.lightning_modules import *
from contextualized.regression.trainers import *
from contextualized.functions import LINK_FUNCTIONS
from contextualized.utils import DummyParamPredictor, DummyYPredictor


class TestRegression(unittest.TestCase):
    """
    Test regression models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Shared unit test setup code.
        """
        n = 100
        c_dim = 4
        x_dim = 5
        y_dim = 3
        C = torch.rand((n, c_dim)) - 0.5
        W_1 = C.sum(axis=1).unsqueeze(-1) ** 2
        W_2 = -C.sum(axis=1).unsqueeze(-1)
        b_1 = C[:, 0].unsqueeze(-1)
        b_2 = C[:, 1].unsqueeze(-1)
        # W_full = torch.cat((W_1, W_2), axis=1)
        # b_full = b_1 + b_2
        X = torch.rand((n, x_dim)) - 0.5
        Y_1 = X[:, 0].unsqueeze(-1) * W_1 + b_1
        Y_2 = X[:, 1].unsqueeze(-1) * W_2 + b_2
        Y_3 = X.sum(axis=1).unsqueeze(-1)
        Y = torch.cat((Y_1, Y_2, Y_3), axis=1)

        self.k = 10
        self.epochs = 1
        self.batch_size = 32
        self.c_dim, self.x_dim, self.y_dim = c_dim, x_dim, y_dim
        self.C, self.X, self.Y = C.numpy(), X.numpy(), Y.numpy()

    def _quicktest(self, model, univariate=False, correlation=False, markov=False):
        """

        :param model:
        :param univariate:  (Default value = False)
        :param correlation:  (Default value = False)
        :param markov:  (Default value = False)

        """
        print(f"\n{type(model)} quicktest")
        if correlation:
            dataloader = model.dataloader(self.C, self.X, batch_size=self.batch_size)
            trainer = CorrelationTrainer(max_epochs=self.epochs)
            y_true = np.tile(self.X[:, :, np.newaxis], (1, 1, self.X.shape[-1]))
        elif markov:
            dataloader = model.dataloader(self.C, self.X, batch_size=self.batch_size)
            trainer = MarkovTrainer(max_epochs=self.epochs)
            y_true = self.X
        else:
            dataloader = model.dataloader(
                self.C, self.X, self.Y, batch_size=self.batch_size
            )
            trainer = RegressionTrainer(max_epochs=self.epochs)
            if univariate:
                y_true = np.tile(self.Y[:, :, np.newaxis], (1, 1, self.X.shape[-1]))
            else:
                y_true = self.Y
        y_preds = trainer.predict_y(model, dataloader)
        err_init = ((y_true - y_preds) ** 2).mean()
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        if correlation:
            rhos = trainer.predict_correlation(model, dataloader)
        if markov:
            omegas = trainer.predict_precision(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        err_trained = ((y_true - y_preds) ** 2).mean()
        assert err_trained < err_init, "Model failed to converge"

    def test_naive(self):
        """
        Test Naive Multivariate regression.
        """
        # Naive Multivariate
        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["identity"],
            },
            link_fn=LINK_FUNCTIONS["identity"],
        )
        self._quicktest(model)

        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_type="ngam",
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["identity"],
            },
            link_fn=LINK_FUNCTIONS["identity"],
        )
        self._quicktest(model)

        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["softmax"],
            },
            link_fn=LINK_FUNCTIONS["identity"],
        )
        self._quicktest(model)

        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["identity"],
            },
            link_fn=LINK_FUNCTIONS["logistic"],
        )
        self._quicktest(model)

        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["softmax"],
            },
            link_fn=LINK_FUNCTIONS["logistic"],
        )
        self._quicktest(model)

        parambase = DummyParamPredictor((self.y_dim, self.x_dim), (self.y_dim, 1))
        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["softmax"],
            },
            link_fn=LINK_FUNCTIONS["logistic"],
            base_param_predictor=parambase,
        )
        self._quicktest(model)

        ybase = DummyYPredictor((self.y_dim, 1))
        model = NaiveContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            encoder_kwargs={
                "width": 25,
                "layers": 2,
                "link_fn": LINK_FUNCTIONS["softmax"],
            },
            link_fn=LINK_FUNCTIONS["logistic"],
            base_y_predictor=ybase,
        )
        self._quicktest(model)

    def test_subtype(self):
        """
        Test subtype multivariate regression.
        """
        # Subtype Multivariate
        parambase = DummyParamPredictor((self.y_dim, self.x_dim), (self.y_dim, 1))
        ybase = DummyYPredictor((self.y_dim, 1))
        model = ContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model)

    def test_multitask(self):
        """
        Test multitask multivariate regression.
        """
        # Multitask Multivariate
        parambase = DummyParamPredictor((self.x_dim,), (1,))
        ybase = DummyYPredictor((1,))
        model = MultitaskContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model)

    def test_tasksplit(self):
        """
        Test tasksplit multivariate regression.
        """
        # Tasksplit Multivariate
        parambase = DummyParamPredictor((self.x_dim,), (1,))
        ybase = DummyYPredictor((1,))
        model = TasksplitContextualizedRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model)

    def test_univariate_subtype(self):
        """
        Test naive univariate regression.
        """
        # Naive Univariate
        parambase = DummyParamPredictor(
            (self.y_dim, self.x_dim, 1), (self.y_dim, self.x_dim, 1)
        )
        ybase = DummyYPredictor((self.y_dim, self.x_dim, 1))
        model = ContextualizedUnivariateRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, univariate=True)

    def test_univariate_tasksplit(self):
        """
        Test task-split univariate regression.
        """
        # Tasksplit Univariate
        parambase = DummyParamPredictor((1,), (1,))
        ybase = DummyYPredictor((1,))
        model = TasksplitContextualizedUnivariateRegression(
            self.c_dim,
            self.x_dim,
            self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, univariate=True)

    def test_correlation_subtype(self):
        """
        Test correlation.
        """
        # Correlation
        parambase = DummyParamPredictor(
            (self.x_dim, self.x_dim, 1), (self.x_dim, self.x_dim, 1)
        )
        ybase = DummyYPredictor((self.x_dim, self.x_dim, 1))
        model = ContextualizedCorrelation(
            self.c_dim,
            self.x_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, correlation=True)

    def test_correlation_tasksplit(self):
        """
        Test task-split correlation.
        """
        # Tasksplit Correlation
        parambase = DummyParamPredictor((1,), (1,))
        ybase = DummyYPredictor((1,))
        model = TasksplitContextualizedCorrelation(
            self.c_dim,
            self.x_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, correlation=True)

    def test_markov_subtype(self):
        """
        Test Markov Graphs.
        """
        # Markov Graph
        parambase = DummyParamPredictor((self.x_dim, self.x_dim), (self.x_dim, 1))
        ybase = DummyYPredictor((self.x_dim, 1))
        model = ContextualizedMarkovGraph(
            self.c_dim,
            self.x_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, markov=True)


if __name__ == "__main__":
    unittest.main()
