"""
Unit tests for Contextualized Regression.
"""

import unittest
import numpy as np
import torch


# from contextualized.modules import NGAM, MLP, SoftSelect, Explainer
from contextualized.regression.lightning_modules import *
from contextualized.regression.trainers import *
from contextualized.regression.datamodules import RegressionDataModule
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

    def _quicktest(
        self,
        model,
        univariate=False,
        correlation=False,
        markov=False,
        dataclass_type="dataloader",
    ):
        """
        :param model:
        :param univariate:  (Default value = False)
        :param correlation:  (Default value = False)
        :param markov:  (Default value = False)
        :param dm_pred_dl_type: Dataset to use for predict (Default value = 'Full', choose from 'test', 'train', 'val', 'full')

        """

        print(f"\n{type(model)} quicktest")

        get_dataclass = {
            "datamodule": lambda x, **kwargs: x.datamodule(**kwargs),
            "dataloader": lambda x, **kwargs: x.dataloader(**kwargs),
        }

        # get dataclass & trainer
        if correlation:
            dataclass = get_dataclass[dataclass_type](
                model,
                C=self.C,
                X=np.hstack((self.X, self.Y)),
                batch_size=self.batch_size,
            )
            trainer = CorrelationTrainer(max_epochs=self.epochs)

        elif markov:
            # use concatenated X,Y for X -> Y prediction
            dataclass = get_dataclass[dataclass_type](
                model,
                C=self.C,
                X=np.hstack((self.X, self.Y)),
                batch_size=self.batch_size,
            )
            trainer = MarkovTrainer(max_epochs=self.epochs)
        else:
            dataclass = get_dataclass[dataclass_type](
                model, C=self.C, X=self.X, Y=self.Y, batch_size=self.batch_size
            )
            trainer = RegressionTrainer(max_epochs=self.epochs, univariate=univariate)

        # train / eval models
        if type(dataclass_type) in (
            pl.LightningDataModule,
            RegressionDataModule,
        ):  # datamodule
            err_init = {}
            err_trained = {}
            # pre-train mse/preds
            for p_type in ["train", "test", "val", "full"]:
                y_preds = trainer.predict_y(model, dataclass, dm_pred_type=p_type)

            # train
            trainer.fit(model, dataclass)
            trainer.validate(model, dataclass)
            trainer.test(model, dataclass)

            # post-train predictions
            for p_type in ["train", "test", "val", "full"]:
                y_preds = trainer.predict_y(model, dataclass, p_type)

                beta_preds, mu_preds = trainer.predict_params(model, dataclass, p_type)

                if correlation:
                    rhos = trainer.predict_correlation(model, dataclass, p_type)
                if markov:
                    omegas = trainer.predict_precision(model, dataclass, p_type)

                err_trained[p_type] = ((y_true - y_preds) ** 2).mean()

                assert (
                    err_trained[p_type] < err_init[p_type]
                ), "Model failed to converge"

        else:  # dataloader
            # pre-train mse/preds
            y_preds = trainer.predict_y(model=model, dataclass=dataclass)
            mse_pre = trainer.measure_mses(
                model, dataclass, dm_pred_type="test", individual_preds=False
            )

            # train
            trainer.fit(model, dataclass)
            trainer.validate(model, dataclass)
            trainer.test(model, dataclass)

            # post-train mse/preds
            y_preds = trainer.predict_y(model, dataclass)
            mse_post = trainer.measure_mses(
                model, dataclass, dm_pred_type="test", individual_preds=False
            )

            beta_preds, mu_preds = trainer.predict_params(model, dataclass)

            if correlation:
                rhos = trainer.predict_correlation(model, dataclass)
            if markov:
                omegas = trainer.predict_precision(model, dataclass)

            assert mse_post < mse_pre, "Model failed to converge"

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        # self._quicktest(model, dataclass="dataloader")

        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        # self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, dataclass_type="dataloader")
        self._quicktest(model, dataclass_type="datamodule")

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
        self._quicktest(model, univariate=True, dataclass_type="dataloader")
        self._quicktest(model, univariate=True, dataclass_type="datamodule")

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
        self._quicktest(model, univariate=True, dataclass_type="dataloader")
        self._quicktest(model, univariate=True, dataclass_type="datamodule")

    def test_correlation_subtype(self):
        """
        Test correlation.
        """
        # Correlation
        parambase = DummyParamPredictor(
            (self.x_dim + self.y_dim, self.x_dim + self.y_dim, 1),
            (self.x_dim + self.y_dim, self.x_dim + self.y_dim, 1),
        )
        ybase = DummyYPredictor((self.x_dim + self.y_dim, self.x_dim + self.y_dim, 1))
        model = ContextualizedCorrelation(
            self.c_dim,
            self.x_dim + self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, correlation=True, dataclass_type="dataloader")
        self._quicktest(model, correlation=True, dataclass_type="datamodule")

    def test_correlation_tasksplit(self):
        """
        Test task-split correlation.
        """
        # Tasksplit Correlation
        parambase = DummyParamPredictor((1,), (1,))
        ybase = DummyYPredictor((1,))
        model = TasksplitContextualizedCorrelation(
            self.c_dim,
            self.x_dim + self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, correlation=True, dataclass_type="dataloader")
        self._quicktest(model, correlation=True, dataclass_type="datamodule")

    def test_markov_subtype(self):
        """
        Test Markov Graphs.
        """
        # Markov Graph
        parambase = DummyParamPredictor(
            (self.y_dim + self.x_dim, 1), (self.y_dim + self.x_dim, 1)
        )
        ybase = DummyYPredictor((self.y_dim + self.x_dim, 1))
        model = ContextualizedMarkovGraph(
            self.c_dim,
            self.x_dim + self.y_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, markov=True, dataclass_type="dataloader")
        self._quicktest(model, markov=True, dataclass_type="datamodule")

    def test_neighborhood_subtype(self):
        """
        Test Neighborhood Selection.
        """
        parambase = DummyParamPredictor((self.x_dim, self.x_dim), (self.x_dim, 1))
        ybase = DummyYPredictor((self.x_dim, 1))
        model = ContextualizedNeighborhoodSelection(
            self.c_dim,
            self.x_dim,
            base_param_predictor=parambase,
            base_y_predictor=ybase,
        )
        self._quicktest(model, markov=True)


if __name__ == "__main__":
    unittest.main()
