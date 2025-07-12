"""
This class contains tools for solving context-specific regression problems:

Y = g(beta(C)*X + mu(C))

C: Context
X: Explainable features
Y: Outcome, aka response (regreession) or labels (classification)
g: Link Function for contextualized generalized linear models.

Implemented with PyTorch Lightning
"""

from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as pl

from contextualized.regression.regularizers import REGULARIZERS
from contextualized.regression.losses import MSE
from contextualized.functions import LINK_FUNCTIONS

from contextualized.regression.metamodels import (
    NaiveMetamodel,
    SubtypeMetamodel,
    MultitaskMetamodel,
    TasksplitMetamodel,
    SINGLE_TASK_METAMODELS,
    MULTITASK_METAMODELS,
)
from contextualized.regression.datasets import (
    DataIterable,
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)


class ContextualizedRegressionBase(pl.LightningModule):
    """
    Abstract class for Contextualized Regression.
    """

    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        univariate=False,
        num_archetypes=10,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        metamodel_type="subtype",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
        base_y_predictor=None,
        base_param_predictor=None,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.metamodel_type = metamodel_type
        self.fit_intercept = fit_intercept
        self.link_fn = LINK_FUNCTIONS[link_fn]
        if loss_fn == "mse":
            self.loss_fn = MSE
        else:
            raise ValueError("Supported loss_fn's: mse")
        self.model_regularizer = REGULARIZERS[model_regularizer]
        self.base_y_predictor = base_y_predictor
        self.base_param_predictor = base_param_predictor
        self._build_metamodel(
            context_dim, 
            x_dim,
            y_dim,
            univariate,
            num_archetypes,
            encoder_type,
            encoder_kwargs,
            **kwargs,
        )

    @abstractmethod
    def _build_metamodel(
        self, 
        context_dim, 
        x_dim,
        y_dim,
        univariate,
        num_archetypes,
        encoder_type,
        encoder_kwargs,
        **kwargs
    ):
        """

        :param *args:
        :param **kwargs:

        """
        # builds the metamodel
        self.metamodel = SINGLE_TASK_METAMODELS[self.metamodel_type](
            context_dim, 
            x_dim,
            y_dim,
            univariate,
            num_archetypes,
            encoder_type,
            encoder_kwargs,
            **kwargs
        )

    @abstractmethod
    def dataloader(self, C, X, Y, batch_size=32):
        """

        :param C:
        :param X:
        :param Y:
        :param batch_size:  (Default value = 32)

        """
        # returns the dataloader for this class

    @abstractmethod
    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        # MSE loss by default

    @abstractmethod
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """

        :param batch:
        :param batch_idx:
        :param dataload_idx:

        """
        # returns predicted params on the given batch

    @abstractmethod
    def _params_reshape(self, beta_preds, mu_preds, dataloader):
        """

        :param beta_preds:
        :param mu_preds:
        :param dataloader:

        """
        # reshapes the batch parameter predictions into beta (y_dim, x_dim)

    @abstractmethod
    def _y_reshape(self, y_preds, dataloader):
        """

        :param y_preds:
        :param dataloader:

        """
        # reshapes the batch y predictions into a desirable format

    def forward(self, *args, **kwargs):
        """

        :param *args:

        """
        beta, mu = self.metamodel(*args)
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        if self.base_param_predictor is not None:
            base_beta, base_mu = self.base_param_predictor.predict_params(*args)
            beta = beta + base_beta.to(beta.device)
            mu = mu + base_mu.to(mu.device)
        return beta, mu

    def configure_optimizers(self):
        """
        Set up optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({"test_loss": loss})
        return loss

    def _predict_from_models(self, X, beta_hat, mu_hat):
        """

        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        return self.link_fn((beta_hat * X).sum(axis=-1).unsqueeze(-1) + mu_hat)

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        if self.base_y_predictor is not None:
            Y_base = self.base_y_predictor.predict_y(C, X)
            Y = Y + Y_base.to(Y.device)
        return Y

    def _dataloader(self, C, X, Y, dataset_constructor, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param dataset_constructor:
        :param **kwargs:

        """
        kwargs["num_workers"] = kwargs.get("num_workers", 0)
        kwargs["batch_size"] = kwargs.get("batch_size", 32)
        return DataLoader(dataset=DataIterable(dataset_constructor(C, X, Y)), **kwargs)


class NaiveContextualizedRegression(ContextualizedRegressionBase):
    """See NaiveMetamodel"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        kwargs["univariate"] = False
        self.metamodel = NaiveMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, X, Y, _ = batch
        beta_hat, mu_hat = self.predict_step(batch, batch_idx)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, _, _, _ = batch
        beta_hat, mu_hat = self(C)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, MultivariateDataset, **kwargs)


class ContextualizedRegression(ContextualizedRegressionBase):
    """Supports SubtypeMetamodel and NaiveMetamodel, see selected metamodel for docs"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        self.metamodel = SINGLE_TASK_METAMODELS[self.metamodel_type](*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        (
            C,
            X,
            Y,
        ) = batch
        beta_hat, mu_hat = self.predict_step(batch, batch_idx)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, _, _ = batch
        beta_hat, mu_hat = self(C)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, MultivariateDataset, **kwargs)


class MultitaskContextualizedRegression(ContextualizedRegressionBase):
    """See MultitaskMetamodel"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        kwargs["univariate"] = False
        self.metamodel = MultitaskMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.predict_step(batch, batch_idx)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, _, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, y_idx = data
            betas[n_idx, y_idx] = beta_hats
            mus[n_idx, y_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, y_idx = data
            ys[n_idx, y_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, MultitaskMultivariateDataset, **kwargs)


class TasksplitContextualizedRegression(ContextualizedRegressionBase):
    """See TasksplitMetamodel"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        kwargs["univariate"] = False
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.predict_step(batch, batch_idx)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, _, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, y_idx = data
            betas[n_idx, y_idx] = beta_hats
            mus[n_idx, y_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, y_idx = data
            ys[n_idx, y_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, MultitaskMultivariateDataset, **kwargs)


class ContextualizedUnivariateRegression(ContextualizedRegression):
    """Supports SubtypeMetamodel and NaiveMetamodel, see selected metamodel for docs"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        kwargs["univariate"] = True
        self.metamodel = SINGLE_TASK_METAMODELS[self.metamodel_type](*args, **kwargs)

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats.squeeze(-1)
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, UnivariateDataset, **kwargs)


class TasksplitContextualizedUnivariateRegression(TasksplitContextualizedRegression):
    """See TasksplitMetamodel"""

    def _build_metamodel(self, *args, **kwargs):
        """

        :param *args:
        :param **kwargs:

        """
        kwargs["univariate"] = True
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self.predict_step(batch, batch_idx)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, T, _, _, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = betas.copy()
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, x_idx, y_idx = data
            betas[n_idx, y_idx, x_idx] = beta_hats.squeeze(-1)
            mus[n_idx, y_idx, x_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        """

        :param preds:
        :param dataloader:

        """
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, x_idx, y_idx = data
            ys[n_idx, y_idx, x_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(
                -1
            )
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        return self._dataloader(C, X, Y, MultitaskUnivariateDataset, **kwargs)


class ContextualizedCorrelation(ContextualizedUnivariateRegression):
    """Using univariate contextualized regression to estimate Pearson's correlation
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)

    def dataloader(self, C, X, Y=None, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        if Y is not None:
            print(
                "Passed a Y, but this is self-correlation between X featuers. Ignoring Y."
            )
        return super().dataloader(C, X, X, **kwargs)


class TasksplitContextualizedCorrelation(TasksplitContextualizedUnivariateRegression):
    """Using multitask univariate contextualized regression to estimate Pearson's correlation
    See TasksplitMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)

    def dataloader(self, C, X, Y=None, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        if Y is not None:
            print(
                "Passed a Y, but this is self-correlation between X featuers. Ignoring Y."
            )
        return super().dataloader(C, X, X, **kwargs)


class ContextualizedNeighborhoodSelection(ContextualizedRegression):
    """Using singletask multivariate contextualized regression to do edge-regression for
    estimating conditional dependencies
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(
        self,
        context_dim,
        x_dim,
        model_regularizer=REGULARIZERS["l1"](1e-3, mu_ratio=0),
        **kwargs,
    ):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(
            context_dim, x_dim, x_dim, model_regularizer=model_regularizer, **kwargs
        )
        self.register_buffer("diag_mask", torch.ones(x_dim, x_dim) - torch.eye(x_dim))

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, _, _, _ = batch
        beta_hat, mu_hat = self(C)
        beta_hat = beta_hat * self.diag_mask.expand(beta_hat.shape[0], -1, -1)
        return beta_hat, mu_hat

    def dataloader(self, C, X, Y=None, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """

        if Y is not None:
            print(
                "Passed a Y, but this is a Markov Graph between X featuers. Ignoring Y."
            )
        return super().dataloader(C, X, X, **kwargs)


class ContextualizedMarkovGraph(ContextualizedRegression):
    """Using singletask multivariate contextualized regression to do edge-regression for
    estimating conditional dependencies
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
        self.register_buffer("diag_mask", torch.ones(x_dim, x_dim) - torch.eye(x_dim))

    def predict_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        C, _, _, _ = batch
        beta_hat, mu_hat = self(C)
        beta_hat = beta_hat + torch.transpose(
            beta_hat, 1, 2
        )  # hotfix to enforce symmetry
        beta_hat = beta_hat * self.diag_mask.expand(beta_hat.shape[0], -1, -1)
        return beta_hat, mu_hat

    def dataloader(self, C, X, Y=None, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """

        if Y is not None:
            print(
                "Passed a Y, but this is a Markov Graph between X featuers. Ignoring Y."
            )
        return super().dataloader(C, X, X, **kwargs)
