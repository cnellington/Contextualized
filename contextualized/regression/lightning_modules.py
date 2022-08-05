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
import pytorch_lightning as pl

from contextualized.regression import LINK_FUNCTIONS, LOSSES, REGULARIZERS
from contextualized.regression.metamodels import NaiveMetamodel, SubtypeMetamodel, MultitaskMetamodel, TasksplitMetamodel
from contextualized.regression.datasets import DataIterable, MultivariateDataset, UnivariateDataset, MultitaskMultivariateDataset, MultitaskUnivariateDataset, distributed_worker_init_fn

class ContextualizedRegressionBase(pl.LightningModule):
    def __init__(self, *args, learning_rate=1e-3, link_fn=LINK_FUNCTIONS['identity'],
                 loss_fn=LOSSES['mse'], model_regularizer=REGULARIZERS['none'], 
                 base_y_predictor=None, base_param_predictor=None, 
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.link_fn = link_fn
        self.loss_fn = loss_fn
        self.model_regularizer = model_regularizer
        self.base_y_predictor = base_y_predictor
        self.base_param_predictor = base_param_predictor
        self._build_metamodel(*args, **kwargs)

    @abstractmethod
    def _build_metamodel(*args, **kwargs):
        # builds the metamodel
        pass

    @abstractmethod
    def dataloader(self, C, X, Y, batch_size=32):
        # returns the dataloader for this class
        pass

    @abstractmethod
    def _batch_loss(self, batch, batch_idx):
        # MSE loss by default
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx):
        # returns predicted params on the given batch
        pass

    @abstractmethod
    def _params_reshape(self, beta_preds, mu_preds, dataloader):
        # reshapes the batch parameter predictions into beta (y_dim, x_dim)
        pass

    @abstractmethod
    def _y_reshape(self, y_preds, dataloader):
        # reshapes the batch y predictions into a desirable format
        pass

    def forward(self, *args):
        beta, mu = self.metamodel(*args)
        if self.base_param_predictor is not None:
            base_beta, base_mu = self.base_param_predictor.predict_params(*args)
            beta = beta + base_beta
            mu = mu + base_mu
        return beta, mu

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'val_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({'test_loss': loss})
        return loss

    def _predict_from_models(self, X, beta_hat, mu_hat):
        return self.link_fn((beta_hat * X).sum(axis=-1).unsqueeze(-1) + mu_hat)

    def _predict_y(self, C, X, beta_hat, mu_hat):
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        if self.base_y_predictor is not None:
            Y = Y + self.base_y_predictor.predict_y(C, X)
        return Y
    
    def _dataloader(self, C, X, Y, dataset_constructor, **kwargs):
        kwargs['num_workers'] = kwargs.get('num_workers', 0)
        kwargs['batch_size'] = kwargs.get('batch_size', 32)
        return DataLoader(dataset=DataIterable(dataset_constructor(C, X, Y)), worker_init_fn=distributed_worker_init_fn, **kwargs)


class NaiveContextualizedRegression(ContextualizedRegressionBase):
    """
    See NaiveMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = NaiveMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _ = batch
        beta_hat, mu_hat = self(C)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss  = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        C, X, Y, _ = batch
        beta_hat, mu_hat = self(C)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, MultivariateDataset, **kwargs)


class ContextualizedRegression(ContextualizedRegressionBase):
    """
    See SubtypeMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = SubtypeMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _, = batch
        beta_hat, mu_hat = self(C)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss  = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        C, X, Y, _ = batch
        beta_hat, mu_hat = self(C)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, MultivariateDataset, **kwargs)


class MultitaskContextualizedRegression(ContextualizedRegressionBase):
    """
    See MultitaskMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = MultitaskMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss  = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, y_idx = data
            betas[n_idx, y_idx] = beta_hats
            mus[n_idx, y_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, y_idx = data
            ys[n_idx, y_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, MultitaskMultivariateDataset, **kwargs)


class TasksplitContextualizedRegression(ContextualizedRegressionBase):
    """
    See TasksplitMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss  = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, y_idx = data
            betas[n_idx, y_idx] = beta_hats
            mus[n_idx, y_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, y_idx = data
            ys[n_idx, y_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, MultitaskMultivariateDataset, **kwargs)


class ContextualizedUnivariateRegression(ContextualizedRegression):
    """
    See SubtypeMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = True
        self.metamodel = SubtypeMetamodel(*args, **kwargs)

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, n_idx = data
            betas[n_idx] = beta_hats.squeeze(-1)
            mus[n_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, X, _, n_idx = data
            ys[n_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, UnivariateDataset, **kwargs)


class TasksplitContextualizedUnivariateRegression(ContextualizedRegressionBase):
    """
    See TasksplitMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = True
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        pred_loss = self.loss_fn(Y, self._predict_y(C, X, beta_hat, mu_hat))
        reg_loss  = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat

    def _params_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        mus = betas.copy()
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, x_idx, y_idx = data
            betas[n_idx, y_idx, x_idx] = beta_hats.squeeze(-1)
            mus[n_idx, y_idx, x_idx] = mu_hats.squeeze(-1)
        return betas, mus

    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            C, _, X, _, n_idx, x_idx, y_idx = data
            ys[n_idx, y_idx, x_idx] = self._predict_y(C, X, beta_hats, mu_hats).squeeze(-1)
        return ys

    def dataloader(self, C, X, Y, **kwargs):
        return self._dataloader(C, X, Y, MultitaskUnivariateDataset, **kwargs)


class ContextualizedCorrelation(ContextualizedUnivariateRegression):
    """
    Using univariate contextualized regression to estimate Pearson's correlation
    See SubtypeMetamodel for assumptions and full docstring
    """
    def __init__(self, context_dim, x_dim, **kwargs):
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
    
    def dataloader(self, C, X, **kwargs):
        return super().dataloader(C, X, X, **kwargs)


class TasksplitContextualizedCorrelation(TasksplitContextualizedUnivariateRegression):
    """
    Using multitask univariate contextualized regression to estimate Pearson's correlation
    See TasksplitMetamodel for assumptions and full docstring
    """
    def __init__(self, context_dim, x_dim, **kwargs):
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
    
    def dataloader(self, C, X, **kwargs):
        return super().dataloader(C, X, X, **kwargs)
