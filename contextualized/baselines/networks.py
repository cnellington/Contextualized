"""
Baseline network models for learning the causal structure of data.
Includes:
    - CorrelationNetwork: learns the correlation matrix of the data
    - MarkovNetwork: learns the Markov blanket of each variable
    - BayesianNetwork: learns the DAG structure of the data

    - GroupedNetworks: learns a separate model for each group of data. This is a wrapper around the above models.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from contextualized.dags.graph_utils import project_to_dag_torch


dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
mse_loss = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)


def dag_loss(w, alpha, rho):
    d = w.shape[-1]
    m = torch.linalg.matrix_exp(w * w)
    h = torch.trace(m) - d
    return alpha * h + 0.5 * rho * h * h


class NOTEARSTrainer(pl.Trainer):
    def predict(self, model, dataloader):
        preds = super().predict(model, dataloader)
        return torch.cat(preds)

    def predict_w(self, model, dataloader, project_to_dag=True):
        preds = self.predict(model, dataloader)
        W = model.W.detach() * model.diag_mask
        if project_to_dag:
            W = torch.tensor(project_to_dag_torch(W.numpy(force=True))[0])
        W_batch = W.unsqueeze(0).expand(len(preds), -1, -1)
        return W_batch.numpy()


class NOTEARS(pl.LightningModule):
    """
    NOTEARS model for learning the DAG structure of data.
    """

    def __init__(self, x_dim, l1=1e-3, alpha=1e-8, rho=1e-8, learning_rate=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.l1 = l1
        self.alpha = alpha
        self.rho = rho
        diag_mask = torch.ones(x_dim, x_dim) - torch.eye(x_dim)
        self.register_buffer("diag_mask", diag_mask)
        init_mat = (torch.rand(x_dim, x_dim) * 2e-2 - 1e-2) * diag_mask
        self.W = nn.parameter.Parameter(init_mat, requires_grad=True)
        self.tolerance = 0.25
        self.prev_dag = 0.0

    def forward(self, X):
        W = self.W * self.diag_mask
        return dag_pred(X, W)

    def _batch_loss(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true)
        mse = mse_loss(x_true, x_pred)
        l1 = l1_loss(self.W, self.l1)
        dag = dag_loss(self.W, self.alpha, self.rho)
        return 0.5 * mse + l1 + dag

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x_true = batch
        x_pred = self(x_true).detach()
        mse = mse_loss(x_true, x_pred)
        dag = dag_loss(self.W, 1e12, 1e12).detach()
        loss = mse + dag
        self.log_dict({"val_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._batch_loss(batch, batch_idx)
        self.log_dict({"test_loss": loss})
        return loss

    def on_train_epoch_end(self, *args, **kwargs):
        dag = dag_loss(self.W, self.alpha, self.rho).item()
        if (
            dag > self.tolerance * self.prev_dag
            and self.alpha < 1e12
            and self.rho < 1e12
        ):
            self.alpha = self.alpha + self.rho * dag
            self.rho = self.rho * 10
        self.prev_dag = dag

    def dataloader(self, X, **kwargs):
        kwargs["batch_size"] = kwargs.get("batch_size", 32)
        X_tensor = torch.Tensor(X).to(torch.float32)
        return DataLoader(dataset=X_tensor, **kwargs)


class CorrelationNetwork:
    """
    Standard correlation network fit with linear regression.
    """

    def fit(self, X):
        self.p = X.shape[-1]
        self.regs = [[LinearRegression() for _ in range(self.p)] for _ in range(self.p)]
        for i in range(self.p):
            for j in range(self.p):
                self.regs[i][j].fit(X[:, j, np.newaxis], X[:, i, np.newaxis])
        return self

    def predict(self, n):
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                betas[i, j] = self.regs[i][j].coef_.squeeze()
        corrs = betas * betas.T
        return np.tile(np.expand_dims(corrs, axis=0), (n, 1, 1))

    def measure_mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            for j in range(self.p):
                residual = (
                    self.regs[i][j].predict(X[:, j, np.newaxis]) - X[:, i, np.newaxis]
                )
                residual = residual[:, 0]
                mses += (residual**2) / self.p**2
        return mses


class MarkovNetwork:
    """
    Standard Markov Network fit with neighborhood regression.
    """

    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.p = -1
        self.regs = []

    def fit(self, X):
        self.p = X.shape[-1]
        self.regs = [LinearRegression() for _ in range(self.p)]
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            self.regs[i].fit(X * mask, X[:, i, np.newaxis])
        return self

    def predict(self, n):
        betas = np.zeros((self.p, self.p))
        for i in range(self.p):
            betas[i] = self.regs[i].coef_.squeeze()
            betas[i, i] = 0
        precision = -np.sign(betas) * np.sqrt(np.abs(betas * betas.T))
        return np.tile(np.expand_dims(precision, axis=0), (n, 1, 1))

    def measure_mses(self, X):
        mses = np.zeros(len(X))
        for i in range(self.p):
            mask = np.ones_like(X)
            mask[:, i] = 0
            residual = self.regs[i].predict(X * mask) - X[:, i, np.newaxis]
            residual = residual[:, 0]
            mses += (residual**2) / self.p
        return mses


class BayesianNetwork:
    """
    A standard Bayesian Network fit with NOTEARS loss.
    """

    def __init__(self, **kwargs):
        self.p = -1
        self.model = None
        self.trainer = None
        self.l1 = kwargs.get("l1", 1e-3)
        self.alpha = kwargs.get("alpha", 1e-8)
        self.rho = kwargs.get("rho", 1e-8)
        self.learning_rate = kwargs.get("learning_rate", 1e-2)

    def fit(self, X, max_epochs=50):
        self.p = X.shape[-1]
        self.model = NOTEARS(
            self.p,
            l1=self.l1,
            alpha=self.alpha,
            rho=self.rho,
            learning_rate=self.learning_rate,
        )
        dataset = self.model.dataloader(X)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.trainer = NOTEARSTrainer(
            max_epochs=max_epochs, auto_lr_find=True, accelerator=accelerator, devices=1
        )
        self.trainer.fit(self.model, dataset)
        return self

    def predict(self, n):
        dummy_X = np.zeros((n, self.p))
        dummy_dataset = self.model.dataloader(dummy_X)
        return self.trainer.predict_w(self.model, dummy_dataset)

    def measure_mses(self, X):
        mses = np.zeros(len(X))
        W_pred = self.model.W.detach()
        X_preds = (
            dag_pred(torch.tensor(X, dtype=torch.float32), W_pred).detach().numpy()
        )
        return ((X_preds - X) ** 2).mean(axis=1)


class GroupedNetworks:
    """
    Fit a separate network for each group.
    Wrapper around CorrelationNetwork, MarkovNetwork, or BayesianNetwork.
    Assumes that the labels are 0-indexed integers and already learned.
    """

    def __init__(self, model_class):
        self.model_class = model_class
        self.models = {}
        self.p = -1

    def fit(self, X, labels):
        self.p = X.shape[-1]
        for label in np.unique(labels):
            model = self.model_class().fit(X[labels == label])
            self.models[label] = model
        return self

    def predict(self, labels):
        networks = np.zeros((len(labels), self.p, self.p))
        for label in np.unique(labels):
            label_idx = labels == label
            networks[label_idx] = self.models[label].predict(label_idx.sum())
        return networks

    def measure_mses(self, X, labels):
        mses = np.zeros(len(X))
        for label in np.unique(labels):
            label_idx = labels == label
            mses[label_idx] = self.models[label].measure_mses(X[label_idx])
        return mses
