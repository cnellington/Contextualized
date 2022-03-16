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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, random_split
import pytorch_lightning as pl

from contextualized.modules import NGAM, MLP, SoftSelect, Explainer


ENCODERS = {
    'mlp': MLP,
    'ngam': NGAM,
}
MODELS = ['multivariate', 'univariate']
METAMODELS = ['simple', 'subtype', 'multitask', 'tasksplit']
LINK_FUNCTIONS = [
    lambda x: x,
    lambda x: F.softmax(x, dim=1),
    lambda x: 1 / (1 + torch.exp(-x))
]


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression models
    """
    def predict_params(self, model, dataloader):
        """
        Returns context-specific regression models 
        - beta (numpy.ndarray): (n, y_dim, x_dim)
        - mu (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)
        return model._params_reshape(preds, dataloader)
    
    def predict_y(self, model, dataloader):
        """
        Returns context-specific predictions of the response Y
        - y_hat (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)
        return model._y_reshape(preds, dataloader)


class Dataset:
    """
    Superclass for datastreams (iterators) used to train contextualized.regression models 
    """
    def __init__(self, C, X, Y, dtype=torch.float):
        self.C = torch.tensor(C, dtype=dtype)
        self.X = torch.tensor(X, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        self.n = C.shape[0]
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.dtype = dtype

    def __iter__(self):
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        return self
    
    @abstractmethod
    def __next__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass


class MultivariateDataset(Dataset):
    def __next__(self):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1),
            self.Y[self.n_i].unsqueeze(-1),
            self.n_i,
        )
        self.n_i += 1
        return ret
        
    def __len(self):
        return self.n


class UnivariateDataset(Dataset):
    def __next__(self):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1).unsqueeze(-1),
            self.Y[self.n_i].expand(self.x_dim, -1).T.unsqueeze(-1),
            self.n_i,
        )
        self.n_i += 1
        return ret
        
    def __len(self):
        return self.n


class MultitaskMultivariateDataset(Dataset):
    def __next__(self):
        if self.y_i >= self.y_dim:
            self.n_i += 1
            self.y_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = torch.zeros(self.y_dim)
        t[self.y_i] = 1
        ret = (
            self.C[self.n_i],
            t,
            self.X[self.n_i],
            self.Y[self.n_i, self.y_i].unsqueeze(0),
            self.n_i,
            self.y_i,
        )
        self.y_i += 1
        return ret
        
    def __len(self):
        return self.n * self.y_dim


class MultitaskUnivariateDataset(Dataset):
    def __next__(self):
        if self.y_i >= self.y_dim:
            self.x_i += 1
            self.y_i = 0
        if self.x_i >= self.x_dim:
            self.n_i += 1
            self.x_i = 0
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = torch.zeros(self.x_dim + self.y_dim)
        t[self.x_i] = 1
        t[self.x_dim + self.y_i] = 1
        ret = (
            self.C[self.n_i],
            t,
            self.X[self.n_i, self.x_i].unsqueeze(0),
            self.Y[self.n_i, self.y_i].unsqueeze(0),
            self.n_i,
            self.x_i,
            self.y_i,
        )
        self.y_i += 1
        return ret
    
    def __len__(self):
        return self.n * self.x_dim * self.y_dim    


class DataIterable(IterableDataset):
    """
    Dataset wrapper, required by PyTorch
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        return iter(self.dataset)


def MSE(beta, mu, x, y, link_fn=lambda x: x):
    """
    Returns
    - MSE (scalar torch.tensor): the mean squared-error or L2-error 
        of multivariate and univariate regression problems. Default
        loss for contextualized.regression models.
    
    MV/UV: Multivariate/Univariate
    MT/ST: Multi-task/Single-task

    MV ST: beta (y_dim, x_dim),    mu (y_dim, 1),        x (y_dim, x_dim),    y (y_dim, 1)
    MV MT: beta (x_dim,),          mu (1,),              x (x_dim,),          y (1,)
    UV ST: beta (y_dim, x_dim, 1), mu (y_dim, x_dim, 1), x (y_dim, x_dim, 1), y (y_dim, x_dim, 1)
    UV MT: beta (1,),              mu (1,),              x (1,),              y (1,)
    """
    y_hat = link_fn((beta * x).sum(axis=-1).unsqueeze(-1) + mu)
    residual = y_hat - y
    return residual.pow(2).mean()


class NaiveMetamodel(nn.Module):
    """
    Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) --> {beta, mu} --> (X, Y)
    """
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels
        
        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        encoder_type (str: mlp): encoder module to use
        encoder_kwargs (dict): encoder args and kwargs
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        self.mu_dim = x_dim if univariate else 1
        out_dim = (x_dim + self.mu_dim) * y_dim
        self.context_encoder = encoder(context_dim, out_dim, **encoder_kwargs)

    def forward(self, C):
        W = self.context_encoder(C)
        W = torch.reshape(W, (W.shape[0], self.y_dim, self.x_dim + self.mu_dim))
        beta = W[:, :, :self.x_dim]
        mu = W[:, :, self.x_dim:]
        return beta, mu


class SubtypeMetamodel(nn.Module):
    """
    Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z} --> {beta, mu} --> (X)
    
    Z: latent variable, causal parent of both the context and regression model
    """
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, num_archetypes=10, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels
        
        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        num_archetypes (int: 10): number of atomic regression models in {Z}
        encoder_type (str: mlp): encoder module to use
        encoder_kwargs (dict): encoder args and kwargs
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        mu_dim = x_dim if univariate else 1
        self.context_encoder = encoder(context_dim, num_archetypes, **encoder_kwargs)
        self.explainer = Explainer(num_archetypes, (self.y_dim, x_dim + mu_dim))

    def forward(self, C):
        Z = self.context_encoder(C)
        W = self.explainer(Z)
        beta = W[:, :, :self.x_dim]
        mu = W[:, :, self.x_dim:]
        return beta, mu


class MultitaskMetamodel(nn.Module):
    """
    Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z} --> {beta, mu} --> (X)
    (T) <---/
    
    Z: latent variable, causal parent of the context, regression model, and task (T)
    """
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, num_archetypes=10, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels
        
        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        num_archetypes (int: 10): number of atomic regression models in {Z}
        encoder_type (str: mlp): encoder module to use
        encoder_kwargs (dict): encoder args and kwargs
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        beta_dim = 1 if univariate else x_dim
        task_dim = y_dim + x_dim if univariate else y_dim
        self.context_encoder = encoder(context_dim + task_dim, num_archetypes, **encoder_kwargs)
        self.explainer = Explainer(num_archetypes, (beta_dim + 1, ))

    def forward(self, C, T):
        CT = torch.cat((C, T), 1)
        Z = self.context_encoder(CT)
        W = self.explainer(Z)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu


class TasksplitMetamodel(nn.Module):
    """
    Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z_c} --> {beta, mu} --> (X)
    (T) <-- {Z_t} ----^
    
    Z_c: latent context variable, causal parent of the context and regression model
    Z_t: latent task variable, causal parent of the task and regression model
    """
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, 
            context_archetypes=10, task_archetypes=10,
            context_encoder_type='mlp', 
            context_encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x},
            task_encoder_type='mlp',
            task_encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x},
            ):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels
        
        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        context_archetypes (int: 10): number of atomic regression models in {Z_c}
        task_archetypes (int: 10): number of atomic regression models in {Z_t}
        context_encoder_type (str: mlp): context encoder module to use
        context_encoder_kwargs (dict): context encoder args and kwargs
        task_encoder_type (str: mlp): task encoder module to use
        task_encoder_kwargs (dict): task encoder args and kwargs
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        context_encoder = ENCODERS[context_encoder_type]
        task_encoder = ENCODERS[task_encoder_type]
        beta_dim = 1 if univariate else x_dim
        task_dim = y_dim + x_dim if univariate else y_dim
        self.context_encoder = context_encoder(context_dim, context_archetypes, **context_encoder_kwargs)
        self.task_encoder = task_encoder(task_dim, task_archetypes, **task_encoder_kwargs)
        self.explainer = SoftSelect((context_archetypes, task_archetypes), (beta_dim + 1, ))

    def forward(self, C, T):
        Z_c = self.context_encoder(C) 
        Z_t = self.task_encoder(T)
        W = self.explainer(Z_c, Z_t)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu


class ContextualizedRegressionBase(pl.LightningModule):
    def __init__(self, *args, learning_rate=1e-3, link_fn=lambda x: x, **kwargs):
        super().__init__()
        self.link_fn = link_fn
        self.learning_rate = learning_rate
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
        return self.metamodel(*args)
    
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


class NaiveContextualizedRegression(ContextualizedRegressionBase):
    """
    See NaiveMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = NaiveMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _ = batch
        beta_hat, mu_hat = self.metamodel(C)
        return MSE(beta_hat, mu_hat, X, Y, link_fn=self.link_fn)
     
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
            for beta_hat, mu_hat, n_i in zip(beta_hats, mu_hats, n_idx):
                betas[n_i] = beta_hat
                mus[n_i] = mu_hat.squeeze(-1)
        return betas, mus
    
    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, X, _, n_idx = data
            for beta_hat, mu_hat, x, n_i in zip(beta_hats, mu_hats, X, n_idx):
                ys[n_i] = self.link_fn((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze(-1)
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultivariateDataset(C, X, Y)), batch_size=batch_size)


class ContextualizedRegression(ContextualizedRegressionBase):
    """
    See SubtypeMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = SubtypeMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _, = batch
        beta_hat, mu_hat = self.metamodel(C)
        return MSE(beta_hat, mu_hat, X, Y, link_fn=self.link_fn)
     
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
            for beta_hat, mu_hat, n_i in zip(beta_hats, mu_hats, n_idx):
                betas[n_i] = beta_hat
                mus[n_i] = mu_hat.squeeze(-1)
        return betas, mus
    
    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, X, _, n_idx = data
            for beta_hat, mu_hat, x, n_i in zip(beta_hats, mu_hats, X, n_idx):
                ys[n_i] = self.link_fn((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze(-1)
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultivariateDataset(C, X, Y)), batch_size=batch_size)


class MultitaskContextualizedRegression(ContextualizedRegressionBase):
    """
    See MultitaskMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = MultitaskMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        return MSE(beta_hat, mu_hat, X, Y, link_fn=self.link_fn)
     
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
            for beta_hat, mu_hat, n_i, y_i in zip(beta_hats, mu_hats, n_idx, y_idx):
                betas[n_i, y_i] = beta_hat
                mus[n_i, y_i] = mu_hat.squeeze()
        return betas, mus
    
    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, X, _, n_idx, y_idx = data
            for beta_hat, mu_hat, x, n_i, y_i in zip(beta_hats, mu_hats, X, n_idx, y_idx):
                ys[n_i, y_i] = self.link_fn((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskMultivariateDataset(C, X, Y)), batch_size=batch_size)


class TasksplitContextualizedRegression(ContextualizedRegressionBase):
    """
    See TasksplitMetamodel
    """
    def _build_metamodel(self, *args, **kwargs):
        kwargs['univariate'] = False
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        return MSE(beta_hat, mu_hat, X, Y, link_fn=self.link_fn)
     
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
            for beta_hat, mu_hat, n_i, y_i in zip(beta_hats, mu_hats, n_idx, y_idx):
                betas[n_i, y_i] = beta_hat
                mus[n_i, y_i] = mu_hat.squeeze()
        return betas, mus
    
    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, X, _, n_idx, y_idx = data
            for beta_hat, mu_hat, x, n_i, y_i in zip(beta_hats, mu_hats, X, n_idx, y_idx):
                ys[n_i, y_i] = self.link_fn((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskMultivariateDataset(C, X, Y)), batch_size=batch_size)


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
        return MSE(beta_hat, mu_hat, X, Y, link_fn=self.link_fn)
     
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
            for beta_hat, mu_hat, n_i, x_i, y_i in zip(beta_hats, mu_hats, n_idx, x_idx, y_idx):
                betas[n_i, y_i, x_i] = beta_hat.squeeze()
                mus[n_i, y_i, x_i] = mu_hat.squeeze()
        return betas, mus
    
    def _y_reshape(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, ds.y_dim, ds.x_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, X, _, n_idx, x_idx, y_idx = data
            for beta_hat, mu_hat, x, n_i, x_i, y_i in zip(beta_hats, mu_hats, X, n_idx, x_idx, y_idx):
                ys[n_i, y_i, x_i] = self.link_fn(beta_hat * x + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskUnivariateDataset(C, X, Y)), batch_size=batch_size)


if __name__ == '__main__':
    n = 100
    c_dim = 4
    x_dim = 2
    y_dim = 3
    C = torch.rand((n, c_dim)) - .5 
    W_1 = C.sum(axis=1).unsqueeze(-1) ** 2
    W_2 = - C.sum(axis=1).unsqueeze(-1)
    b_1 = C[:, 0].unsqueeze(-1)
    b_2 = C[:, 1].unsqueeze(-1)
    W_full = torch.cat((W_1, W_2), axis=1)
    b_full = b_1 + b_2
    X = torch.rand((n, x_dim)) - .5
    Y_1 = X[:, 0].unsqueeze(-1) * W_1 + b_1
    Y_2 = X[:, 1].unsqueeze(-1) * W_2 + b_2
    Y_3 = X.sum(axis=1).unsqueeze(-1)
    Y = torch.cat((Y_1, Y_2, Y_3), axis=1)

    k = 10
    epochs = 2
    batch_size = 1
    C, X, Y = C.numpy(), X.numpy(), Y.numpy()

    def quicktest(model):
        print(f'{type(model)} quicktest')
        dataloader = model.dataloader(C, X, Y, batch_size=32)
        trainer = RegressionTrainer(max_epochs=1)
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)


    # Naive Multivariate
    model = NaiveContextualizedRegression(c_dim, x_dim, y_dim)
    quicktest(model)

    # Subtype Multivariate
    model = ContextualizedRegression(c_dim, x_dim, y_dim)
    quicktest(model)

    # Multitask Multivariate
    model = MultitaskContextualizedRegression(c_dim, x_dim, y_dim)
    quicktest(model)

    # Tasksplit Multivariate
    model = TasksplitContextualizedRegression(c_dim, x_dim, y_dim)
    quicktest(model)

    # Tasksplit Univariate
    model = TasksplitContextualizedUnivariateRegression(c_dim, x_dim, y_dim)
    quicktest(model)
