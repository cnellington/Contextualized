from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
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
LINK_FNS = [
    lambda x: x,
    lambda x: F.softmax(x, dim=1)
]


class Dataset:
    def __init__(self, C, X, Y, dtype=torch.float):
        """
        C: (n x c_dim)
        X: (n x x_dim)
        Y: (n x y_dim)
        """
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
        pass
        
    def __len(self):
        return self.n
    
class UnivariateDataset(Dataset):
    def __next__(self):
        pass
        
    def __len(self):
        return self.n
    
class MultitaskMultivariateDataset(Dataset):
    def __next__(self):
        pass
        
    def __len(self):
        return self.n * self.y_dim
    
class MultitaskUnivariateDataset(Dataset):
    def __next__(self):
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        t = torch.zeros(self.x_dim + self.y_dim)
        t[self.x_i] = 1
        t[self.x_dim + self.y_i] = 1
        ret = (
            self.C[self.n_i],
            t,
            self.X[self.n_i, self.x_i:self.x_i+1],
            self.Y[self.n_i, self.y_i:self.y_i+1],
            self.n_i,
            self.x_i,
            self.y_i,
        )
        self.y_i += 1
        if self.y_i >= self.y_dim:
            self.x_i += 1
            self.y_i = 0
        if self.x_i >= self.x_dim:
            self.n_i += 1
            self.x_i = 0
        return ret
    
    def __len__(self):
        return self.n * self.x_dim * self.y_dim    

class DataIterable(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        return iter(self.dataset)


# class ContextualizedRegressionAbstract(pl.LightningModule):
def MSE(beta, mu, x, y, link_fn=lambda x: x):
    y_hat = link_fn((beta * x).sum(axis=1).unsqueeze(-1) + mu)
    residual = y_hat - y
    return residual.pow(2).mean()


class NaiveContextualizedRegression(pl.LightningModule):
    def __init__(self, context_dim, x_dim, y_dim, encoder_type='mlp', univariate=False,
                 num_archetypes=10, encoder_width=25, encoder_layers=2):
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.link_fn = link_fn

        encoder = ENCODERS[encoder_type]
        mu_dim = x_dim if univariate else 1
        out_dim = (x_dim + mu_dim) * y_dim
        self.context_encoder = encoder(context_dim, out_dim, encoder_width, encoder_layers)

    def forward(self, C):
        W = self.context_encoder(C)
        W = torch.reshape(out, (out.shape[0], self.y_dim, x_dim + mu_dim))
        beta = W[:, :, :self.x_dim]
        mu = W[:, :, self.x_dim:]
        return beta, mu


class SubtypeContextualizedRegression(pl.LightningModule):
    def __init__(self, context_dim, x_dim, y_dim, encoder_type='mlp', univariate=False,
                 num_archetypes=10, encoder_width=25, encoder_layers=2, link_fn=lambda x: x):
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.link_fn = link_fn

        encoder = ENCODERS[encoder_type]
        mu_dim = x_dim if univariate else 1
        self.context_encoder = encoder(context_dim, num_archetypes, encoder_width, encoder_layers)
        self.explainer = Explainer(num_archetypes, (self.y_dim, x_dim + mu_dim))

    def forward(self, C):
        Z_pre = self.context_encoder(C)
        Z = self.link_fn(Z_pre)
        W = self.explainer(Z)
        beta = W[:, :, :self.x_dim]
        mu = W[:, :, self.x_dim:]
        return beta, mu


class MultitaskContextualizedRegression(pl.LightningModule):
    def __init__(self, context_dim, x_dim, y_dim, encoder_type='mlp', univariate=False,
                 num_archetypes=10, encoder_width=25, encoder_layers=2, link_fn=lambda x: x):
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.link_fn = link_fn

        encoder = ENCODERS[encoder_type]
        beta_dim = 1 if univariate else x_dim
        task_dim = y_dim + x_dim if univariate else y_dim
        self.context_encoder = encoder(context_dim + task_dim, num_archetypes, encoder_width, encoder_layers)
        self.explainer = Explainer(num_archetypes, (beta_dim + 1, ))

    def forward(self, C, T):
        CT = torch.cat((C, T), 1)
        Z_pre = self.context_encoder(CT)
        Z = self.link_fn(Z_pre)
        W = self.explainer(Z)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu

    
class TasksplitContextualizedRegression(pl.LightningModule):
    def __init__(self, context_dim, x_dim, y_dim, encoder_type='mlp', univariate=False,
                 num_archetypes=10, encoder_width=25, encoder_layers=2, link_fn=lambda x: x):
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.link_fn = link_fn

        encoder = ENCODERS[encoder_type]
        beta_dim = 1 if univariate else x_dim
        task_dim = y_dim + x_dim if univariate else y_dim
        self.context_encoder = encoder(context_dim, num_archetypes, encoder_width, encoder_layers)
        self.task_encoder = encoder(task_dim, num_archetypes, encoder_width, encoder_layers)
        self.explainer = SoftSelect((num_archetypes, num_archetypes), (beta_dim + 1, ))

    def forward(self, C, T):
        Z_c_pre = self.context_encoder(C) 
        Z_t_pre = self.task_encoder(T)
        Z_c = self.link_fn(Z_c_pre)
        Z_t = self.link_fn(Z_t_pre)
        W = self.explainer(Z_c, Z_t)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        loss = MSE(beta_hat, mu_hat, X, Y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx):
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self(C, T)
        return beta_hat, mu_hat
    
    def _coef_preds(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        betas = np.zeros((ds.n, 
                          ds.x_dim, 
                          ds.y_dim))
        mus = betas.copy()
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, _, _, n_idx, x_idx, y_idx = data
            for beta_hat, mu_hat, n_i, x_i, y_i in zip(beta_hats, mu_hats, n_idx, x_idx, y_idx):
                betas[n_i, x_i, y_i] = beta_hat
                mus[n_i, x_i, y_i] = mu_hat
        return betas, mus
    
    def _y_preds(self, preds, dataloader):
        ds = dataloader.dataset.dataset
        ys = np.zeros((ds.n, 
                       ds.x_dim, 
                       ds.y_dim))
        for (beta_hats, mu_hats), data in zip(preds, dataloader):
            _, _, X, Y, n_idx, x_idx, y_idx = data
            for beta_hat, mu_hat, x, y, n_i, x_i, y_i in zip(beta_hats, mu_hats, X, Y, n_idx, x_idx, y_idx):
                ys[n_i, x_i, y_i] = beta_hat * y + mu_hat
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskUnivariateDataset(C, X, Y)), batch_size=batch_size)
    
class ContextualizedUnivariateRegression:
    def __init__(self, context_dim, x_dim, y_dim, metamodel='tasksplit'):
        if metamodel == 'tasksplit':
            self.model = TasksplitContextualizedRegression(context_dim, x_dim, y_dim, univariate=True)