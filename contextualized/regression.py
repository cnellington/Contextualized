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


class CRTrainer(pl.Trainer):
    def predict_params(self, model, dataloader):
        preds = super().predict(model, dataloader)
        return model._params_reshape(preds, dataloader)
    
    def predict_y(self, model, dataloader):
        preds = super().predict(model, dataloader)
        return model._y_reshape(preds, dataloader)


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
        self.n_i += 1
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1),
            self.Y[self.n_i].unsqueeze(-1),
            self.n_i,
        )
        return ret
        
    def __len(self):
        return self.n


class UnivariateDataset(Dataset):
    def __next__(self):
        self.n_i += 1
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        ret = (
            self.C[self.n_i],
            self.X[self.n_i].expand(self.y_dim, -1).unsqueeze(-1),
            self.Y[self.n_i].expand(self.x_dim, -1).T.unsqueeze(-1),
            self.n_i,
        )
        return ret
        
    def __len(self):
        return self.n


class MultitaskMultivariateDataset(Dataset):
    def __next__(self):
        self.y_i += 1
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
        return ret
        
    def __len(self):
        return self.n * self.y_dim


class MultitaskUnivariateDataset(Dataset):
    def __next__(self):
        self.y_i += 1
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
        return ret
    
    def __len__(self):
        return self.n * self.x_dim * self.y_dim    


class DataIterable(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        return iter(self.dataset)


def MSE(beta, mu, x, y, link_fn=lambda x: x):
    """
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
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
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
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, num_archetypes=10, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
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
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, num_archetypes=10, encoder_type='mlp', 
            encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x}):
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
    def __init__(self, context_dim, x_dim, y_dim, univariate=False, 
            context_archetypes=10, task_archetypes=10,
            context_encoder_type='mlp', 
            context_encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x},
            task_encoder_type='mlp',
            task_encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': lambda x: x},
            ):
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
    @abstractmethod
    def dataloader(self, C, X, Y, batch_size=32):
        # returns the dataloader for this class
        pass

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['univariate'] = False
        self.metamodel = NaiveMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _ = batch
        beta_hat, mu_hat = self.metamodel(C)
        return MSE(beta_hat, mu_hat, X, Y)
     
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
                ys[n_i] = ((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze(-1)
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultivariateDataset(C, X, Y)), batch_size=batch_size)


class SubtypeContextualizedRegression(ContextualizedRegressionBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['univariate'] = False
        self.metamodel = SubtypeMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, X, Y, _, = batch
        beta_hat, mu_hat = self.metamodel(C)
        return MSE(beta_hat, mu_hat, X, Y)
     
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
                ys[n_i] = ((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze(-1)
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultivariateDataset(C, X, Y)), batch_size=batch_size)


class MultitaskContextualizedRegression(ContextualizedRegressionBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['univariate'] = False
        self.metamodel = MultitaskMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        return MSE(beta_hat, mu_hat, X, Y)
     
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
                ys[n_i, y_i] = ((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskMultivariateDataset(C, X, Y)), batch_size=batch_size)


class TasksplitContextualizedRegression(ContextualizedRegressionBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['univariate'] = False
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        return MSE(beta_hat, mu_hat, X, Y)
     
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
                ys[n_i, y_i] = ((beta_hat * x).sum(axis=-1).unsqueeze(-1) + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskMultivariateDataset(C, X, Y)), batch_size=batch_size)


class TasksplitContextualizedUnivariateRegression(ContextualizedRegressionBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs['univariate'] = True
        self.metamodel = TasksplitMetamodel(*args, **kwargs)

    def _batch_loss(self, batch, batch_idx):
        C, T, X, Y, _, _, _ = batch
        beta_hat, mu_hat = self.metamodel(C, T)
        return MSE(beta_hat, mu_hat, X, Y)
     
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
                ys[n_i, y_i, x_i] = (beta_hat * x + mu_hat).squeeze()
        return ys
    
    def dataloader(self, C, X, Y, batch_size=32):
        return DataLoader(dataset=DataIterable(MultitaskUnivariateDataset(C, X, Y)), batch_size=batch_size)
