import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def MSE(beta, mu, x, y, link_fn=lambda x: x):
    y_hat = link_fn((beta * x).sum(axis=1).unsqueeze(-1) + mu)
    residual = y_hat - y
    return residual.pow(2).mean()


class SoftSelect(nn.Module):
    """
    Parameter sharing for multiple context encoders:
    Batched computation for mapping many subtypes onto d-dimensional archetypes
    """
    def __init__(self, in_dims, out_shape):
        super(SoftSelect, self).__init__()
        init_mat = torch.rand(list(out_shape) + list(in_dims)) * 2e-2 - 1e-2
        self.archetypes = nn.parameter.Parameter(init_mat, requires_grad=True)

    def forward(self, *batch_weights):
        batch_size = batch_weights[0].shape[0]
        expand_dims = [batch_size] + [-1 for _ in range(len(self.archetypes.shape))]
        batch_archetypes = self.archetypes.unsqueeze(0).expand(expand_dims)
        for batch_w in batch_weights[::-1]:
            batch_w = batch_w.unsqueeze(-1)
            d = len(batch_archetypes.shape) - len(batch_w.shape)
            for _ in range(d):
                batch_w = batch_w.unsqueeze(1)
            batch_archetypes = torch.matmul(batch_archetypes, batch_w).squeeze(-1)
        return batch_archetypes


class Explainer(nn.Module):
    """
    2D subtype-archetype parameter sharing
    """
    def __init__(self, k, out_shape):
        super(Explainer, self).__init__()
        self.softselect = SoftSelect((k, ), out_shape)

    def forward(self, batch_subtypes):
        return self.softselect(batch_subtypes)


class NGAM(nn.Module):
    """
    Generalized additive model implemented with neural networks for feature-specific functions
    """
    def __init__(self, input_dim, output_dim, width=25, layers=2, activation=nn.ReLU):
        super(NGAM, self).__init__()
        self.intput_dim = input_dim
        self.output_dim = output_dim
        hidden_layers = lambda: [layer for _ in range(0, layers - 2) for layer in (nn.Linear(width, width), activation())]
        nam_layers = lambda: [nn.Linear(1, width), activation()] + hidden_layers() + [nn.Linear(width, output_dim)]
        self.nams = nn.ModuleList([nn.Sequential(*nam_layers()) for _ in range(input_dim)])

    def forward(self, x):
        batch_size = x.shape[0]
        ret = torch.zeros((batch_size, self.output_dim))
        for i, nam in enumerate(self.nams):
            ret += nam(x[:, i].unsqueeze(-1))
        return ret

    
class MLP(nn.Module):
    """
    multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim, width=25, layers=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        hidden_layers = lambda: [layer for _ in range(0, layers - 2) for layer in (nn.Linear(width, width), activation())]
        mlp_layers = [nn.Linear(input_dim, width), activation()] + hidden_layers() + [nn.Linear(width, output_dim)]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        ret = self.mlp(x)
        return ret


class ContextualizedRegressionModule(nn.Module):
    """
    Estimates the weights and offset in a context-specific and task-specific regression of X onto Y
    """
    def __init__(self, context_dim, task_dim, beta_dim=1, subtype='sample', 
                 num_archetypes=10, encoder_width=25, encoder_layers=2, activation=nn.ReLU, 
                 link_fn=lambda x: x):
        super(ContextualizedRegressionModule, self).__init__()
        self.context_dim = context_dim
        self.task_dim = task_dim
        self.beta_dim = beta_dim
        self.subtype = subtype
        self.link_fn = link_fn

        self.context_encoder = MLP(context_dim, num_archetypes, width=encoder_width, layers=encoder_layers)
        self.task_encoder = MLP(task_dim, num_archetypes, width=encoder_width, layers=encoder_layers)
        if subtype == 'sample':
            self.softselect = SoftSelect((num_archetypes, ), (beta_dim + 1, ))
        if subtype == 'modal':
            self.softselect = SoftSelect((num_archetypes, num_archetypes, ), (beta_dim + 1, ))

    def forward(self, c, t):
        Z_context = self.context_encoder(c)
        Z_t = self.task_encoder(t)
        Z = None
        if self.subtype == 'sample':
            Z = (self.link_fn(Z_context + Z_t), )
        if self.subtype == 'modal':
            Z = (self.link_fn(Z_context), self.link_fn(Z_t), )
        W = self.softselect(*Z)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu


class MultivariateDataset:
    def __init__(self, C, X, Y, batch_size=1, dtype=torch.float, device=torch.device('cpu')):
        """
        C: (n x c_dim)
        X: (n x x_dim)
        Y: (n x y_dim)
        """
        self.C = torch.tensor(C, dtype=dtype, device=device)
        self.X = torch.tensor(X, dtype=dtype, device=device)
        self.Y = torch.tensor(Y, dtype=dtype, device=device)
        self.n_i = 0
        self.y_i = 0
        self.n = C.shape[0]
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.task_dim = self.y_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        
    def __iter__(self):
        self.n_i = 0
        self.y_i = 0
        return self
        
    def sample(self):
        t = torch.zeros(self.task_dim)
        t[self.y_i] = 1
        ret = (
            self.C[self.n_i].unsqueeze(0),
            t.unsqueeze(0),
            self.X[self.n_i].unsqueeze(0),
            self.Y[self.n_i, self.y_i:self.y_i+1].unsqueeze(0),
        )
        self.y_i += 1
        if self.y_i >= self.y_dim:
            self.n_i += 1
            self.y_i = 0
        return ret
    
    def __next__(self):
        """
        Returns a batch_size sample (c, t, x, y)
        If there are fewer than batch_size samples remaining, returns n - batch_size samples
        c: (batch_size, c_dim)
        t: (batch_size, task_dim * 2)  [x_task y_task]
        x: (batch_size, x_dim)
        y: (batch_size, 1)
        """
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        C_batch, T_batch, X_batch, Y_batch = self.sample()
        while len(C_batch) < self.batch_size and self.n_i < self.n:
            C_s, T_s, X_s, Y_s = self.sample()
            C_batch = torch.cat((C_batch, C_s))
            T_batch = torch.cat((T_batch, T_s))
            X_batch = torch.cat((X_batch, X_s))
            Y_batch = torch.cat((Y_batch, Y_s))
        return C_batch, T_batch, X_batch, Y_batch
    
    def __len__(self):
        return self.n * self.y_dim


class ContextualizedRegression:
    def __init__(self, context_dim, x_dim, y_dim, subtype='sample', num_archetypes=10, 
                 l1=0, link_fn=lambda x: x, encoder_width=25, encoder_layers=2, encoder_link_fn=lambda x: x, 
                 encoder_activation=nn.ReLU, bootstraps=None, device=torch.device('cpu')):
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_dim = y_dim
        self.link_fn = link_fn
        module_params = {
            'context_dim': context_dim,
            'task_dim': y_dim,
            'beta_dim': x_dim,
            'subtype': subtype,
            'num_archetypes': num_archetypes,
            'encoder_width': encoder_width,
            'encoder_layers': encoder_layers,
            'activation': encoder_activation,
            'link_fn': encoder_link_fn,
        }
        if bootstraps is None:
            self.model = ContextualizedRegressionModule(**module_params)
            self.models = [self.model]
        else:
            self.model = None
            self.models = [ContextualizedRegressionModule(**module_params) for _ in range(bootstraps)]
        self.to(device)

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(self.device)
        return self

    def _loss(self, outputs, X, Y):
        beta, mu = outputs
        mse = MSE(beta, mu, X, Y, link_fn=self.link_fn)
        return mse

    def _fit(self, model, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, 
             validation_set=None, es_patience=np.inf, es_epoch=0, silent=False):
        model.train()
        device = next(model.parameters()).device
        opt = optimizer(model.parameters(), lr=lr)
        dataset = MultivariateDataset(C, X, Y, batch_size=batch_size, dtype=torch.float, device=device)
        val_dataset = None
        if validation_set is not None:
            val_dataset = MultivariateDataset(*validation_set, batch_size=batch_size, dtype=torch.float, device=device)
        progress_bar = tqdm(range(epochs), disable=silent)
        min_loss = np.inf  # for early stopping
        es_count = 0
        for epoch in progress_bar:
            for batch_i, (c, t, x, y) in enumerate(dataset):
                loss = None
                # Train
                outputs = model(c, t)
                loss = self._loss(outputs, x, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # Update UI and check validation set for early stopping
                train_desc = f'[Train MSE: {loss.item():.4f}] [Sample: {batch_size * batch_i}/{len(dataset)}] Epoch'
                if val_dataset is not None:
                    val_loss_batches = []
                    for c_val, t_val, x_val, y_val in val_dataset:
                        val_outputs = model(c_val, t_val)
                        val_loss_batch = self._loss(val_outputs, x_val, y_val).item()
                        val_loss_batches.append(val_loss_batch)
                    val_loss = np.mean(val_loss_batches)
                    train_desc = f"[Val MSE: {val_loss:.4f}] " + train_desc
                    if epoch >= es_epoch:
                        if val_loss < min_loss:
                            min_loss = val_loss
                            es_count = 0
                        else:
                            es_count += 1
                progress_bar.set_description(train_desc)
                if es_count > es_patience:
                    model.eval()
                    return
        model.eval()

    def fit(self, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, validation_set=None, es_patience=np.inf, es_epoch=0, silent=False):
        fit_params = {
            'C': C, 'X': X, 'Y': Y, 'epochs': epochs, 'batch_size': batch_size,
            'optimizer': optimizer, 'lr': lr,
            'validation_set': validation_set, 'es_epoch': es_epoch, 'es_patience': es_patience,
            'silent': silent,
        }
        if self.model:
            fit_params['model'] = self.model
            self._fit(**fit_params)
        else:
            for model in self.models:
                boot_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
                fit_params.update({
                    'model': model, 'C': C[boot_idx], 'X': X[boot_idx], 'Y': Y[boot_idx],
                })
                self._fit(**fit_params)

    def _predict_coefs(self, model, C):
        """
        Predict a (1, c_dim, x_dim, y_dim) matrix of regression coefficients and offsets for each context
        beta[i,j] and mu[i,j] solve the regression problem y_j = beta[i,j] * x_i + mu[i,j]
        returns a numpy matrix
        """
        n = C.shape[0]
        betas = torch.zeros((n, self.x_dim, self.y_dim))
        mus = torch.zeros((n, self.y_dim))
        for i in range(n):  # Predict per-sample to avoid OOM
            C_i = torch.Tensor(C[i])
            for t_y in range(self.y_dim):
                task = torch.zeros(self.y_dim)
                task[t_y] = 1
                beta, mu = model(C_i.unsqueeze(0), task.unsqueeze(0))
                beta, mu = beta.cpu().detach(), mu.cpu().detach()
                betas[i, :, t_y] = beta.squeeze()
                mus[i, t_y] = mu.squeeze()
        return betas.unsqueeze(0).numpy(), mus.unsqueeze(0).numpy()

    def predict_coefs(self, C, all_bootstraps=False):
        """
        Predict an (x_dim, y_dim) matrix of regression coefficients for each context
        Returns a numpy matrix (1 or bootstraps, n, x_dim, y_dim)
        """
        betas, mus = self._predict_coefs(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_coefs(model, C)
            betas = np.concatenate((betas, betas_i), axis=0)
            mus = np.concatenate((mus, mus_i), axis=0)
        if all_bootstraps:
            return betas, mus
        return betas.mean(axis=0), mus.mean(axis=0)
    
    def predict_y(self, C, X, all_bootstraps=False):
        betas, mus = self.predict_coefs(C, all_bootstraps=True)
        y_hat = np.zeros((len(self.models), len(C), self.y_dim))
        for i in range(len(self.models)):
            for y_i in range(self.y_dim):
                beta_y = betas[i, :, :, y_i]
                mu_y = mus[i, :, y_i]
                y_hat[i, :, y_i] = self.link_fn((beta_y * X).sum(axis=1) + mu_y)
        if all_bootstraps:
            return y_hat
        return y_hat.mean(axis=0)

    def get_mse(self, C, X, Y, all_bootstraps=False):
        """
        Returns the MSE of the model on a dataset
        Returns a numpy array (1 or bootstraps, )
        """
        dataset = MultivariateDataset(C, X, Y, batch_size=1, device=self.device)
        mses = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            for c, t, x, y in dataset:
                betas, mus = model(c, t)
                mse = MSE(betas, mus, x, y, link_fn=self.link_fn).item()
                mses[i] += 1 / len(dataset) * mse
        if not all_bootstraps:
            return mses.mean()
        return mses


class UnivariateDataset:
    def __init__(self, C, X, Y, batch_size=1, dtype=torch.float, device=torch.device('cpu')):
        """
        C: (n x c_dim)
        X: (n x x_dim)
        Y: (n x y_dim)
        """
        self.C = torch.tensor(C, dtype=dtype, device=device)
        self.X = torch.tensor(X, dtype=dtype, device=device)
        self.Y = torch.tensor(Y, dtype=dtype, device=device)
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        self.n = C.shape[0]
        self.c_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

    def __iter__(self):
        self.n_i = 0
        self.x_i = 0
        self.y_i = 0
        return self
    
    def sample(self):
        t = torch.zeros(self.x_dim + self.y_dim)
        t[self.x_i] = 1
        t[self.x_dim + self.y_i] = 1
        ret = (
            self.C[self.n_i].unsqueeze(0),
            t.unsqueeze(0),
            self.X[self.n_i, self.x_i:self.x_i+1].unsqueeze(0),
            self.Y[self.n_i, self.y_i:self.y_i+1].unsqueeze(0),
        )
        self.y_i += 1
        if self.y_i >= self.y_dim:
            self.x_i += 1
            self.y_i = 0
        if self.x_i >= self.x_dim:
            self.n_i += 1
            self.x_i = 0
        return ret

    def __next__(self):
        """
        Returns a batch_size paired sample (c, t, x, y)
        If there are fewer than batch_size samples remaining, returns n - batch_size samples
        c: (batch_size, c_dim)
        t: (batch_size, task_dim)  [x_task x y_task]
        x: (batch_size, 1)
        y: (batch_size, 1)
        """
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        C_batch, T_batch, X_batch, Y_batch = self.sample()
        while len(C_batch) < self.batch_size and self.n_i < self.n:
            C_s, T_s, X_s, Y_s = self.sample()
            C_batch = torch.cat((C_batch, C_s))
            T_batch = torch.cat((T_batch, T_s))
            X_batch = torch.cat((X_batch, X_s))
            Y_batch = torch.cat((Y_batch, Y_s))
        return C_batch, T_batch, X_batch, Y_batch
    
    def __len__(self):
        return self.n * self.x_dim * self.y_dim

    def __next__(self):
        """
        Returns a batch_size paired sample (c, t, x, y)
        If there are fewer than batch_size samples remaining, returns n - batch_size samples
        c: (batch_size, c_dim)
        t: (batch_size, task_dim)  [x_task x y_task]
        x: (batch_size, 1)
        y: (batch_size, 1)
        """
        if self.n_i >= self.n:
            self.n_i = 0
            raise StopIteration
        C_batch, T_batch, X_batch, Y_batch = self.sample()
        while len(C_batch) < self.batch_size and self.n_i < self.n:
            C_s, T_s, X_s, Y_s = self.sample()
            C_batch = torch.cat((C_batch, C_s))
            T_batch = torch.cat((T_batch, T_s))
            X_batch = torch.cat((X_batch, X_s))
            Y_batch = torch.cat((Y_batch, Y_s))
        return C_batch, T_batch, X_batch, Y_batch
    
    def __len__(self):
        return self.n * self.x_dim * self.y_dim
    

class ContextualizedUnivariateRegression:
    def __init__(self, context_dim, x_dim, y_dim, subtype='sample', num_archetypes=10,
                 l1=0, link_fn=lambda x: x, encoder_width=25, encoder_layers=2, encoder_activation=nn.ReLU, 
                 encoder_link_fn=lambda x: x, bootstraps=None, device=torch.device('cpu')):
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.taskpair_dim = x_dim + y_dim
        self.link_fn = link_fn
        module_params = {
            'context_dim': context_dim,
            'task_dim': self.taskpair_dim,
            'beta_dim': 1,
            'subtype': subtype,
            'num_archetypes': num_archetypes,
            'encoder_width': encoder_width,
            'encoder_layers': encoder_layers,
            'activation': encoder_activation,
            'link_fn': encoder_link_fn,
        }
        if bootstraps is None:
            self.model = ContextualizedRegressionModule(**module_params)
            self.models = [self.model]
        else:
            self.model = None
            self.models = [ContextualizedRegressionModule(**module_params) for _ in range(bootstraps)]
        self.to(device)

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(self.device)
        return self

    def _loss(self, outputs, X, Y):
        beta, mu = outputs
        mse = MSE(beta, mu, X, Y, link_fn=self.link_fn)
        return mse

    def _fit(self, model, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, 
             validation_set=None, es_patience=np.inf, es_epoch=0, silent=False):
        model.train()
        device = next(model.parameters()).device
        opt = optimizer(model.parameters(), lr=lr)
        dataset = UnivariateDataset(C, X, Y, batch_size=batch_size, dtype=torch.float, device=device)
        val_dataset = None
        if validation_set is not None:
            val_dataset = UnivariateDataset(*validation_set, batch_size=batch_size, dtype=torch.float, device=device)
        progress_bar = tqdm(range(epochs), disable=silent)
        min_loss = np.inf  # for early stopping
        es_count = 0
        for epoch in progress_bar:
            for batch_i, (c, t, x, y) in enumerate(dataset):
                loss = None
                # Train
                outputs = model(c, t)
                loss = self._loss(outputs, x, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # Update UI and check validation set for early stopping
                train_desc = f'[Train MSE: {loss.item():.4f}] [Sample: {batch_size * batch_i}/{len(dataset)}] Epoch'
                if val_dataset is not None:
                    val_loss_batches = []
                    for c_val, t_val, x_val, y_val in val_dataset:
                        val_outputs = model(c_val, t_val)
                        val_loss_batch = self._loss(val_outputs, x_val, y_val).item()
                        val_loss_batches.append(val_loss_batch)
                    val_loss = np.mean(val_loss_batches)
                    train_desc = f"[Val MSE: {val_loss:.4f}] " + train_desc
                    if epoch >= es_epoch:
                        if val_loss < min_loss:
                            min_loss = val_loss
                            es_count = 0
                        else:
                            es_count += 1
                progress_bar.set_description(train_desc)
                if es_count > es_patience:
                    model.eval()
                    return
        model.eval()
        
    def fit(self, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, validation_set=None, es_patience=np.inf, es_epoch=0, silent=False):
        fit_params = {
            'C': C, 'X': X, 'Y': Y, 'epochs': epochs, 'batch_size': batch_size,
            'optimizer': optimizer, 'lr': lr,
            'validation_set': validation_set, 'es_epoch': es_epoch, 'es_patience': es_patience,
            'silent': silent,
        }
        if self.model:
            fit_params['model'] = self.model
            self._fit(**fit_params)
        else:
            for model in self.models:
                boot_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
                fit_params.update({
                    'model': model, 'C': C[boot_idx], 'X': X[boot_idx], 'Y': Y[boot_idx],
                })
                self._fit(**fit_params)

    def _predict_coefs(self, model, C):
        """
        Predict a (1, c_dim, x_dim, y_dim) matrix of regression coefficients and offsets for each context
        beta[i,j] and mu[i,j] solve the regression problem y_j = beta[i,j] * x_i + mu[i,j]
        returns a numpy matrix
        """
        n = C.shape[0]
        betas = torch.zeros((n, self.x_dim, self.y_dim))
        mus = torch.zeros((n, self.x_dim, self.y_dim))
        for i in range(n):  # Predict per-sample to avoid OOM
            C_i = torch.Tensor(C[i])
            for t_x in range(self.x_dim):
                for t_y in range(self.y_dim):
                    task = torch.zeros(self.taskpair_dim)
                    task[t_x] = 1
                    task[self.x_dim + t_y] = 1
                    beta, mu = model(C_i.unsqueeze(0), task.unsqueeze(0))
                    beta, mu = beta.cpu().detach(), mu.cpu().detach()
                    betas[i, t_x, t_y] = beta.squeeze()
                    mus[i, t_x, t_y] = mu.squeeze()
        return betas.unsqueeze(0).numpy(), mus.unsqueeze(0).numpy()

    def predict_coefs(self, C, all_bootstraps=False):
        """
        Predict an (x_dim, y_dim) matrix of regression coefficients for each context
        Returns a numpy matrix (1 or bootstraps, n, x_dim, y_dim)
        """
        betas, mus = self._predict_coefs(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_coefs(model, C)
            betas = np.concatenate((betas, betas_i), axis=0)
            mus = np.concatenate((mus, mus_i), axis=0)
        if all_bootstraps:
            return betas, mus
        return betas.mean(axis=0), mus.mean(axis=0)

    def predict_correlation(self, C, all_bootstraps=False):
        """
        Requires x_dim == y_dim
        Predict an (x_dim, y_dim) matrix of squared Pearson's correlation coefficients for each context
        Returns a numpy matrix (n, x_dim, y_dim, 1 or bootstraps)
        """
        betas, mus = self.predict_coefs(C, all_bootstraps=True)
        betas_T = np.transpose(betas, (0, 1, 3, 2))
        rho = betas * betas_T
        if all_bootstraps:
            return rho
        return rho.mean(axis=0)

    def get_mse(self, C, X, Y, all_bootstraps=False):
        """
        Returns the MSE of the model on a dataset
        Returns a numpy array (1 or bootstraps, )
        """
        dataset = UnivariateDataset(C, X, Y, batch_size=1, device=self.device)
        mses = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            for c, t, x, y in dataset:
                betas, mus = model(c, t)
                mse = MSE(betas, mus, x, y, link_fn=self.link_fn).item()
                mses[i] += 1 / len(dataset) * mse
        if not all_bootstraps:
            return mses.mean()
        return mses
