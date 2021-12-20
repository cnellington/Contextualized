import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from correlator.dataset import Dataset, to_pairwise


DTYPE = torch.float 
DEVICE = torch.device('cuda') 


def MSE(beta, mu, x, y):
    residual = beta.squeeze() * x + mu.squeeze() - y
    return residual.pow(2).mean()


class ContextualRegressorModule(nn.Module):
    """
    rho(c) = beta(c) * beta'(c)
    beta(c) = sigma(A @ f(c) + b)
    
    beta_{a_i, b_j} = sigma(A(t_i, t_j) @ f(c) + b)
    f(c) = sigma(dense(c))
    A(t_i, t_j) = <g(t_i, t_j), A_{1..K}>
    g(t_i, t_j) = softmax(dense(t_i, t_j))
    """
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=None, encoder_width=25, encoder_layers=2, final_dense_size=10):
        super(ContextualRegressorModule, self).__init__()
        self.context_encoder_in_shape = (context_dim, 1)
        self.context_encoder_out_shape = (final_dense_size,)
        taskpair_dim = max(x_dim, y_dim) * 2
        task_encoder_in_shape = (taskpair_dim, 1)
        task_encoder_out_shape = (final_dense_size,)
        self.use_archetypes = num_archetypes is not None

        default_layers = lambda: [layer for _ in range(0, encoder_layers - 2) for layer in (nn.Linear(encoder_width, encoder_width), nn.ReLU())] \
            + [nn.Linear(encoder_width, final_dense_size), nn.ReLU()] 
        context_encoder_layers = [nn.Linear(context_dim, encoder_width), nn.ReLU()] + default_layers()
        beta_task_encoder_layers = [nn.Linear(taskpair_dim, encoder_width), nn.ReLU()] + default_layers()
        mu_task_encoder_layers = [nn.Linear(taskpair_dim, encoder_width), nn.ReLU()] + default_layers()
        
        if self.use_archetypes:
            beta_task_encoder_layers = beta_task_encoder_layers[:-2] + [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]
            mu_task_encoder_layers = mu_task_encoder_layers[:-2] + [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]
            init_beta_archetypes = torch.rand(num_archetypes, final_dense_size)
            init_mu_archetypes = torch.rand(num_archetypes, final_dense_size)
            self.beta_archetypes = nn.parameter.Parameter(init_beta_archetypes, requires_grad=True)
            self.mu_archetypes = nn.parameter.Parameter(init_mu_archetypes, requires_grad=True)
        
        self.context_encoder = nn.Sequential(*context_encoder_layers)
        self.beta_task_encoder = nn.Sequential(*beta_task_encoder_layers)
        self.mu_task_encoder = nn.Sequential(*mu_task_encoder_layers)
        self.flatten = nn.Flatten(0, 1)
        
    def forward(self, c, t):
        Z = self.context_encoder(c).unsqueeze(-1)
        W_beta, W_mu = None, None
        if self.use_archetypes:
            A_beta = self.beta_task_encoder(t).unsqueeze(1)
            A_mu = self.mu_task_encoder(t).unsqueeze(1)
            batch_size = A_beta.shape[0]
            batch_beta_archetypes = self.beta_archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
            batch_mu_archetypes = self.mu_archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
            W_beta = torch.bmm(A_beta, batch_beta_archetypes)
            W_mu = torch.bmm(A_mu, batch_mu_archetypes)
        else:
            W_beta = self.beta_task_encoder(t).unsqueeze(1)
            W_mu = self.mu_task_encoder(t).unsqueeze(1)
        beta = torch.bmm(W_beta, Z)
        mu = torch.bmm(W_mu, Z)
        return self.flatten(beta), self.flatten(mu), W_beta, W_mu, Z


class ContextualCorrelator:
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=None, encoder_width=25, encoder_layers=2, final_dense_size=10, l1=0, bootstraps=None):
        module_params = {
            'context_dim': context_dim,
            'x_dim': x_dim, 
            'y_dim': y_dim,
            'num_archetypes': num_archetypes,
            'encoder_width': encoder_width,
            'encoder_layers': encoder_layers,
            'final_dense_size': final_dense_size,
        }
        if bootstraps is None:
            self.model = ContextualRegressorModule(**module_params)
            self.models = [self.model]
        else:
            self.model = None
            self.models = [ContextualRegressorModule(**module_params) for _ in range(bootstraps)]
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.l1 = l1

    def _loss(self, outputs, X, Y):
        beta, mu, W_beta, W_mu, Z = outputs
        mse = MSE(beta, mu, X, Y)
        l1_beta = self.l1 * torch.norm(W_beta, 1)
        l1_mu = self.l1 * torch.norm(W_mu, 1)
        l1_z = self.l1 * torch.norm(Z, 1)
        return mse + l1_beta + l1_mu + l1_z

    def _fit(self, model, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, validation_set=None, es_patience=None, es_epoch=0, silent=False):
        model.train()
        opt = optimizer(model.parameters(), lr=lr)
        db = Dataset(C, X, Y)
        if validation_set is not None:
            Cval, Xval, Yval = validation_set
            val_db = Dataset(Cval, Xval, Yval)
        progress_bar = tqdm(range(epochs), disable=silent)
        min_loss = np.inf
        es_count = 0
        for epoch in progress_bar:
            for batch_start in range(0, len(X), batch_size):
                C_paired, T_paired, X_paired, Y_paired = db.load_data(batch_start=batch_start, batch_size=batch_size) 
                outputs = model(C_paired, T_paired)
                loss = self._loss(outputs, X_paired, Y_paired)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_desc = f'[Train MSE: {loss.item():.4f}] [Sample: {batch_start}/{len(X)}] Epoch'
                if validation_set is not None:  # Validation set loss
                    Cval_paired, Tval_paired, Xval_paired, Yval_paired = val_db.load_data(batch_size=batch_size)
                    val_outputs = model(Cval_paired, Tval_paired)
                    val_loss = self._loss(val_outputs, Xval_paired, Yval_paired).item()
                    if es_patience is not None and epoch >= es_epoch:  # Early stopping
                        if val_loss < min_loss:
                            min_loss = val_loss
                            es_count = 0
                        else:
                            es_count += 1
                    train_desc = f"[Val MSE: {val_loss:.4f}] " + train_desc
                progress_bar.set_description(train_desc)
                if es_patience is not None and es_count > es_patience:
                    model.eval()
                    return
        model.eval()

    def fit(self, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, validation_set=None, es_patience=None, es_epoch=0, silent=False):
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

    def _predict_regression(self, model, C):
        """
        Predict a (p_x, p_y) matrix of regression coefficients and offsets for each context
        beta[i,j] and mu[i,j] solve the regression problem y_j = beta[i,j] * x_i + mu[i,j]
        """
        n = C.shape[0]
        X_temp = np.zeros((n, self.x_dim))
        Y_temp = np.zeros((n, self.y_dim))
        C, T, _, _ = to_pairwise(C, X_temp, Y_temp)
        betas, mus, _, _, _ = model(C, T)
        betas = torch.reshape(betas.detach(), (n, self.x_dim, self.y_dim, 1))
        mus = torch.reshape(mus.detach(), (n, self.x_dim, self.y_dim, 1))
        return betas, mus

    def predict_regression(self, C, all_bootstraps=False):
        betas, mus = self._predict_regression(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_regression(model, C)
            betas = torch.cat((betas, betas_i), dim=-1)
            mus = torch.cat((mus, mus_i), dim=-1)
        if all_bootstraps:
            return betas, mus
        return betas.mean(axis=-1), mus.mean(axis=-1)

    def predict_correlation(self, C, all_bootstraps=False):
        """
        Predict a (p_x, p_y) matrix of squared Pearson's correlation coefficients for each context
        """
        betas, mus = self.predict_regression(C, all_bootstraps=True)
        betas_T = np.transpose(betas, axes=(0, 2, 1, 3))
        rho = betas * betas_T
        if all_bootstraps:
            return rho
        return rho.mean(axis=-1)

    def get_mse(self, C, X, Y, all_bootstraps=False):
        """
        Returns the MSE of the model on a dataset
        """
        n = len(X)
        db = Dataset(C, X, Y)
        mses = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            for batch_start in range(0, n):
                C_paired, T_paired, X_paired, Y_paired = db.load_data(batch_start=batch_start, batch_size=1) 
                betas, mus, _, _, _ = model(C_paired, T_paired)
                mse = MSE(betas, mus, X_paired, Y_paired).detach().item()
                mses[i] += 1 / n * mse
        if not all_bootstraps:
            return mses.mean()
        return mses
