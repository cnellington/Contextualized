import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from correlator.dataset import Dataset, to_pairwise


DTYPE = torch.float 
DEVICE = torch.device('cpu') 


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

    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=10, encoder_width=25, final_dense_size=10):
        super(ContextualRegressorModule, self).__init__()
        self.context_encoder_in_shape = (context_dim, 1)
        self.context_encoder_out_shape = (final_dense_size,)
        taskpair_dim = max(x_dim, y_dim) * 2
        task_encoder_in_shape = (taskpair_dim, 1)
        task_encoder_out_shape = (final_dense_size,)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, encoder_width), 
            nn.ReLU(),
            nn.Linear(encoder_width, final_dense_size), 
            nn.ReLU(), 
        )
        self.beta_task_encoder = nn.Sequential(
            nn.Linear(taskpair_dim, encoder_width), 
            nn.ReLU(),
            nn.Linear(encoder_width, num_archetypes),
            nn.Softmax(dim=1), 
        )
        self.mu_task_encoder = nn.Sequential(
            nn.Linear(taskpair_dim, encoder_width), 
            nn.ReLU(),
            nn.Linear(encoder_width, num_archetypes),
            nn.Softmax(dim=1), 
        )
        init_beta_archetypes = torch.rand(num_archetypes, final_dense_size)
        init_mu_archetypes = torch.rand(num_archetypes, final_dense_size)
        self.beta_archetypes = nn.parameter.Parameter(init_beta_archetypes, requires_grad=True)
        self.mu_archetypes = nn.parameter.Parameter(init_mu_archetypes, requires_grad=True)
        self.flatten = nn.Flatten(0, 1)

    def forward(self, c, t):
        Z = self.context_encoder(c).unsqueeze(-1)
        A_beta = self.beta_task_encoder(t).unsqueeze(1)
        A_mu = self.mu_task_encoder(t).unsqueeze(1)
        batch_size = A_beta.shape[0]
        batch_beta_archetypes = self.beta_archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_mu_archetypes = self.mu_archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
        W_beta = torch.bmm(A_beta, batch_beta_archetypes)
        W_mu = torch.bmm(A_mu, batch_mu_archetypes)
        beta = torch.bmm(W_beta, Z)
        mu = torch.bmm(W_mu, Z)
        return self.flatten(beta), self.flatten(mu)


class ContextualCorrelator:
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=10, encoder_width=25, final_dense_size=10, bootstraps=None):
        module_params = {
            'context_dim': context_dim,
            'x_dim': x_dim, 
            'y_dim': y_dim,
            'num_archetypes': num_archetypes,
            'encoder_width': encoder_width,
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

    def get_mse(self, C, X, Y):
        C, T, X, Y = Dataset(C, X, Y).load_data()
        betas, mus = self.model(C, T)
        mse = MSE(betas, mus, X, Y)
        return mse.detach().item()

    def _fit(self, model, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, es_patience=None):
        model.train()
        opt = optimizer(model.parameters(), lr=lr)
        db = Dataset(C, X, Y)
        # todo: log nondecreasing loss for early stopping
        for _ in tqdm(range(epochs)):
            for batch_start in range(0, len(X) + batch_size - 1, batch_size):
                C, T, X, Y = db.load_data(batch_start=batch_start, batch_size=batch_size) 
                betas, mus = model(C, T)
                loss = MSE(betas, mus, X, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()

    def fit(self, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, es_patience=None):
        fit_params = {
            'C': C, 'X': X, 'Y': Y, 'epochs': epochs, 'batch_size': batch_size, 
            'optimizer': optimizer, 'lr': lr, 'es_patience': es_patience
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
        betas, mus = model(C, T)
        betas = betas.detach().numpy().reshape((n, self.x_dim, self.y_dim, 1))
        mus = mus.detach().numpy().reshape((n, self.x_dim, self.y_dim, 1))
        return betas, mus

    def predict_regression(self, C, bootstrap_mean=True):
        betas, mus = self._predict_regression(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_regression(model, C)
            betas = np.concatenate((betas, betas_i), axis=-1)
            mus = np.concatenate((mus, mus_i), axis=-1)
        if bootstrap_mean:
            return betas.mean(axis=-1), mus.mean(axis=-1)
        return betas, mus

    def predict_correlation(self, C, bootstrap_mean=True):
        """
        Predict a (p_x, p_y) matrix of squared Pearson's correlation coefficients for each context
        """
        betas, mus = self.predict_regression(C, bootstrap_mean=False)
        betas_T = np.transpose(betas, axes=(0, 2, 1, 3))
        rho = betas * betas_T
        if bootstrap_mean:
            return rho.mean(axis=-1)
        return rho

