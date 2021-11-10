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

    def __init__(self, context_shape, task_shape, num_archetypes=2, encoder_width=25, final_dense_size=10):
        super(ContextualRegressorModule, self).__init__()
        self.context_encoder_in_shape = (context_shape[-1], 1)
        self.context_encoder_out_shape = (final_dense_size,)
        task_encoder_in_shape = (task_shape[-1] * 2, 1)
        task_encoder_out_shape = (final_dense_size,)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_shape[-1], encoder_width), 
            nn.ReLU(),
            nn.Linear(encoder_width, final_dense_size), 
            nn.ReLU(), 
        )
        self.beta_task_encoder = nn.Sequential(
            nn.Linear(task_shape[-1], encoder_width), 
            nn.ReLU(),
            nn.Linear(encoder_width, num_archetypes),
            nn.Softmax(dim=1), 
        )
        self.mu_task_encoder = nn.Sequential(
            nn.Linear(task_shape[-1], encoder_width), 
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
    def __init__(self, context_shape, task_shape, num_archetypes=2, encoder_width=25, final_dense_size=10):
        self.model = ContextualRegressorModule(context_shape, task_shape, 
                        num_archetypes=num_archetypes,
                        encoder_width=encoder_width,
                        final_dense_size=final_dense_size)
        self._px = None
        self._py = None

    def get_mse(self, C, X, Y):
        C, T, X, Y = Dataset(C, X, Y).load_data()
        betas, mus = self.model(C, T)
        mse = MSE(betas, mus, X, Y)
        return mse.detach().item()

    def fit(self, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, es_patience=None):
        self._px, self._py = X.shape[-1], Y.shape[-1]
        self.model.train()
        opt = optimizer(self.model.parameters(), lr=lr)
        db = Dataset(C, X, Y)
        # todo: log nondecreasing loss for early stopping
        for _ in tqdm(range(epochs)):
            for batch_start in range(0, len(X) + batch_size - 1, batch_size):
                C, T, X, Y = db.load_data(batch_start=batch_start, batch_size=batch_size) 
                betas, mus = self.model(C, T)
                loss = MSE(betas, mus, X, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.eval()

    def predict_regression(self, C):
        """
        Predict a (p_x, p_y) matrix of regression coefficients and offsets for each context
        beta[i,j] and mu[i,j] solve the regression problem y_j = beta[i,j] * x_i + mu[i,j]
        """
        n = C.shape[0]
        X_temp = np.zeros((n, self._px))
        Y_temp = np.zeros((n, self._py))
        C, T, _, _ = to_pairwise(C, X_temp, Y_temp)
        betas, mus = self.model(C, T)
        betas = betas.detach().numpy().reshape((n, self._px, self._py))
        mus = mus.detach().numpy().reshape((n, self._px, self._py))
        return betas, mus

    def predict_correlation(self, C):
        """
        Predict a (p_x, p_y) matrix of squared Pearson's correlation coefficients for each context
        """
        betas, mus = self.predict_regression(C)
        betas_T = np.transpose(betas, axes=(0, 2, 1))
        rho = betas * betas_T
        return rho

