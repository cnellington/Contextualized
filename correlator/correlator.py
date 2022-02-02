import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from correlator.dataset import Dataset


DTYPE = torch.float
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=0, encoder_width=25, encoder_layers=2, final_dense_size=10,
        activation=nn.ReLU):
        super(ContextualRegressorModule, self).__init__()
        self.context_encoder_in_shape = (context_dim, 1)
        self.context_encoder_out_shape = (final_dense_size,)
        taskpair_dim = max(x_dim, y_dim) * 2
        task_encoder_in_shape = (taskpair_dim, 1)
        task_encoder_out_shape = (final_dense_size,)
        self.use_archetypes = num_archetypes > 0

        default_layers = lambda: [layer for _ in range(0, encoder_layers - 2) for layer in (nn.Linear(encoder_width, encoder_width), activation())] \
            + [nn.Linear(encoder_width, final_dense_size), activation()]
        context_encoder_layers = [nn.Linear(context_dim, encoder_width), activation()] + default_layers()
        beta_task_encoder_layers = [nn.Linear(taskpair_dim, encoder_width), activation()] + default_layers()
        mu_task_encoder_layers = [nn.Linear(taskpair_dim, encoder_width), activation()] + default_layers()

        if self.use_archetypes:
            beta_task_encoder_layers = beta_task_encoder_layers[:-2] + [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]
            mu_task_encoder_layers = mu_task_encoder_layers[:-2] + [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]
            init_beta_archetypes = (torch.rand(num_archetypes, final_dense_size) - 0.5) * 1e-2
            init_mu_archetypes = (torch.rand(num_archetypes, final_dense_size) - 0.5) * 1e-2
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
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=0,
        encoder_width=25, encoder_layers=2, final_dense_size=10, l1=0,
        bootstraps=None, device=torch.device('cpu'), activation=nn.ReLU):
        module_params = {
            'context_dim': context_dim,
            'x_dim': x_dim,
            'y_dim': y_dim,
            'num_archetypes': num_archetypes,
            'encoder_width': encoder_width,
            'encoder_layers': encoder_layers,
            'final_dense_size': final_dense_size,
            'activation': activation
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
        self.to(device)

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(self.device)
        return self

    def _loss(self, outputs, X, Y):
        beta, mu, W_beta, W_mu, Z = outputs
        mse = MSE(beta, mu, X, Y)
        l1_beta = self.l1 * torch.norm(W_beta, 1)
        l1_mu = self.l1 * torch.norm(W_mu, 1)
        l1_z = self.l1 * torch.norm(Z, 1)
        return mse + l1_beta + l1_mu + l1_z

    def _fit(self, model, C, X, Y, epochs, batch_size, optimizer=torch.optim.Adam, lr=1e-3, validation_set=None, es_patience=None, es_epoch=0, silent=False):
        model.train()
        device = next(model.parameters()).device
        opt = optimizer(model.parameters(), lr=lr)
        db = Dataset(C, X, Y, device=device)
        if validation_set is not None:
            Cval, Xval, Yval = validation_set
            val_db = Dataset(Cval, Xval, Yval, device=device)
        progress_bar = tqdm(range(epochs), disable=silent)
        min_loss = np.inf  # for early stopping
        es_count = 0
        for epoch in progress_bar:
            for batch_start in range(0, len(X), batch_size):
                data_paired = db.load_data(batch_start=batch_start, batch_size=batch_size)  # todo: revise load_data to use a combinatorial product of indices for C, X, Y to make batch_size useful
                loss = None
                for C_paired, T_paired, X_paired, Y_paired in zip(*data_paired):  # This makes batch_size irrelevant. Maybe remove?
                    outputs = model(C_paired.unsqueeze(0), T_paired.unsqueeze(0))
                    loss = self._loss(outputs, X_paired.unsqueeze(0), Y_paired.unsqueeze(0))  # this loop w unsqueeze is a hacky fix for the todo above
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
        returns a numpy matrix
        """
        n = C.shape[0]
        betas = torch.zeros((n * self.x_dim * self.y_dim, 1))
        mus = torch.zeros((n * self.x_dim * self.y_dim, 1))
        X_temp = np.zeros((n, self.x_dim))
        Y_temp = np.zeros((n, self.y_dim))
        db = Dataset(C, X_temp, Y_temp, device=self.device)
        C_paired, T_paired, _, _ = db.load_data(batch_start=0, batch_size=1) 
        betas, mus, _, _, _ = model(C_paired, T_paired) 
        betas, mus = betas.cpu().detach().numpy(), mus.cpu().detach().numpy()
        for i in range(1, n):  # Predict per-sample to avoid OOM
                C_paired, T_paired, _, _ = db.load_data(batch_start=i, batch_size=1) 
                betas_i, mus_i, _, _, _ = model(C_paired, T_paired)
                betas_i, mus_i = betas_i.cpu().detach().numpy(), mus_i.cpu().detach().numpy()
                betas = np.concatenate((betas, betas_i))
                mus = np.concatenate((mus, mus_i))
        betas = betas.reshape((n, self.x_dim, self.y_dim, 1))
        mus = mus.reshape((n, self.x_dim, self.y_dim, 1))
        return betas, mus

    def predict_regression(self, C, all_bootstraps=False):
        """
        Predict an (x_dim, y_dim) matrix of regression coefficients for each context
        Returns a numpy matrix (n, x_dim, y_dim, 1 or bootstraps)
        """
        betas, mus = self._predict_regression(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_regression(model, C)
            betas = torch.concatenate((betas, betas_i), dim=-1)
            mus = torch.concatenate((mus, mus_i), dim=-1)
        if all_bootstraps:
            return betas, mus
        return betas.mean(axis=-1), mus.mean(axis=-1)

    def predict_correlation(self, C, all_bootstraps=False):
        """
        Predict an (x_dim, y_dim) matrix of squared Pearson's correlation coefficients for each context
        Returns a numpy matrix (n, x_dim, y_dim, 1 or bootstraps)
        """
        betas, mus = self.predict_regression(C, all_bootstraps=True)
        betas_T = np.transpose(betas, (0, 2, 1, 3))
        rho = betas * betas_T
        if all_bootstraps:
            return rho
        return rho.mean(axis=-1)

    def get_mse(self, C, X, Y, all_bootstraps=False):
        """
        Returns the MSE of the model on a dataset
        Returns a numpy array (1 or bootstraps, )
        """
        n = len(X)
        db = Dataset(C, X, Y, device=self.device)
        mses = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            for batch_start in range(0, n):
                C_paired, T_paired, X_paired, Y_paired = db.load_data(batch_start=batch_start, batch_size=1)
                betas, mus, _, _, _ = model(C_paired, T_paired)
                mse = MSE(betas, mus, X_paired, Y_paired).item()
                mses[i] += 1 / n * mse
        if not all_bootstraps:
            return mses.mean()
        return mses
