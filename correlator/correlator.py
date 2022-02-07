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
    residual = beta * x + mu - y
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
        self.context_dim = context_dim
        self.task_dim = max(x_dim, y_dim)
        self.final_dense_size = final_dense_size
        self.use_archetypes = num_archetypes > 0

        hidden_layers = lambda: [layer for _ in range(0, encoder_layers - 2) for layer in (nn.Linear(encoder_width, encoder_width), activation())]
        context_encoder_layers = [nn.Linear(self.context_dim, encoder_width), activation()] + hidden_layers() + [nn.Linear(encoder_width, self.final_dense_size)]
        task_encoder_layers = [nn.Linear(self.task_dim * 2, encoder_width), activation()] + hidden_layers() + [nn.Linear(encoder_width, self.final_dense_size * 2)]

        if self.use_archetypes:
            task_encoder_layers = task_encoder_layers[:-1] + [nn.Linear(encoder_width, num_archetypes), nn.Softmax(dim=1)]  # remove dense, add softmax
            init_archetypes = (torch.rand(num_archetypes, final_dense_size * 2) - 0.5) * 2e-2  # Unif[-1e-2, 1e-2]
            self.archetypes = nn.parameter.Parameter(init_archetypes, requires_grad=True)

        self.context_encoder = nn.Sequential(*context_encoder_layers)
        self.task_encoder = nn.Sequential(*task_encoder_layers)

    def forward(self, c, t):
        batch_size = c.shape[0]
        Z = self.context_encoder(c)
        A = self.task_encoder(t)
        if self.use_archetypes:
            batch_archetypes = self.archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
            W = torch.bmm(A.unsqueeze(1), batch_archetypes)
        else:
            W = A
        W = W.reshape((batch_size, 2, self.final_dense_size))
        out = torch.bmm(W, Z.unsqueeze(-1)).squeeze(-1)
        beta, mu = out.T
        return beta, mu, W, Z


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
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_dim = max(x_dim, y_dim)
        self.l1 = l1
        self.to(device)

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(self.device)
        return self

    def _loss(self, outputs, X, Y):
        beta, mu, W, Z = outputs
        mse = MSE(beta, mu, X, Y)
        l1_w = self.l1 * torch.norm(W, 1)
        l1_z = self.l1 * torch.norm(Z, 1)
        return mse + l1_w + l1_z

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
                # Train context encoder
                for C_paired, T_paired, X_paired, Y_paired in zip(*data_paired):  # This makes batch_size irrelevant. Maybe remove?
                    outputs = model(C_paired.unsqueeze(0), T_paired.unsqueeze(0))
                    loss = self._loss(outputs, X_paired.unsqueeze(0), Y_paired.unsqueeze(0))  # this loop w unsqueeze is a hacky fix for the todo above
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # Update UI and check validation set for early stopping
                train_desc = f'[Train MSE: {loss.item():.4f}] [Sample: {batch_start}/{len(X)}] Epoch'
                if validation_set is not None:
                    Cval_paired, Tval_paired, Xval_paired, Yval_paired = val_db.load_data(batch_size=batch_size)
                    val_outputs = model(Cval_paired, Tval_paired)
                    val_loss = self._loss(val_outputs, Xval_paired, Yval_paired).item()
                    if es_patience is not None and epoch >= es_epoch:
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
        betas = np.zeros((n, self.x_dim, self.y_dim, 1))
        mus = np.zeros((n, self.x_dim, self.y_dim, 1))
        for i in range(n):  # Predict per-sample to avoid OOM
            C_i = torch.Tensor(C[i])
            for t_x in range(self.x_dim):
                for t_y in range(self.y_dim):
                    task = torch.zeros(self.task_dim * 2)
                    task[t_x] = 1
                    task[self.task_dim + t_y] = 1
                    beta, mu, _, _ = model(C_i.unsqueeze(0), task.unsqueeze(0))
                    beta, mu = beta.cpu().detach().numpy(), mu.cpu().detach().numpy()
                    betas[i, t_x, t_y] = beta
                    mus[i, t_x, t_y] = mu
        return betas, mus

    def predict_regression(self, C, all_bootstraps=False):
        """
        Predict an (x_dim, y_dim) matrix of regression coefficients for each context
        Returns a numpy matrix (n, x_dim, y_dim, 1 or bootstraps)
        """
        betas, mus = self._predict_regression(self.models[0], C)
        for model in self.models[1:]:
            betas_i, mus_i = self._predict_regression(model, C)
            betas = np.concatenate((betas, betas_i), axis=-1)
            mus = np.concatenate((mus, mus_i), axis=-1)
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
                betas, mus, _, _ = model(C_paired, T_paired)
                mse = MSE(betas, mus, X_paired, Y_paired).item()
                mses[i] += 1 / n * mse
        if not all_bootstraps:
            return mses.mean()
        return mses
