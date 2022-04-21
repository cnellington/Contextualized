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
from contextualized.regression.lightning_modules import *
from contextualized.regression.trainers import *
from contextualized.regression import ENCODERS, LINK_FUNCTIONS

if __name__ == '__main__':
    n = 100
    c_dim = 4
    x_dim = 5
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

    def quicktest(model, univariate=False, correlation=False):
        print(f'{type(model)} quicktest')
        dataloader = None
        trainer = None
        if correlation:
            dataloader = model.dataloader(C, X, batch_size=32)
            trainer = CorrelationTrainer(max_epochs=1)
        else:
            dataloader = model.dataloader(C, X, Y, batch_size=32)
            trainer = RegressionTrainer(max_epochs=1)
        y_preds = trainer.predict_y(model, dataloader)
        y_true = Y
        if univariate:
            y_true = np.tile(y_true[:,:,np.newaxis], (1, 1, X.shape[-1]))
        if correlation:
            y_true = np.tile(X[:,:,np.newaxis], (1, 1, X.shape[-1]))
        err_init = ((y_true - y_preds)**2).mean()
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        if correlation:
            rhos = trainer.predict_correlation(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        err_trained = ((y_true - y_preds)**2).mean()
        assert err_trained < err_init
        print()
        
    def correlation_quicktest(model):
        print(f'{type(model)} quicktest')
        dataloader = model.dataloader(C, X, batch_size=32)
        trainer = CorrelationTrainer(max_epochs=1)
        y_preds = trainer.predict_y(model, dataloader)
        y_true = np.tile(X[:,:,np.newaxis], (1, 1, X.shape[-1]))
        err_init = ((y_true - y_preds)**2).mean()
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        network_preds = trainer.predict_network(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        err_trained = ((y_true - y_preds)**2).mean()
        assert err_trained < err_init
        print()

    # Naive Multivariate
    model = NaiveContextualizedRegression(c_dim, x_dim, y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['identity']},
        link_fn=LINK_FUNCTIONS['identity'])
    quicktest(model)

    model = NaiveContextualizedRegression(c_dim, x_dim, y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['softmax']},
        link_fn=LINK_FUNCTIONS['identity'])
    quicktest(model)

    model = NaiveContextualizedRegression(c_dim, x_dim, y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['identity']},
        link_fn=LINK_FUNCTIONS['logistic'])
    quicktest(model)

    model = NaiveContextualizedRegression(c_dim, x_dim, y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['softmax']},
        link_fn=LINK_FUNCTIONS['logistic'])
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

    # Naive Univariate
    model = ContextualizedUnivariateRegression(c_dim, x_dim, y_dim)
    quicktest(model, univariate=True)

    # Tasksplit Univariate
    model = TasksplitContextualizedUnivariateRegression(c_dim, x_dim, y_dim)
    quicktest(model, univariate=True)
    
    # Correlation
    model = ContextualizedCorrelation(c_dim, x_dim)
    quicktest(model, correlation=True)
    
    # Tasksplit Correlation
    model = TasksplitContextualizedCorrelation(c_dim, x_dim)
    quicktest(model, correlation=True)
