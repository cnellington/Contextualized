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
from contextualized.regression.trainers import RegressionTrainer
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

    def quicktest(model):
        print(f'{type(model)} quicktest')
        dataloader = model.dataloader(C, X, Y, batch_size=32)
        trainer = RegressionTrainer(max_epochs=1)
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        print()

    # Naive Multivariate
    model = NaiveContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['identity']},
        link_fn=LINK_FUNCTIONS['identity'])
    quicktest(model)

    model = NaiveContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['softmax']},
        link_fn=LINK_FUNCTIONS['identity'])
    quicktest(model)

    # Naive Multivariate
    model = NaiveContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['identity']},
        link_fn=LINK_FUNCTIONS['softmax'])
    quicktest(model)

    model = NaiveContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim,
        encoder_kwargs={'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['softmax']},
        link_fn=LINK_FUNCTIONS['softmax'])
    quicktest(model)

    # Subtype Multivariate
    model = ContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim)
    quicktest(model)

    # Multitask Multivariate
    model = MultitaskContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim)
    quicktest(model)

    # Tasksplit Multivariate
    model = TasksplitContextualizedRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim)
    quicktest(model)

    # Univariate
    model = ContextualizedUnivariateRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim)
    quicktest(model)

    # Tasksplit Univariate
    model = TasksplitContextualizedUnivariateRegression(context_dim=c_dim, x_dim=x_dim, y_dim=y_dim)
    quicktest(model)
