import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from contextualized.modules import NGAM, MLP, SoftSelect, Explainer
from contextualized.regression import RegressionTrainer, ContextualizedUnivariateRegression, TasksplitContextualizedUnivariateRegression


ENCODERS = {
    'mlp': MLP,
    'ngam': NGAM,
}
METAMODELS = ['simple', 'subtype', 'multitask', 'tasksplit']
LINK_FNS = [
    lambda x: x,
    lambda x: F.softmax(x, dim=1)
]


class NetworkTrainer(RegressionTrainer):
    def predict_network(self, model, dataloader):
        betas, _ = super().predict_params(model, dataloader)
        rhos = betas * np.transpose(betas, (0, 2, 1))
        return rhos


class ContextualizedCorrelation(ContextualizedUnivariateRegression):
    def __init__(self, context_dim, x_dim, **kwargs):
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
    
    def dataloader(self, C, X, **kwargs):
        return super().dataloader(C, X, X, **kwargs)


if __name__ == '__main__':
    n = 100
    c_dim = 4
    x_dim = 2
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
        dataloader = model.dataloader(C, X, batch_size=32)
        trainer = NetworkTrainer(max_epochs=1)
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        network_preds = trainer.predict_network(model, dataloader)

    # Correlation
    model = ContextualizedCorrelation(c_dim, x_dim)
    quicktest(model)
