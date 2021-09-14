import numpy as np
import torch


DTYPE = torch.float 
DEVICE = torch.device('cpu') 


def MSE(beta, mu, xi, xj):
    residual = beta.squeeze() * xi + mu.squeeze() - xj
    return residual.pow(2).mean()


class Trainer:
    def __init__(self, model, optimizer, database):
        self.model = model
        self.optimizer = optimizer
        self.db = database

    def train(self, epochs, batch_size=32, lr=1e-3):
        self.model.train()
        opt = self.optimizer(self.model.parameters(), lr=lr)
        while self.db.epoch < epochs:
            C, T, X = self.db.load_data(batch_size=batch_size, device=DEVICE)
            betas, mus = self.model(C, T)
            loss = MSE(betas, mus, X[:,0], X[:,1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    def test(self):
        self.model.eval()
        C, T, X = self.db.get_test()
        betas, mus = self.model(C, T)
        loss = MSE(betas, mus, X[:,0], X[:,1])
        return loss.detach().item()

