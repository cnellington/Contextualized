import numpy as np
import torch


DTYPE = torch.float 
DEVICE = torch.device('cpu') 


def MSE(beta, xi, xj):
    return (beta.unsqueeze(-1) * xi - xj).pow(2).mean()


class Trainer:
    def __init__(self, model, optimizer, database):
        self.model = model
        self.optimizer = optimizer
        self.db = database

    def train(self, epochs, batch_size=32, lr=1e-3):
        self.model.train()
        opt = self.optimizer(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            X, T, C = self.db.load_data(batch_size=batch_size, device=DEVICE)
            betas = self.model(C, T)
            loss = MSE(betas, X[:,0], X[:,1])
            opt.zero_grad()
            loss.backward()
            opt.step()

