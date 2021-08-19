from experiments.simulation.dataloader import Dataloader
from model.correlator import ContextualCorrelator
import numpy as np
import torch

dtype = torch.float
device = torch.device('cpu')


def loss(beta, xi, xj):
    loss_val = (beta.unsqueeze(-1) * xi - xj).pow(2)
    return loss_val


def main():
    dl = Dataloader(2, 2, 1)
    C, X, Cov, dist_ids = dl.simulate(2)
    C, Ti, Tj, Xi, Xj, sample_ids = dl.split_tasks(C, X)
    C = torch.tensor(C, dtype=dtype, device=device)
    X_in = np.hstack((Xi, Xj))
    X_in = torch.tensor(X_in, dtype=dtype, device=device)
    Xi = torch.tensor(Xi, dtype=dtype, device=device)
    Xj = torch.tensor(Xj, dtype=dtype, device=device)
    
    cc = ContextualCorrelator(C.shape)
    cc.train()
    lr = 1e-3
    opt = torch.optim.Adam(cc.parameters(), lr=lr)
    for _ in range(10000):
        beta_pred = cc(C)
        loss_val = loss(beta_pred, Xi, Xj)
        opt.zero_grad()
        loss_val.sum().backward()
        print(loss_val.sum().detach())
        opt.step()    


if __name__ == '__main__':
    main()
