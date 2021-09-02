import time
import unittest
import numpy as np
import torch
from experiments.simulation import dataloader
from model.correlator import ContextualCorrelator


dtype = torch.float
device = torch.device('cpu')


def loss(beta, xi, xj):
    loss_val = (beta.unsqueeze(-1) * xi - xj).pow(2).sum()
    return loss_val


class TestCorrelator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCorrelator, self).__init__(*args, **kwargs)

    def test_convergence(self):
        k, p, c, c_n, k_arch = 4, 8, 4, 100, 2
        dl = dataloader.Dataloader(k, p, c)
        C, X, Cov, dist_ids = dl.simulate(c_n)
        C, Ti, Tj, Xi, Xj, sample_ids = dl.split_tasks(C, X)
        C = torch.tensor(C, dtype=dtype, device=device)
        Xi = torch.tensor(Xi, dtype=dtype, device=device)
        Xj = torch.tensor(Xj, dtype=dtype, device=device)
        T_in = np.hstack((Ti, Tj))
        T_in = torch.tensor(T_in, dtype=dtype, device=device)

        cc = ContextualCorrelator(C.shape, T_in.shape, num_archetypes=k_arch)
        cc.train()
        lr = 1e-3
        opt = torch.optim.Adam(cc.parameters(), lr=lr)
        loss_vals = []
        for _ in range(10):
            start = time.time()
            beta_pred = cc(C, T_in)
            loss_val = loss(beta_pred, Xi, Xj)
            loss_detached = float(loss_val.detach().numpy())
            loss_vals.append(loss_detached)
            opt.zero_grad()
            print('forward', time.time() - start)
            start = time.time()
            loss_val.backward()
            print('backward', time.time() - start)
            opt.step()
            print()
        
        learn_count = 0
        for i in range(1, len(loss_vals)):
            if loss_vals[i] < loss_vals[i - 1]:
                learn_count += 1
        assert learn_count > 0.8 * len(loss_vals), f'learned on {learn_count} steps'


if __name__ == '__main__':
    unittest.main()
