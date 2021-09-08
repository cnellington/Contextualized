import time
import unittest
import numpy as np
import torch
from model.dataset import SimulationDataset
from model.correlator import ContextualCorrelator
from model.trainer import Trainer, MSE


dtype = torch.float
device = torch.device('cpu')


class TestCorrelator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelator, self).__init__(*args, **kwargs)

    def test_convergence(self):
        k, p, c, k_n, k_arch = 4, 8, 4, 10, 2
        db = SimulationDataset(p, k, c)
        X, T, C, B = db.gen_samples(k_n)

        model = ContextualCorrelator(C.shape, T.shape, num_archetypes=k_arch)
        init_loss = MSE(model(C, T), X[:,0], X[:,1]).detach().numpy()
        optimizer = torch.optim.Adam
        trainer = Trainer(model, optimizer, db)
        trainer.train(10)
        stop_loss = MSE(model(C, T), X[:,0], X[:,1]).detach().numpy()
        assert stop_loss < init_loss


if __name__ == '__main__':
    unittest.main()
