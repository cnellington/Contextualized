import time
import unittest
import numpy as np
import torch
from model.dataset import GaussianSimulator, Dataset
from model.correlator import ContextualRegressor
from model.trainer import Trainer, MSE, DTYPE, DEVICE 


class TestCorrelator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelator, self).__init__(*args, **kwargs)

    def test_convergence(self):
        k, p, c, k_n = 4, 8, 4, 10
        k_arch = k * p ** 2
        epochs = 10
        runs = 10
        converges = np.zeros(runs)
        for i in range(10):
            sim = GaussianSimulator(p, k, c)
            C, X = sim.gen_samples(k_n)
            db = Dataset(C, X, X)
            C_test, T_test, X_test, Y_test = db.get_test()
            model = ContextualRegressor(db.C.shape, db.T.shape, num_archetypes=k_arch)
            betas, mus = model(C_test, T_test)
            init_loss = MSE(betas, mus, X_test, Y_test).detach().item()
            optimizer = torch.optim.Adam
            trainer = Trainer(model, optimizer, db)
            trainer.train(epochs)
            betas, mus = model(C_test, T_test)
            stop_loss = MSE(betas, mus, X_test, Y_test).detach().item()
            converges[i] = stop_loss < init_loss
        assert converges.all()


if __name__ == '__main__':
    unittest.main()
