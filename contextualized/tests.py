import os
import unittest
import numpy as np
import torch
from contextualized.modules import SoftSelect, Explainer, MLP, NGAM, Linear
from contextualized.easy import (
    ContextualizedRegressor,
    ContextualizedBayesianNetworks,
    ContextualizedCorrelationNetworks,
)
from contextualized.baselines import BayesianNetwork, CorrelationNetwork
from contextualized.utils import save, load


class TestModules(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Shared data setup.
        """
        self.N_SAMPLES = 100
        self.X_DIM = 10
        self.Y_DIM = 5
        self.K_ARCHETYPES = 3
        self.WIDTH = 50
        self.LAYERS = 5
        self.X_data = torch.rand((self.N_SAMPLES, self.X_DIM))
        self.IN_DIMS = (3, 4)
        self.OUT_SHAPE = (5, 6)
        self.Z1 = torch.randn(self.N_SAMPLES, self.IN_DIMS[0])
        self.Z2 = torch.randn(self.N_SAMPLES, self.IN_DIMS[1])

    def test_mlp(self):
        """
        Test that the output shape of the MLP is as expected.
        """
        mlp = MLP(self.X_DIM, self.Y_DIM, self.WIDTH, self.LAYERS)
        assert mlp(self.X_data).shape == (self.N_SAMPLES, self.Y_DIM)

    def test_ngam(self):
        """
        Test that the output shape of the NGAM is as expected.
        """
        ngam = NGAM(self.X_DIM, self.Y_DIM, self.WIDTH, self.LAYERS)
        assert ngam(self.X_data).shape == (self.N_SAMPLES, self.Y_DIM)

    def test_softselect(self):
        """
        Test that the output shape of the SoftSelect is as expected.
        """
        softselect = SoftSelect(self.IN_DIMS, self.OUT_SHAPE)
        assert softselect(self.Z1, self.Z2).shape == (self.N_SAMPLES, *self.OUT_SHAPE)

        precycle_vals = softselect.archetypes
        assert precycle_vals.shape == (*self.OUT_SHAPE, *self.IN_DIMS)
        postcycle_vals = softselect.get_archetypes()
        assert postcycle_vals.shape == (*self.IN_DIMS, *self.OUT_SHAPE)
        softselect.set_archetypes(torch.randn(*self.IN_DIMS, *self.OUT_SHAPE))
        assert (softselect.archetypes != precycle_vals).any()
        softselect.set_archetypes(postcycle_vals)
        assert (softselect.archetypes == precycle_vals).all()

    def test_explainer(self):
        explainer = Explainer(self.IN_DIMS[0], self.OUT_SHAPE)
        ret = explainer(self.Z1)

        precycle_vals = explainer.archetypes
        assert precycle_vals.shape == (*self.OUT_SHAPE, self.IN_DIMS[0])
        postcycle_vals = explainer.get_archetypes()
        assert postcycle_vals.shape == (self.IN_DIMS[0], *self.OUT_SHAPE)
        explainer.set_archetypes(torch.randn(self.IN_DIMS[0], *self.OUT_SHAPE))
        assert (explainer.archetypes != precycle_vals).any()
        explainer.set_archetypes(postcycle_vals)
        assert (explainer.archetypes == precycle_vals).all()

    def test_linear(self):
        linear_encoder = Linear(self.X_DIM, self.Y_DIM)
        linear_output = linear_encoder(self.X_data)
        assert linear_output.shape == (self.N_SAMPLES, self.Y_DIM)


class TestSaveLoad(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_save_load(self):
        """
        Test saving and loading of contextualized objects
        """
        C = np.random.uniform(0, 1, size=(100, 2))
        X = np.random.uniform(0, 1, size=(100, 2))
        Y = np.random.uniform(0, 1, size=(100, 2))
        C2 = np.random.uniform(0, 1, size=(100, 2))
        X2 = np.random.uniform(0, 1, size=(100, 2))
        Y2 = np.random.uniform(0, 1, size=(100, 2))
        mlp = MLP(2, 2, 50, 5)
        Y_pred = mlp(torch.Tensor(X)).detach().numpy()
        save(mlp, "unittest_model.pt")
        del mlp
        mlp_loaded = load("unittest_model.pt")
        Y_pred_loaded = mlp_loaded(torch.Tensor(X)).detach().numpy()
        assert np.all(Y_pred == Y_pred_loaded)
        os.remove("unittest_model.pt")

        model = ContextualizedRegressor()
        model.fit(C, X, Y)
        Y_pred = model.predict(C, X)
        save(model, "unittest_model.pt")
        del model
        model_loaded = load("unittest_model.pt")
        Y_pred_loaded = model_loaded.predict(C, X)
        assert np.all(Y_pred == Y_pred_loaded)
        os.remove("unittest_model.pt")
        model_loaded.fit(C2, X2, Y2)
        Y_pred2 = model_loaded.predict(C2, X2)
        assert not np.all(Y_pred_loaded == Y_pred2)
        save(model_loaded, "unittest_model.pt")
        del model_loaded
        model_loaded2 = load("unittest_model.pt")
        Y_pred_loaded2 = model_loaded2.predict(C2, X2)
        assert np.all(Y_pred2 == Y_pred_loaded2)
        os.remove("unittest_model.pt")

        model = ContextualizedBayesianNetworks()
        model.fit(C, X)
        pred = model.predict_networks(C)
        save(model, "unittest_model.pt")
        del model
        model_loaded = load("unittest_model.pt")
        pred_loaded = model_loaded.predict_networks(C)
        assert np.all(np.array(pred) == np.array(pred_loaded))
        os.remove("unittest_model.pt")
        model_loaded.fit(C2, X2)
        pred2 = model_loaded.predict_networks(C2)
        assert not np.all(np.array(pred_loaded) == np.array(pred2))
        save(model_loaded, "unittest_model.pt")
        del model_loaded
        model_loaded2 = load("unittest_model.pt")
        pred_loaded2 = model_loaded2.predict_networks(C2)
        assert np.all(np.array(pred2) == np.array(pred_loaded2))
        os.remove("unittest_model.pt")

        model = ContextualizedCorrelationNetworks()
        model.fit(C, X)
        pred = model.predict_correlation(C)
        save(model, "unittest_model.pt")
        del model
        model_loaded = load("unittest_model.pt")
        pred_loaded = model_loaded.predict_correlation(C)
        assert np.all(np.array(pred) == np.array(pred_loaded))
        os.remove("unittest_model.pt")
        model_loaded.fit(C2, X2)
        pred2 = model_loaded.predict_correlation(C2)
        assert not np.all(np.array(pred_loaded) == np.array(pred2))
        save(model_loaded, "unittest_model.pt")
        del model_loaded
        model_loaded2 = load("unittest_model.pt")
        pred_loaded2 = model_loaded2.predict_correlation(C2)
        assert np.all(np.array(pred2) == np.array(pred_loaded2))
        os.remove("unittest_model.pt")

        model = BayesianNetwork()
        model.fit(X)
        pred = model.measure_mses(X)
        save(model, "unittest_model.pt")
        del model
        model_loaded = load("unittest_model.pt")
        pred_loaded = model_loaded.measure_mses(X)
        assert np.all(np.array(pred) == np.array(pred_loaded))
        os.remove("unittest_model.pt")
        model_loaded.fit(X2)
        pred2 = model_loaded.measure_mses(X2)
        assert not np.all(np.array(pred_loaded) == np.array(pred2))
        save(model_loaded, "unittest_model.pt")
        del model_loaded
        model_loaded2 = load("unittest_model.pt")
        pred_loaded2 = model_loaded2.measure_mses(X2)
        assert np.all(np.array(pred2) == np.array(pred_loaded2))
        os.remove("unittest_model.pt")

        model = CorrelationNetwork()
        model.fit(X)
        pred = model.measure_mses(X)
        save(model, "unittest_model.pt")
        del model
        model_loaded = load("unittest_model.pt")
        pred_loaded = model_loaded.measure_mses(X)
        assert np.all(np.array(pred) == np.array(pred_loaded))
        os.remove("unittest_model.pt")
        model_loaded.fit(X2)
        pred2 = model_loaded.measure_mses(X2)
        assert not np.all(np.array(pred_loaded) == np.array(pred2))
        save(model_loaded, "unittest_model.pt")
        del model_loaded
        model_loaded2 = load("unittest_model.pt")
        pred_loaded2 = model_loaded2.measure_mses(X2)
        assert np.all(np.array(pred2) == np.array(pred_loaded2))
        os.remove("unittest_model.pt")


if __name__ == "__main__":
    unittest.main()
