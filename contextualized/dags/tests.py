"""
Unit tests for DAG models.
"""
import unittest
import numpy as np
import igraph as ig

from contextualized.dags.torch_notmad import NOTMAD_model
from contextualized.dags.datamodules import CXW_DataModule
from contextualized.dags.callbacks import DynamicAlphaRho
from contextualized.dags import graph_utils
from pytorch_lightning import Trainer


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=scale, size=n)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=scale, size=n)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "uniform":
            z = np.random.uniform(low=-scale, high=scale, size=n)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "logistic":
            # x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            x = np.random.binomial(1, sigmoid(np.matmul(X, w))) * 1.0
        elif sem_type == "poisson":
            # x = np.random.poisson(np.exp(X @ w)) * 1.0
            x = np.random.poisson(np.exp(np.matmul(X, w))) * 1.0
        else:
            raise ValueError("unknown sem type")
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or has length d")
        scale_vec = noise_scale
    if not graph_utils.is_dag(W):
        raise ValueError("W must be a DAG")
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/d X'X = true cov
            # X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            X = np.sqrt(d) * np.matmul(np.diag(scale_vec), np.linalg.inv(np.eye(d) - W))
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


class TestNOTMAD(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNOTMAD, self).__init__(*args, **kwargs)

    def setUp(self):
        self.C, self.W, self.X = self._create_cwx_dataset(500)

    def _create_cwx_dataset(self, n=1000):
        C = np.linspace(1, 2, n).reshape((n, 1))
        blank = np.zeros_like(C)
        W_00 = blank
        W_01 = C - 2
        W_02 = blank
        W_03 = blank
        W_10 = blank
        W_11 = blank
        W_12 = blank
        W_13 = blank
        W_20 = blank
        W_21 = C**2
        W_22 = blank
        W_23 = blank
        W_30 = blank
        W_31 = C**3
        W_32 = C
        W_33 = blank

        W = np.array(
            [
                [W_00, W_01, W_02, W_03],
                [W_10, W_11, W_12, W_13],
                [W_20, W_21, W_22, W_23],
                [W_30, W_31, W_32, W_33],
            ]
        ).squeeze()

        def _dag_pred(self, x, w):
            return np.matmul(x, w).squeeze()

        W = np.transpose(W, (2, 0, 1))
        X = np.zeros((n, 4))
        X_pre = np.random.uniform(-1, 1, (n, 4))
        for i, w in enumerate(W):
            eps = np.random.normal(0, 0.01, 4)
            eps = 0
            X_new = simulate_linear_sem(w, 1, "uniform", noise_scale=0.1)[0]
            # X_new = _dag_pred(X_p[np.newaxis, :], w)
            X[i] = X_new + eps

        return C, W, X

    def _quicktest(self, model, datamodule, n_epochs=5):
        print(f"\n{type(model)} quicktest")

        trainer = Trainer(max_epochs=n_epochs, callbacks=[DynamicAlphaRho()])

        trainer.tune(model)
        trainer.fit(model, datamodule)
        trainer.validate(model, datamodule)
        trainer.test(model, datamodule)
        # data
        C_train = trainer.model.datamodule.C_train
        C_test = trainer.model.datamodule.C_test
        W_train = trainer.model.datamodule.W_train
        W_test = trainer.model.datamodule.W_test
        X_train = trainer.model.datamodule.X_train
        X_test = trainer.model.datamodule.X_test

        # Evaluate results
        torch_notmad_preds_train = trainer.model.predict_w(
            C_train, confirm_project_to_dag=True
        )
        torch_notmad_preds = trainer.model.predict_w(C_test).squeeze().detach().numpy()

        torch_notmad_preds_train = trainer.model.predict_w(
            C_train, confirm_project_to_dag=True
        )
        torch_notmad_preds = trainer.model.predict_w(C_test).squeeze().detach().numpy()

        mse = lambda true, pred: ((true - pred) ** 2).mean()
        dag_pred = lambda x, w: np.matmul(x, w).squeeze()
        dags_pred = lambda xs, w: [dag_pred(x, w) for x in xs]

        example_preds = dags_pred(X_train, torch_notmad_preds_train)
        actual_preds = dags_pred(X_train, W_train)

        print(f"train L2: {mse(torch_notmad_preds_train, W_train)}")
        print(f"test L2:  {mse(torch_notmad_preds, W_test)}")
        print(f"train mse: {mse(dag_pred(X_train, torch_notmad_preds_train), X_train)}")
        print(f"test mse:  {mse(dag_pred(X_test, torch_notmad_preds), X_test)}")

    def test_notmad(self):
        # 5 archetypes
        k = 5
        INIT_MAT = np.random.uniform(-0.01, 0.01, size=(k, 4, 4))
        datamodule = CXW_DataModule(self.C, self.X, self.W)
        model = NOTMAD_model(
            datamodule,
            init_mat=INIT_MAT,
            n_archetypes=k,
        )
        self._quicktest(model, datamodule, n_epochs=5)

        # 6 archetypes
        k = 6
        INIT_MAT = np.random.uniform(-0.01, 0.01, size=(k, 4, 4))
        model = NOTMAD_model(
            datamodule,
            init_mat=INIT_MAT,
            n_archetypes=k,
        )
        self._quicktest(model, datamodule, n_epochs=5)


if __name__ == "__main__":
    unittest.main()
