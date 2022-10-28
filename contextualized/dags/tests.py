"""
Unit tests for DAG models.
"""
import unittest
import numpy as np
import igraph as ig

from contextualized.dags.lightning_modules import NOTMAD
from contextualized.dags import graph_utils
from contextualized.dags.trainers import GraphTrainer


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
        (
            self.C_train,
            self.C_test,
            self.C_val,
            self.X_train,
            self.X_test,
            self.X_val,
            self.W_train,
            self.W_test,
            self.W_val,
        ) = self._create_cwx_dataset()

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

        W = np.transpose(W, (2, 0, 1))
        X = np.zeros((n, 4))
        for i, w in enumerate(W):
            x = simulate_linear_sem(w, 1, "uniform", noise_scale=0.1)[0]
            X[i] = x
        train_idx = np.logical_or(C < 1.7, C >= 1.9)[:, 0]
        test_idx = np.logical_and(C >= 1.8, C < 1.9)[:, 0]
        val_idx = np.logical_and(C >= 1.7, C < 1.8)[:, 0]
        return (
            C[train_idx],
            C[test_idx],
            C[val_idx],
            X[train_idx],
            X[test_idx],
            X[val_idx],
            W[train_idx],
            W[test_idx],
            W[val_idx],
        )

    def _quicktest(self, model, n_epochs=5):
        print(f"\n{type(model)} quicktest")

        trainer = GraphTrainer(max_epochs=n_epochs)
        train_dataloader = model.dataloader(self.C_train, self.X_train, batch_size=10)
        test_dataloader = model.dataloader(self.C_test, self.X_test, batch_size=10)
        val_dataloader = model.dataloader(self.C_val, self.X_val, batch_size=10)
        trainer.tune(model, train_dataloader)
        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.validate(model, val_dataloader)
        trainer.test(model, test_dataloader)
        train_preds = trainer.predict_params(
            model, train_dataloader, project_to_dag=True
        )
        test_preds = trainer.predict_params(model, test_dataloader, project_to_dag=True)
        val_preds = trainer.predict_params(model, val_dataloader, project_to_dag=True)

        mse = lambda true, pred: ((true - pred) ** 2).mean()
        print(f"train L2: {mse(train_preds, self.W_train)}")
        print(f"test L2:  {mse(test_preds, self.W_test)}")
        print(f"val L2:   {mse(val_preds, self.W_val)}")
        print(
            f"train mse: {mse(graph_utils.dag_pred_np(self.X_train, train_preds), self.X_train)}"
        )
        print(
            f"test mse:  {mse(graph_utils.dag_pred_np(self.X_test, test_preds), self.X_test)}"
        )
        print(
            f"val mse:   {mse(graph_utils.dag_pred_np(self.X_val, val_preds), self.X_val)}"
        )

    def test_notmad(self):
        # 1 archetype
        k = 1
        INIT_MAT = np.random.uniform(-0.01, 0.01, size=(k, 4, 4))
        model = NOTMAD(
            self.C_train.shape[-1],
            self.X_train.shape[-1],
            init_mat=INIT_MAT,
            num_archetypes=k,
        )
        self._quicktest(model)

        # 10 archetypes
        k = 20
        INIT_MAT = np.random.uniform(-0.01, 0.01, size=(k, 4, 4))
        model = NOTMAD(
            self.C_train.shape[-1],
            self.X_train.shape[-1],
            init_mat=INIT_MAT,
            num_archetypes=k,
        )
        self._quicktest(model)


if __name__ == "__main__":
    unittest.main()
