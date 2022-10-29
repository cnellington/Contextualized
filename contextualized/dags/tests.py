"""
Unit tests for DAG models.
"""
import unittest
import numpy as np
import igraph as ig
from pytorch_lightning.utilities.seed import seed_everything


from contextualized.dags.lightning_modules import NOTMAD
from contextualized.dags import graph_utils
from contextualized.dags.trainers import GraphTrainer
from contextualized.dags.losses import mse_loss as mse


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
            self.C,
            self.X,
            self.W,
            self.train_idx,
            self.test_idx,
            self.val_idx,
        ) = self._create_cwx_dataset()
        self.C_train, self.C_test, self.C_val = (
            self.C[self.train_idx],
            self.C[self.test_idx],
            self.C[self.val_idx],
        )
        self.X_train, self.X_test, self.X_val = (
            self.X[self.train_idx],
            self.X[self.test_idx],
            self.X[self.val_idx],
        )
        self.W_train, self.W_test, self.W_val = (
            self.W[self.train_idx],
            self.W[self.test_idx],
            self.W[self.val_idx],
        )

    def _create_cwx_dataset(self, n=500):
        np.random.seed(0)
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
        train_idx = np.argwhere(np.logical_or(C < 1.7, C >= 1.9)[:, 0])[:, 0]
        np.random.shuffle(train_idx)
        test_idx = np.argwhere(np.logical_and(C >= 1.8, C < 1.9)[:, 0])[:, 0]
        val_idx = np.argwhere(np.logical_and(C >= 1.7, C < 1.8)[:, 0])[:, 0]
        return (
            C,
            X,
            W,
            train_idx,
            test_idx,
            val_idx,
        )

    def _evaluate(self, train_preds, test_preds, val_preds):
        return (
            mse(train_preds, self.W_train),
            mse(test_preds, self.W_test),
            mse(val_preds, self.W_val),
            mse(graph_utils.dag_pred_np(self.X_train, train_preds), self.X_train),
            mse(graph_utils.dag_pred_np(self.X_test, test_preds), self.X_test),
            mse(graph_utils.dag_pred_np(self.X_val, val_preds), self.X_val),
        )

    def _train(self, n_epochs):
        seed_everything(0)
        k = 6
        INIT_MAT = np.random.uniform(-0.01, 0.01, size=(k, 4, 4))
        model = NOTMAD(
            self.C.shape[-1],
            self.X.shape[-1],
            init_mat=INIT_MAT,
            num_archetypes=k,
        )
        train_dataloader = model.dataloader(self.C_train, self.X_train, batch_size=1)
        test_dataloader = model.dataloader(self.C_test, self.X_test, batch_size=10)
        val_dataloader = model.dataloader(self.C_val, self.X_val, batch_size=10)
        trainer = GraphTrainer(max_epochs=n_epochs, callbacks=[], deterministic=True)
        trainer.tune(model)
        trainer.fit(model, train_dataloader)
        trainer.validate(model, val_dataloader)
        trainer.test(model, test_dataloader)

        # Evaluate results
        preds_train = trainer.predict_params(
            model, train_dataloader, project_to_dag=True
        )
        preds_test = trainer.predict_params(model, test_dataloader, project_to_dag=True)
        preds_val = trainer.predict_params(model, val_dataloader, project_to_dag=True)
        return preds_train, preds_test, preds_val

    def test_notmad(self):
        train_preds, test_preds, val_preds = self._train(10)
        print(train_preds[0])
        print(self.W_train[0])
        print(train_preds[-1])
        print(self.W_train[-1])
        train_l2, test_l2, val_l2, train_mse, test_mse, val_mse = self._evaluate(
            train_preds, test_preds, val_preds
        )
        print(f"Train L2: {train_l2}")
        print(f"Test L2:  {test_l2}")
        print(f"Val L2:   {val_l2}")
        print(f"Train mse: {train_mse}")
        print(f"Test mse:  {test_mse}")
        print(f"Val mse:   {val_mse}")
        assert train_l2 < 1e-1
        assert test_l2 < 1e-1
        assert val_l2 < 1e-1
        assert train_mse < 1e-2
        assert test_mse < 1e-2
        assert val_mse < 1e-2


if __name__ == "__main__":
    unittest.main()
