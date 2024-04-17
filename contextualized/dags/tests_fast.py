"""
Unit tests for DAG models.
"""
import unittest
import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateFinder


from contextualized.dags.lightning_modules import NOTMAD
from contextualized.dags import graph_utils
from contextualized.dags.trainers import GraphTrainer
from contextualized.dags.losses import mse_loss as mse


class TestNOTMAD(unittest.TestCase):
    """Unit tests for NOTMAD."""

    def __init__(self, *args, **kwargs):
        super(TestNOTMAD, self).__init__(*args, **kwargs)

    def setUp(self):
        seed_everything(0)
        self.n = 10
        self.c_dim = 4
        self.x_dim = 3
        self.C = np.random.uniform(-1, 1, size=(self.n, self.c_dim))
        self.X = np.random.uniform(-1, 1, size=(self.n, self.x_dim))

    def _train(self, model_args, n_epochs):
        k = 6
        INIT_MAT = np.random.uniform(-0.1, 0.1, size=(k, 4, 4)) * np.tile(
            1 - np.eye(4), (k, 1, 1)
        )
        model = NOTMAD(
            self.C.shape[-1],
            self.X.shape[-1],
            archetype_params={
                "l1": 0.0,
                "dag": model_args.get(
                    "dag",
                    {
                        "loss_type": "NOTEARS",
                        "params": {
                            "alpha": 1e-1,
                            "rho": 1e-2,
                            "h_old": 0.0,
                            "tol": 0.25,
                            "use_dynamic_alpha_rho": True,
                        },
                    },
                ),
                "init_mat": INIT_MAT,
                "num_factors": model_args.get("num_factors", 0),
                "factor_mat_l1": 0.0,
                "num_archetypes": model_args.get("num_archetypes", k),
            },
        )
        dataloader = model.dataloader(
            self.C, self.X, batch_size=1, num_workers=0
        )
        trainer = GraphTrainer(
            max_epochs=n_epochs, deterministic=True, enable_progress_bar=False
        )
        predict_trainer = GraphTrainer(
            deterministic=True, enable_progress_bar=False, devices=1
        )
        init_preds = predict_trainer.predict_params(
            model, dataloader, project_to_dag=False,
        )
        assert init_preds.shape == (self.n, self.x_dim, self.x_dim)
        init_mse = mse(graph_utils.dag_pred_np(self.X, init_preds), self.X)
        trainer.fit(model, dataloader)
        final_preds = predict_trainer.predict_params(
            model, dataloader, project_to_dag=False
        )
        assert final_preds.shape == (self.n, self.x_dim, self.x_dim)
        final_mse = mse(graph_utils.dag_pred_np(self.X, final_preds), self.X)
        assert final_mse < init_mse

    def test_notmad_dagma(self):
        self._train(
            {
                "dag": {
                    "loss_type": "DAGMA",
                    "params": {
                        "alpha": 1.0,
                    },
                }
            },
            1,
        )

    def test_notmad_notears(self):
        self._train({}, 1)

    def test_notmad_factor_graphs(self):
        """
        Unit tests for factor graph feature of NOTMAD.
        """
        self._train({"num_factors": 3}, 1)


if __name__ == "__main__":
    unittest.main()
