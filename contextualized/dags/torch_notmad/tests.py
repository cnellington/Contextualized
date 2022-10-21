import time
import torch
import unittest
import numpy as np

from contextualized.modules import ENCODERS, SoftSelect, Explainer
from contextualized.functions import LINK_FUNCTIONS
from contextualized.dags.notmad_helpers.simulation import simulate_linear_sem
from contextualized.regression.lightning_modules import *
from contextualized.regression.trainers import *

from contextualized.dags.torch_notmad.torch_notmad import NOTMAD_model
from contextualized.dags.torch_notmad.datamodules import (
    CX_Dataset,
    CX_DataModule,
    CXW_Dataset,
    CXW_DataModule,
)
from contextualized.dags.torch_notmad.callbacks import DynamicAlphaRho
from pytorch_lightning import Trainer


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
