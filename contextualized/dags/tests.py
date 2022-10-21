"""
Unit tests for DAG models.
"""
import unittest
import numpy as np
import tensorflow as tf
from contextualized.dags import NOTMAD


class TestDAGs(unittest.TestCase):
    """
    Test DAGs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Shared set-up function.
        """
        n_samples = 1000
        C = np.linspace(1, 2, n_samples).reshape((n_samples, 1))
        blank = np.zeros_like(C)
        W = np.zeros((4, 4, n_samples))
        W[0, 1] = C - 2
        W[2, 1] = C**2
        W[3, 1] = C**3
        W[3, 2] = C
        W = np.transpose(W, (2, 0, 1))

        X_pre = np.random.uniform(-1, 1, (n_samples, 4))
        X = np.zeros_like(X_pre)
        for i, (w, X_p) in enumerate(zip(W, X_pre)):
            eps = np.random.normal(0, 0.01, 4)
            X[i] = self._dag_pred(X_p[np.newaxis, :], w) + eps

        idx = np.logical_and(C > 1.7, C < 1.9).squeeze()
        test_idx = np.argwhere(idx).squeeze()
        train_idx = np.argwhere(~idx).squeeze()
        split = lambda arr: (arr[train_idx], arr[test_idx])

        self.epochs = 1
        self.batch_size = 32
        self.W_train, self.W_test = split(W)
        self.C_train, self.C_test = split(C)
        self.X_train, self.X_test = split(X)

    def _dag_pred(self, x, w):
        """

        :param x:
        :param w:

        """
        return tf.matmul(x, w).numpy().squeeze()

    def test_notmad(self):
        """ """
        mse = lambda true, pred: ((true - pred) ** 2).mean()
        k = 5
        sample_specific_loss_params = {"l1": 0.0, "init_alpha": 1e-1, "init_rho": 1e-2}
        # loss_params = {'l1': 1e-2, 'alpha': 1e-1, 'rho': 1e-2}
        archetype_loss_params = {"l1": 0.0, "alpha": 1e-1, "rho": 1e-2}
        init_mat = np.random.uniform(
            -0.01, 0.01, size=(k, self.X_train.shape[-1], self.X_train.shape[-1])
        )  # np.zeros((k, X_train.shape[-1], X_train.shape[-1])) #
        notmad = NOTMAD(
            self.C_train.shape,
            self.X_train.shape,
            k,
            sample_specific_loss_params,
            archetype_loss_params,
            n_encoder_layers=2,
            encoder_width=32,
            activation="linear",
            init_mat=init_mat,
            learning_rate=1e-3,
            project_archs_to_dag=True,  # TODO: should this be variable?
            project_distance=1.0,
            context_activity_regularizer=tf.keras.regularizers.l1(0),
            use_compatibility=False,
            update_compat_by_grad=False,
            pop_model=None,
            base_predictor=None,
        )

        notmad_preds_train = notmad.predict_w(
            self.C_train, project_to_dag=True
        ).squeeze()
        notmad_preds = notmad.predict_w(self.C_test, project_to_dag=True).squeeze()
        param_mse_init = mse(notmad_preds_train, self.W_train)
        dag_mse_init = mse(
            self._dag_pred(self.X_train, notmad_preds_train), self.X_train
        )

        notmad.fit(
            self.C_train,
            self.X_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            es_patience=2,
            verbose=1,
        )
        notmad_preds_train = notmad.predict_w(
            self.C_train, project_to_dag=True
        ).squeeze()
        notmad_preds = notmad.predict_w(self.C_test, project_to_dag=True).squeeze()
        param_mse_trained = mse(notmad_preds_train, self.W_train)
        dag_mse_trained = mse(
            self._dag_pred(self.X_train, notmad_preds_train), self.X_train
        )

        #         assert param_mse_init < param_mse_trained  # not guaranteed
        assert dag_mse_trained < dag_mse_init, "Model failed to converge"


if __name__ == "__main__":
    unittest.main()
