import numpy as np
import copy
import tensorflow as tf
from test_notmad.tf_loss import _DAG_loss

from tensorflow.keras.callbacks import Callback


class _DynamicAlphaRho(Callback):
    """
    Dual-step DAG optimization parameter update, required for NO-TEARS structure learning
    """

    def __init__(self, C_train, base_predictor=None, tol=0.25):
        super(_DynamicAlphaRho, self).__init__()
        self.C_train = C_train
        self.h_old = 0.0
        self.tol = tol
        self.base_predictor = None

    def on_epoch_begin(self, epoch, logs=None):

        pred = np.squeeze(self.model.predict(self.C_train))

        my_dag_loss = tf.reduce_mean(
            _DAG_loss(pred, self.model.alpha.numpy(), self.model.rho.numpy())
        )

        if my_dag_loss > self.tol * self.h_old:
            self.model.alpha.assign(self.model.alpha + self.model.rho * my_dag_loss)
            self.model.rho.assign(self.model.rho * 1.1)
        self.h_old = my_dag_loss
