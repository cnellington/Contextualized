"""
sklearn-like interface to Contextualized Networks.
"""
import numpy as np
from pytorch_lightning import Trainer

from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression.trainers import CorrelationTrainer, MarkovTrainer
from contextualized.regression.lightning_modules import (
    ContextualizedCorrelation,
    TasksplitContextualizedCorrelation,
    ContextualizedMarkovGraph,
)

# from contextualized.dags.torch_notmad.torch_notmad import NOTMAD_model


class ContextualizedNetworks(SKLearnWrapper):
    """
    sklearn-like interface to Contextualized Networks.
    """

    def _split_train_data(self, C, X, **kwargs):
        return super()._split_train_data(C, X, Y_required=False, **kwargs)

    def predict_networks(self, C, with_offsets=False, **kwargs):
        """
        Predicts context-specific networks.
        """
        betas, mus = self.predict_params(C, uses_y=False, **kwargs)
        if with_offsets:
            return betas, mus
        return betas

    def predict_X(self, C, X, **kwargs):
        """
        Predicts X based on context-specific networks.
        """
        # TODO: This may not work correctly for the univariable models.
        return self.predict(C, X, **kwargs)


# TODO: TasksplitContextualizedCorrelation?
class ContextualizedCorrelationNetworks(ContextualizedNetworks):
    """ "
    Easy interface to Contextualized Correlation Networks.
    """

    def __init__(self, **kwargs):
        super().__init__(
            ContextualizedCorrelation, [], [], CorrelationTrainer, **kwargs
        )

    def predict_correlation(self, C, individual_preds=True, squared=True, **kwargs):
        """
        Predict correlation matrices.
        """
        get_dataloader = lambda i: self.models[i].dataloader(
            C, np.zeros((len(C), self.x_dim))
        )
        rhos = np.array(
            [
                self.trainers[i].predict_params(
                    self.models[i], get_dataloader(i), **kwargs
                )[0]
                for i in range(len(self.models))
            ]
        )
        if individual_preds:
            if squared:
                return np.square(rhos)
            return rhos
        else:
            if squared:
                return np.square(np.mean(rhos, axis=0))
            return np.mean(rhos)

    def measure_mses(self, C, X):
        """
        Measure mean-squared errors.
        """
        betas, mus = self.predict_networks(C, with_offsets=True)
        mses = np.zeros(len(C))
        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                residuals = X[:, i] - (betas[:, i, j] * X[:, j] + mus[:, i, j])
                mses += residuals**2 / (X.shape[-1] ** 2)
        return mses


class ContextualizedMarkovNetworks(ContextualizedNetworks):
    """ "
    Easy interface to Contextualized Markov Networks.
    """

    def __init__(self, **kwargs):
        super().__init__(ContextualizedMarkovGraph, [], [], MarkovTrainer, **kwargs)

    def predict_precisions(self, C, individual_preds=True):
        """
        Predict precision matrices.
        """
        get_dataloader = lambda i: self.models[i].dataloader(
            C, np.zeros((len(C), self.x_dim))
        )
        precisions = np.array(
            [
                self.trainers[i].predict_precision(self.models[i], get_dataloader(i))
                for i in range(len(self.models))
            ]
        )
        if individual_preds:
            return precisions
        return np.mean(precisions, axis=0)

    def measure_mses(self, C, X):
        """
        Measure mean-squared errors.
        """
        betas, mus = self.predict_networks(C, with_offsets=True)
        mses = np.zeros(len(C))
        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                residuals = X[:, i] - (betas[:, i, j] * X[:, j] + mus[:, i])
                mses += residuals**2 / (X.shape[-1] ** 2)
        return mses


"""
TODO: The DataModule interface for NOTMAD is strange and seems wrong.
class ContextualizedBayesianNetworks(ContextualizedNetworks):
    def __init__(self, C, X, **kwargs):
        def notmad_constructor(**kwargs):
            return NOTMAD_model(CX_DataModule(C, X), **kwargs)
        super().__init__(notmad_constructor,
            ["sample_specific_loss_params", "archetype_loss_params"],
            [],
            Trainer)

    def predict_networks(self, C, with_offsets=False, **kwargs):
"""
