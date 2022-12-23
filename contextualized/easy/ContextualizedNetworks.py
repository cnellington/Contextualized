"""
sklearn-like interface to Contextualized Networks.
"""
import numpy as np

from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression.trainers import CorrelationTrainer, MarkovTrainer
from contextualized.regression.lightning_modules import (
    ContextualizedCorrelation,
    # TasksplitContextualizedCorrelation, # TODO: Incorporate Tasksplit
    ContextualizedMarkovGraph,
)
from contextualized.dags.lightning_modules import NOTMAD, DEFAULT_DAG_LOSS_PARAMS
from contextualized.dags.trainers import GraphTrainer


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
        return self.predict(C, X, **kwargs)


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

    def measure_mses(self, C, X, individual_preds=False):
        """
        Measure mean-squared errors.
        """
        betas, mus = self.predict_networks(C, individual_preds=True, with_offsets=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                residuals = np.tile(X[:, i], len(betas)) - (
                    betas[:, :, i, j] * np.tile(X[:, j], len(betas)) + mus[:, :, i, j]
                )
                mses += residuals**2 / (X.shape[-1] ** 2)
        if not individual_preds:
            mses = np.mean(mses, axis=0)
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

    def measure_mses(self, C, X, individual_preds=False):
        """
        Measure mean-squared errors.
        """
        betas, mus = self.predict_networks(C, individual_preds=True, with_offsets=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for bootstrap in range(len(betas)):
            for i in range(X.shape[-1]):
                # betas are n_boostraps x n_samples x n_features x n_features
                # preds[bootstrap, sample, i] = X[sample, :].dot(betas[bootstrap, sample, i, :])
                preds = np.array(
                    [
                        X[j].dot(betas[bootstrap, j, i, :]) + mus[bootstrap, j, i]
                        for j in range(len(X))
                    ]
                )
                residuals = X[:, i] - preds
                mses[bootstrap, :] += residuals**2 / (X.shape[-1])
        if not individual_preds:
            mses = np.mean(mses, axis=0)
        return mses


class ContextualizedBayesianNetworks(ContextualizedNetworks):
    """
    Easy interface to Contextualized Bayesian Networks.
    Uses NOTMAD model.
    See this paper:
    https://arxiv.org/abs/2111.01104
    for more details.
    """

    def __init__(self, **kwargs):
        super().__init__(
            NOTMAD,
            extra_model_kwargs=[
                "sample_specific_params",
                "archetype_params",
                "opt_params",
            ],
            extra_data_kwargs=[],
            trainer_constructor=GraphTrainer,
            remove_model_kwargs=[
                "link_fn",
                "univariate",
                "loss_fn",
                "model_regularizer",
            ],
            **kwargs,
        )
        self._init_kwargs["model"]["encoder_kwargs"] = {
            "type": kwargs.get(
                "encoder_type", self._init_kwargs["model"]["encoder_type"]
            ),
            "params": {
                "width": self.constructor_kwargs["encoder_kwargs"]["width"],
                "layers": self.constructor_kwargs["encoder_kwargs"]["layers"],
                "link_fn": self.constructor_kwargs["encoder_kwargs"]["link_fn"],
            },
        }
        archetype_dag_loss_type = kwargs.get("archetype_dag_loss_type", "NOTEARS")
        self._init_kwargs["model"]["archetype_params"] = {
            "l1": kwargs.get("archetype_l1", 0.0),
            "dag": kwargs.get(
                "archetype_dag_params",
                {
                    "loss_type": archetype_dag_loss_type,
                    "params": kwargs.get(
                        "archetype_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[archetype_dag_loss_type],
                    ),
                },
            ),
            "init_mat": kwargs.get("init_mat", None),
            "num_factors": kwargs.get("num_factors", 0),
            "factor_mat_l1": kwargs.get("factor_mat_l1", 0),
            "num_archetypes": kwargs.get("num_archetypes", 0),
        }
        # Possibly update values with convenience parameters
        for k, v in self._init_kwargs["model"]["archetype_params"]["dag"][
            "params"
        ].items():
            self._init_kwargs["model"]["archetype_params"]["dag"]["params"][
                k
            ] = kwargs.get(f"archetype_{k}", v)

        sample_specific_dag_loss_type = kwargs.get(
            "sample_specific_dag_loss_type", "NOTEARS"
        )
        self._init_kwargs["model"]["sample_specific_params"] = {
            "l1": kwargs.get("sample_specific_l1", 0.0),
            "dag": kwargs.get(
                "sample_specific_params",
                {
                    "loss_type": sample_specific_dag_loss_type,
                    "params": kwargs.get(
                        "sample_specific_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[sample_specific_dag_loss_type],
                    ),
                },
            ),
        }
        # Possibly update values with convenience parameters
        for k, v in self._init_kwargs["model"]["sample_specific_params"]["dag"][
            "params"
        ].items():
            self._init_kwargs["model"]["sample_specific_params"]["dag"]["params"][
                k
            ] = kwargs.get(f"sample_specific_{k}", v)

        self._init_kwargs["model"]["opt_params"] = {
            "learning_rate": kwargs.get("learning_rate", 1e-3),
            "step": kwargs.get("step", 50),
        }

    def predict_params(self, C, individual_preds=False, **kwargs):
        """

        :param C:
        :param individual_preds:  (Default value = False)

        """
        # Returns betas
        # TODO: No mus for NOTMAD at present.
        return super().predict_params(
            C, individual_preds, model_includes_mus=False, **kwargs
        )

    def predict_networks(self, C, with_offsets=False, project_to_dag=True, **kwargs):
        """
        Predicts context-specific networks.
        """
        betas = self.predict_params(
            C, uses_y=False, project_to_dag=project_to_dag, **kwargs
        )
        if with_offsets:
            print("No offsets returned by NOTMAD.")
        return betas

    def measure_mses(self, C, X, individual_preds=False):
        """
        Measure mean-squared errors.
        """
        betas = self.predict_networks(C, individual_preds=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for bootstrap in range(len(betas)):
            for i in range(X.shape[-1]):
                # betas are n_boostraps x n_samples x n_features x n_features
                # preds[bootstrap, sample, i] = X[sample, :].dot(betas[bootstrap, sample, i, :])
                preds = np.array(
                    [
                        X[j].dot(betas[bootstrap, j, i, :])  # + mus[bootstrap, j, i]
                        for j in range(len(X))
                    ]
                )
                residuals = X[:, i] - preds
                mses[bootstrap, :] += residuals**2 / (X.shape[-1])
        if not individual_preds:
            mses = np.mean(mses, axis=0)
        return mses
