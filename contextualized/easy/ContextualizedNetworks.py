"""
sklearn-like interface to Contextualized Networks.
"""

from typing import *

import numpy as np

from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression.trainers import CorrelationTrainer, MarkovTrainer
from contextualized.regression.lightning_modules import (
    ContextualizedCorrelation,
    ContextualizedMarkovGraph,
)
from contextualized.dags.lightning_modules import (
    NOTMAD,
    DEFAULT_DAG_LOSS_TYPE,
    DEFAULT_DAG_LOSS_PARAMS,
)
from contextualized.dags.trainers import GraphTrainer
from contextualized.dags.graph_utils import dag_pred_np


class ContextualizedNetworks(SKLearnWrapper):
    """
    sklearn-like interface to Contextualized Networks.
    """

    def _split_train_data(
        self, C: np.ndarray, X: np.ndarray, **kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Splits data into train and test sets.

        Args:
            C (np.ndarray): Contextual features for each sample.
            X (np.ndarray): The data matrix.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: The train and test sets for C and X as ([C_train, X_train], [C_test, X_test]).
        """
        return super()._split_train_data(C, X, Y_required=False, **kwargs)

    def predict_networks(
        self,
        C: np.ndarray,
        with_offsets: bool = False,
        individual_preds: bool = False,
        **kwargs,
    ) -> Union[
        np.ndarray,
        List[np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[List[np.ndarray], List[np.ndarray]],
    ]:
        """Predicts context-specific networks given contextual features.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            with_offsets (bool, optional): If True, returns both the network parameters and offsets. Defaults to False.
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]: The predicted network parameters (and offsets if with_offsets is True). Returned as lists of individual bootstraps if individual_preds is True.
        """
        betas, mus = self.predict_params(
            C, individual_preds=individual_preds, uses_y=False, **kwargs
        )
        if with_offsets:
            return betas, mus
        return betas

    def predict_X(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reconstructs the data matrix based on predicted contextualized networks and the true data matrix.
        Useful for measuring reconstruction error or for imputation.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            X (np.ndarray): The data matrix (n_samples, n_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to False.
            **kwargs: Keyword arguments for the Lightning trainer's predict_y method.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The predicted data matrix, or matrices for each bootstrap if individual_preds is True (n_samples, n_features).
        """
        return self.predict(C, X, individual_preds=individual_preds, **kwargs)


class ContextualizedCorrelationNetworks(ContextualizedNetworks):
    """
    Contextualized Correlation Networks reveal context-varying feature correlations, interaction strengths, dependencies in feature groups.
    Uses the Contextualized Networks model, see the `paper <https://doi.org/10.1101/2023.12.01.569658>`__ for detailed estimation procedures.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 10. Always uses archetypes in the ContextualizedMetaModel.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        super().__init__(
            ContextualizedCorrelation, [], [], CorrelationTrainer, **kwargs
        )

    def predict_correlation(
        self, C: np.ndarray, individual_preds: bool = True, squared: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predicts context-specific correlations between features.

        Args:
            C (Numpy ndarray): Contextual features for each sample (n_samples, n_context_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to True.
            squared (bool, optional): If True, returns the squared correlations. Defaults to True.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The predicted context-specific correlation matrices, or matrices for each bootstrap if individual_preds is True (n_samples, n_features, n_features).
        """
        get_dataloader = lambda i: self.models[i].dataloader(
            C, np.zeros((len(C), self.x_dim))
        )
        rhos = np.array(
            [
                self.trainers[i].predict_params(self.models[i], get_dataloader(i))[0]
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
            return np.mean(rhos, axis=0)

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Measures mean-squared errors.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            X (np.ndarray): The data matrix (n_samples, n_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The mean-squared errors for each sample, or for each bootstrap if individual_preds is True (n_samples).
        """
        betas, mus = self.predict_networks(C, individual_preds=True, with_offsets=True)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                tiled_xi = np.array([X[:, i] for _ in range(len(betas))])
                tiled_xj = np.array([X[:, j] for _ in range(len(betas))])
                residuals = tiled_xi - betas[:, :, i, j] * tiled_xj - mus[:, :, i, j]
                mses += residuals**2 / (X.shape[-1] ** 2)
        if not individual_preds:
            mses = np.mean(mses, axis=0)
        return mses


class ContextualizedMarkovNetworks(ContextualizedNetworks):
    """
    Contextualized Markov Networks reveal context-varying feature dependencies, cliques, and modules.
    Implemented as Contextualized Gaussian Precision Matrices, directly interpretable as Markov Networks.
    Uses the Contextualized Networks model, see the `paper <https://doi.org/10.1101/2023.12.01.569658>`__ for detailed estimation procedures.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 10. Always uses archetypes in the ContextualizedMetaModel.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        super().__init__(ContextualizedMarkovGraph, [], [], MarkovTrainer, **kwargs)

    def predict_precisions(
        self, C: np.ndarray, individual_preds: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predicts context-specific precision matrices.
        Can be converted to context-specific Markov networks by binarizing the networks and setting all non-zero entries to 1.
        Can be converted to context-specific covariance matrices by taking the inverse.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to True.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The predicted context-specific Markov networks as precision matrices, or matrices for each bootstrap if individual_preds is True (n_samples, n_features, n_features).
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

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Measures mean-squared errors.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            X (np.ndarray): The data matrix (n_samples, n_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The mean-squared errors for each sample, or for each bootstrap if individual_preds is True (n_samples).
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
    Contextualized Bayesian Networks and Directed Acyclic Graphs (DAGs) reveal context-dependent causal relationships, effect sizes, and variable ordering.
    Uses the NOTMAD model, see the `paper <https://doi.org/10.48550/arXiv.2111.01104>`__ for detailed estimation procedures.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 16. Always uses archetypes in the ContextualizedMetaModel.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        archetype_dag_loss_type (str, optional): The type of loss to use for the archetype loss. Defaults to "l1".
        archetype_l1 (float, optional): The strength of the l1 regularization for the archetype loss. Defaults to 0.0.
        archetype_dag_params (dict, optional): Parameters for the archetype loss. Defaults to {"loss_type": "l1", "params": {"alpha": 0.0, "rho": 0.0, "s": 0.0, "tol": 1e-4}}.
        archetype_dag_loss_params (dict, optional): Parameters for the archetype loss. Defaults to {"alpha": 0.0, "rho": 0.0, "s": 0.0, "tol": 1e-4}.
        archetype_alpha (float, optional): The strength of the alpha regularization for the archetype loss. Defaults to 0.0.
        archetype_rho (float, optional): The strength of the rho regularization for the archetype loss. Defaults to 0.0.
        archetype_s (float, optional): The strength of the s regularization for the archetype loss. Defaults to 0.0.
        archetype_tol (float, optional): The tolerance for the archetype loss. Defaults to 1e-4.
        archetype_use_dynamic_alpha_rho (bool, optional): Whether to use dynamic alpha and rho for the archetype loss. Defaults to False.
        init_mat (np.ndarray, optional): The initial adjacency matrix for the archetype loss. Defaults to None.
        num_factors (int, optional): The number of factors for the archetype loss. Defaults to 0.
        factor_mat_l1 (float, optional): The strength of the l1 regularization for the factor matrix for the archetype loss. Defaults to 0.
        sample_specific_dag_loss_type (str, optional): The type of loss to use for the sample-specific loss. Defaults to "l1".
        sample_specific_alpha (float, optional): The strength of the alpha regularization for the sample-specific loss. Defaults to 0.0.
        sample_specific_rho (float, optional): The strength of the rho regularization for the sample-specific loss. Defaults to 0.0.
        sample_specific_s (float, optional): The strength of the s regularization for the sample-specific loss. Defaults to 0.0.
        sample_specific_tol (float, optional): The tolerance for the sample-specific loss. Defaults to 1e-4.
        sample_specific_use_dynamic_alpha_rho (bool, optional): Whether to use dynamic alpha and rho for the sample-specific loss. Defaults to False.
    """

    def _parse_private_init_kwargs(self, **kwargs):
        """
        Parses the kwargs for the NOTMAD model.

        Args:
            **kwargs: Keyword arguments for the NOTMAD model, including the encoder, archetype loss, sample-specific loss, and optimization parameters.
        """
        # Encoder Parameters
        self._init_kwargs["model"]["encoder_kwargs"] = {
            "type": kwargs.pop(
                "encoder_type", self._init_kwargs["model"]["encoder_type"]
            ),
            "params": {
                "width": self.constructor_kwargs["encoder_kwargs"]["width"],
                "layers": self.constructor_kwargs["encoder_kwargs"]["layers"],
                "link_fn": self.constructor_kwargs["encoder_kwargs"]["link_fn"],
            },
        }

        # Archetype-specific parameters
        archetype_dag_loss_type = kwargs.pop(
            "archetype_dag_loss_type", DEFAULT_DAG_LOSS_TYPE
        )
        self._init_kwargs["model"]["archetype_loss_params"] = {
            "l1": kwargs.get("archetype_l1", 0.0),
            "dag": kwargs.get(
                "archetype_dag_params",
                {
                    "loss_type": archetype_dag_loss_type,
                    "params": kwargs.get(
                        "archetype_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[archetype_dag_loss_type].copy(),
                    ),
                },
            ),
            "init_mat": kwargs.pop("init_mat", None),
            "num_factors": kwargs.pop("num_factors", 0),
            "factor_mat_l1": kwargs.pop("factor_mat_l1", 0),
            "num_archetypes": kwargs.pop("num_archetypes", 16),
        }

        if self._init_kwargs["model"]["archetype_loss_params"]["num_archetypes"] <= 0:
            print(
                "WARNING: num_archetypes is 0. NOTMAD requires archetypes. Setting num_archetypes to 16."
            )
            self._init_kwargs["model"]["archetype_loss_params"]["num_archetypes"] = 16

        # Possibly update values with convenience parameters
        for param, value in self._init_kwargs["model"]["archetype_loss_params"]["dag"][
            "params"
        ].items():
            self._init_kwargs["model"]["archetype_loss_params"]["dag"]["params"][
                param
            ] = kwargs.pop(f"archetype_{param}", value)
        sample_specific_dag_loss_type = kwargs.pop(
            "sample_specific_dag_loss_type", DEFAULT_DAG_LOSS_TYPE
        )

        # Sample-specific parameters
        self._init_kwargs["model"]["sample_specific_loss_params"] = {
            "l1": kwargs.pop("sample_specific_l1", 0.0),
            "dag": kwargs.pop(
                "sample_specific_loss_params",
                {
                    "loss_type": sample_specific_dag_loss_type,
                    "params": kwargs.pop(
                        "sample_specific_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[sample_specific_dag_loss_type].copy(),
                    ),
                },
            ),
        }

        # Possibly update values with convenience parameters
        for param, value in self._init_kwargs["model"]["sample_specific_loss_params"][
            "dag"
        ]["params"].items():
            self._init_kwargs["model"]["sample_specific_loss_params"]["dag"]["params"][
                param
            ] = kwargs.pop(f"sample_specific_{param}", value)

        # Optimization parameters
        self._init_kwargs["model"]["opt_params"] = {
            "learning_rate": kwargs.pop("learning_rate", 1e-3),
            "step": kwargs.pop("step", 50),
        }

        return [
            "archetype_dag_loss_type",
            "archetype_l1",
            "archetype_dag_params",
            "archetype_dag_loss_params",
            "archetype_dag_loss_type",
            "archetype_alpha",
            "archetype_rho",
            "archetype_s",
            "archetype_tol",
            "archetype_loss_params",
            "archetype_use_dynamic_alpha_rho",
            "init_mat",
            "num_factors",
            "factor_mat_l1",
            "sample_specific_dag_loss_type",
            "sample_specific_alpha",
            "sample_specific_rho",
            "sample_specific_s",
            "sample_specific_tol",
            "sample_specific_loss_params",
            "sample_specific_use_dynamic_alpha_rho",
        ]

    def __init__(self, **kwargs):
        super().__init__(
            NOTMAD,
            extra_model_kwargs=[
                "sample_specific_loss_params",
                "archetype_loss_params",
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

    def predict_params(
        self, C: np.ndarray, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predicts context-specific Bayesian network parameters as linear coefficients in a linear structural equation model (SEM).

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            **kwargs: Keyword arguments for the contextualized.dags.GraphTrainer's predict_params method.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The linear coefficients of the predicted context-specific Bayesian network parameters (n_samples, n_features, n_features). Returned as lists of individual bootstraps if individual_preds is True.
        """
        # No mus for NOTMAD at present.
        return super().predict_params(C, model_includes_mus=False, **kwargs)

    def predict_networks(
        self, C: np.ndarray, project_to_dag: bool = True, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predicts context-specific Bayesian networks.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            project_to_dag (bool, optional): If True, guarantees returned graphs are DAGs by trimming edges until acyclicity is satisified. Defaults to True.
            **kwargs: Keyword arguments for the contextualized.dags.GraphTrainer's predict_params method.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The linear coefficients of the predicted context-specific Bayesian network parameters (n_samples, n_features, n_features). Returned as lists of individual bootstraps if individual_preds is True.
        """
        if kwargs.pop("with_offsets", False):
            print("No offsets can be returned by NOTMAD.")
        betas = self.predict_params(
            C, uses_y=False, project_to_dag=project_to_dag, **kwargs
        )
        return betas

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Measures mean-squared errors.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features)
            X (np.ndarray): The data matrix (n_samples, n_features)
            individual_preds (bool, optional): If True, returns the predictions for each bootstrap. Defaults to False.
            **kwargs: Keyword arguments for the contextualized.dags.GraphTrainer's predict_params method.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The mean-squared errors for each sample, or for each bootstrap if individual_preds is True (n_samples).
        """
        betas = self.predict_networks(C, individual_preds=True, **kwargs)
        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for bootstrap in range(len(betas)):
            X_pred = dag_pred_np(X, betas[bootstrap])
            mses[bootstrap, :] = np.mean((X - X_pred) ** 2, axis=1)
        if not individual_preds:
            mses = np.mean(mses, axis=0)
        return mses
