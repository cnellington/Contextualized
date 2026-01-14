"""
sklearn-like interface to Contextualized Networks.
"""

from typing import *

import numpy as np
import torch
import torch.distributed as dist

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


def _is_distributed() -> bool:
    """Returns True if torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    """Returns the current distributed rank, defaulting to 0 when not distributed."""
    if _is_distributed():
        return dist.get_rank()
    return 0


class ContextualizedNetworks(SKLearnWrapper):
    """
    sklearn-like interface to Contextualized Networks.
    """

    def _split_train_data(
        self,
        C: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        *,
        Y_required: bool = False,
        val_split: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Splits data into train and test sets.

        Notes:
            This override exists to set the default behavior for networks (Y is not required),
            while preserving compatibility with SKLearnWrapper._split_train_data.

        Args:
            C (np.ndarray): Contextual features for each sample.
            X (np.ndarray): The data matrix.
            Y (Optional[np.ndarray], optional): Optional targets. Defaults to None.
            Y_required (bool, optional): Whether Y is required. Defaults to False.
            val_split (Optional[float], optional): Validation split fraction. Defaults to None.
            random_state (Optional[int], optional): Random state for splitting. Defaults to None.
            shuffle (bool, optional): Whether to shuffle before splitting. Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the base implementation.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: The train/test split outputs as returned by
            SKLearnWrapper._split_train_data.
        """
        return super()._split_train_data(
            C,
            X,
            Y,
            Y_required=Y_required,
            val_split=val_split,
            random_state=random_state,
            shuffle=shuffle,
            **kwargs,
        )

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
        None,
    ]:
        """Predicts context-specific networks given contextual features.

        Notes:
            Under DDP, prediction helpers are rank-0 only (by design in the trainers/wrapper).
            In such cases, this method returns None on non-rank-0 processes.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            with_offsets (bool, optional): If True, returns both the network parameters and
                offsets (when available). Defaults to False.
            individual_preds (bool, optional): If True, returns the predictions for each
                bootstrap. Defaults to False.
            **kwargs: Keyword arguments forwarded to predict_params.

        Returns:
            Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, np.ndarray],
            Tuple[List[np.ndarray], List[np.ndarray]], None]:
                The predicted network parameters (and offsets if with_offsets is True).
                Returned as lists of individual bootstraps if individual_preds is True.
                Returns None on non-rank-0 under DDP.
        """
        out = self.predict_params(
            C, individual_preds=individual_preds, uses_y=False, **kwargs
        )
        if out is None:
            return None

        betas, mus = out
        if betas is None:
            return None

        if with_offsets:
            return betas, mus
        return betas

    def predict_X(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Reconstructs the data matrix based on predicted contextualized networks and
        the true data matrix.

        Useful for measuring reconstruction error or for imputation.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            X (np.ndarray): The data matrix (n_samples, n_features).
            individual_preds (bool, optional): If True, returns the predictions for each
                bootstrap. Defaults to False.
            **kwargs: Keyword arguments for the Lightning trainer's prediction method.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The predicted data matrix, or matrices for
            each bootstrap if individual_preds is True (n_samples, n_features).
        """
        return self.predict(C, X, individual_preds=individual_preds, **kwargs)


class ContextualizedCorrelationNetworks(ContextualizedNetworks):
    """
    Contextualized Correlation Networks reveal context-varying feature correlations,
    interaction strengths, and dependencies in feature groups.

    Uses the Contextualized Networks model.

    Notes:
        This implementation includes CPU/DDP-safe prediction behavior:
        - When using a LightningDataModule outside Trainer.fit/predict, setup(stage="predict")
          is called before predict_dataloader().
        - Under DDP, only rank-0 returns numpy outputs; non-rank-0 returns None, while still
          executing the per-model predict loop to avoid collective mismatches/hangs.
    """

    def __init__(self, **kwargs):
        super().__init__(
            ContextualizedCorrelation, [], [], CorrelationTrainer, **kwargs
        )

    def predict_correlation(
        self, C: np.ndarray, individual_preds: bool = True, squared: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Predicts context-specific correlations between features.

        Notes:
            Under DDP, only rank-0 returns numpy outputs. If any per-model prediction returns
            None (rank-0-only behavior), this method returns None.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            individual_preds (bool, optional): If True, returns the predictions for each
                bootstrap. Defaults to True.
            squared (bool, optional): If True, returns the squared correlations. Defaults to True.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The predicted context-specific correlation matrices, or matrices for each
                bootstrap if individual_preds is True (n_samples, n_features, n_features).
                Returns None on non-rank-0 under DDP.
        """
        C_scaled = self._maybe_scale_C(C)
        Y_zero = np.zeros((len(C_scaled), self.x_dim), dtype=np.float32)

        dm = self._build_datamodule(
            C=C_scaled,
            X=np.zeros((len(C_scaled), self.x_dim), dtype=np.float32),
            Y=Y_zero,
            predict_idx=np.arange(len(C_scaled)),
            data_kwargs=dict(
                train_batch_size=self._init_kwargs["data"].get("train_batch_size", 16),
                val_batch_size=self._init_kwargs["data"].get("val_batch_size", 16),
                test_batch_size=self._init_kwargs["data"].get("test_batch_size", 16),
                predict_batch_size=self._init_kwargs["data"].get(
                    "predict_batch_size", 16
                ),
                num_workers=self._init_kwargs["data"].get("num_workers", 0),
                pin_memory=self._init_kwargs["data"].get(
                    "pin_memory", (self.accelerator in ("cuda", "gpu"))
                ),
                persistent_workers=self._init_kwargs["data"].get(
                    "persistent_workers", False
                ),
                drop_last=False,
                shuffle_train=False,
                shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate",
        )

        dm.setup(stage="predict")
        pred_loader = dm.predict_dataloader()

        saw_none = False
        rhos_list: List[np.ndarray] = []

        for i in range(len(self.models)):
            rho_i = self.trainers[i].predict_correlation(self.models[i], pred_loader)
            if rho_i is None:
                saw_none = True
                continue
            rhos_list.append(rho_i)

        if saw_none:
            return None

        rhos = np.array(rhos_list)

        if individual_preds:
            if squared:
                return np.square(rhos)
            return rhos

        mean_rhos = np.mean(rhos, axis=0)
        if squared:
            return np.square(mean_rhos)
        return mean_rhos

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Measures mean-squared errors.

        Notes:
            This method computes MSEs from reconstructions returned by predict_X, including
            handling potential (bootstrap, sample, feature) or (bootstrap, sample, feature, feature)
            tensor shapes, and handling N_hat != N_true by truncation to min(N_hat, N_true).

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            X (np.ndarray): The data matrix (n_samples, n_features).
            individual_preds (bool, optional): If True, returns the MSEs for each bootstrap.
                Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The mean-squared errors for each sample, or for each bootstrap if
                individual_preds is True (n_samples). Returns None on non-rank-0 under DDP.
        """
        X_hat = self.predict_X(C, X, individual_preds=True)
        if X_hat is None:
            return None

        X_hat = np.array(X_hat)

        if X_hat.ndim not in (3, 4):
            raise ValueError(
                f"Unexpected X_hat ndim={X_hat.ndim} with shape {X_hat.shape} in "
                "ContextualizedCorrelationNetworks.measure_mses"
            )

        N_true, F = X.shape

        if X_hat.ndim == 3:
            B, N_hat, F_hat = X_hat.shape
            if F_hat != F:
                raise ValueError(
                    f"Feature dimension mismatch between X_hat (F={F_hat}) and X (F={F}) "
                    "in ContextualizedCorrelationNetworks.measure_mses"
                )

            N_eff = min(N_hat, N_true)
            if N_hat != N_true:
                X_hat = X_hat[:, :N_eff, :]
                X_eff = X[:N_eff, :]
            else:
                X_eff = X

            X_true = X_eff[None, :, :]
            residuals = X_hat - X_true
            mses = (residuals**2).mean(axis=-1)

        else:
            B, N_hat, F1, F2 = X_hat.shape
            if F1 != F:
                raise ValueError(
                    f"Feature dimension mismatch between X_hat (F1={F1}) and X (F={F}) "
                    "in ContextualizedCorrelationNetworks.measure_mses"
                )

            N_eff = min(N_hat, N_true)
            if N_hat != N_true:
                X_hat = X_hat[:, :N_eff, :, :]
                X_eff = X[:N_eff, :]
            else:
                X_eff = X

            X_true = X_eff[None, :, :, None]
            residuals = X_hat - X_true
            mses = (residuals**2).mean(axis=(-1, -2))

        if individual_preds:
            return mses
        return mses.mean(axis=0)


class ContextualizedMarkovNetworks(ContextualizedNetworks):
    """
    Contextualized Markov Networks reveal context-varying feature dependencies, cliques,
    and modules.

    Implemented as Contextualized Gaussian Precision Matrices, directly interpretable as
    Markov Networks.

    Notes:
        This implementation includes CPU/DDP-safe prediction behavior analogous to
        ContextualizedCorrelationNetworks.predict_correlation.
    """

    def __init__(self, **kwargs):
        super().__init__(ContextualizedMarkovGraph, [], [], MarkovTrainer, **kwargs)

    def predict_precisions(
        self, C: np.ndarray, individual_preds: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Predicts context-specific precision matrices.

        Notes:
            Under DDP, only rank-0 returns numpy outputs. If any per-model prediction returns
            None (rank-0-only behavior), this method returns None.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            individual_preds (bool, optional): If True, returns the predictions for each
                bootstrap. Defaults to True.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The predicted context-specific precision matrices, or matrices for each
                bootstrap if individual_preds is True (n_samples, n_features, n_features).
                Returns None on non-rank-0 under DDP.
        """
        C_scaled = self._maybe_scale_C(C)
        Y_zero = np.zeros((len(C_scaled), self.x_dim), dtype=np.float32)

        dm = self._build_datamodule(
            C=C_scaled,
            X=np.zeros((len(C_scaled), self.x_dim), dtype=np.float32),
            Y=Y_zero,
            predict_idx=np.arange(len(C_scaled)),
            data_kwargs=dict(
                train_batch_size=self._init_kwargs["data"].get("train_batch_size", 16),
                val_batch_size=self._init_kwargs["data"].get("val_batch_size", 16),
                test_batch_size=self._init_kwargs["data"].get("test_batch_size", 16),
                predict_batch_size=self._init_kwargs["data"].get(
                    "predict_batch_size", 16
                ),
                num_workers=self._init_kwargs["data"].get("num_workers", 0),
                pin_memory=self._init_kwargs["data"].get(
                    "pin_memory", (self.accelerator in ("cuda", "gpu"))
                ),
                persistent_workers=self._init_kwargs["data"].get(
                    "persistent_workers", False
                ),
                drop_last=False,
                shuffle_train=False,
                shuffle_eval=False,
                dtype=self._init_kwargs["data"].get("dtype", torch.float),
            ),
            task_type="singletask_univariate",
        )

        dm.setup(stage="predict")
        pred_loader = dm.predict_dataloader()

        saw_none = False
        prec_list: List[np.ndarray] = []

        for i in range(len(self.models)):
            p_i = self.trainers[i].predict_precision(self.models[i], pred_loader)
            if p_i is None:
                saw_none = True
                continue
            prec_list.append(p_i)

        if saw_none:
            return None

        precisions = np.array(prec_list)
        if individual_preds:
            return precisions
        return np.mean(precisions, axis=0)

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Measures mean-squared errors.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            X (np.ndarray): The data matrix (n_samples, n_features).
            individual_preds (bool, optional): If True, returns the MSEs for each bootstrap.
                Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The mean-squared errors for each sample, or for each bootstrap if
                individual_preds is True (n_samples). Returns None on non-rank-0 under DDP.
        """
        out = self.predict_networks(C, individual_preds=True, with_offsets=True)
        if out is None:
            return None
        betas, mus = out

        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        F = X.shape[-1]
        for b in range(len(betas)):
            for i in range(F):
                preds = np.array(
                    [
                        X[j].dot(betas[b, j, i, :]) + mus[b, j, i]
                        for j in range(len(X))
                    ]
                )
                residuals = X[:, i] - preds
                mses[b, :] += residuals**2 / F

        if individual_preds:
            return mses
        return np.mean(mses, axis=0)


class ContextualizedBayesianNetworks(ContextualizedNetworks):
    """
    Contextualized Bayesian Networks and Directed Acyclic Graphs (DAGs) reveal
    context-dependent causal relationships, effect sizes, and variable ordering.

    Uses the NOTMAD model.

    Notes:
        This wrapper preserves the HPC/DDP behavior: rank-0 produces arrays, non-rank-0
        returns None where applicable.
    """

    def _parse_private_init_kwargs(self, **kwargs):
        """Parses the kwargs for the NOTMAD model.

        Args:
            **kwargs: Keyword arguments for the NOTMAD model, including the encoder,
                archetype loss, sample-specific loss, and optimization parameters.

        Returns:
            List[str]: Names of kwargs consumed/handled by this parser.
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

        # Sample-specific parameters
        sample_specific_dag_loss_type = kwargs.pop(
            "sample_specific_dag_loss_type", DEFAULT_DAG_LOSS_TYPE
        )
        self._init_kwargs["model"]["sample_specific_loss_params"] = {
            "l1": kwargs.pop("sample_specific_l1", 0.0),
            "dag": kwargs.pop(
                "sample_specific_loss_params",
                {
                    "loss_type": sample_specific_dag_loss_type,
                    "params": kwargs.pop(
                        "sample_specific_dag_loss_params",
                        DEFAULT_DAG_LOSS_PARAMS[
                            sample_specific_dag_loss_type
                        ].copy(),
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
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Predicts context-specific Bayesian network parameters as linear coefficients
        in a linear structural equation model (SEM).

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            **kwargs: Keyword arguments for contextualized.dags.GraphTrainer.predict_params.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The linear coefficients of the predicted context-specific Bayesian network
                parameters (n_samples, n_features, n_features). Returned as lists of
                individual bootstraps if individual_preds is True. Returns None on
                non-rank-0 under DDP.
        """
        # No mus for NOTMAD at present.
        return super().predict_params(C, model_includes_mus=False, **kwargs)

    def predict_networks(
        self, C: np.ndarray, project_to_dag: bool = True, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Predicts context-specific Bayesian networks.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            project_to_dag (bool, optional): If True, guarantees returned graphs are DAGs by
                trimming edges until acyclicity is satisified. Defaults to True.
            **kwargs: Keyword arguments for contextualized.dags.GraphTrainer.predict_params.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The linear coefficients of the predicted context-specific Bayesian network
                parameters (n_samples, n_features, n_features). Returned as lists of
                individual bootstraps if individual_preds is True. Returns None on
                non-rank-0 under DDP.
        """
        if kwargs.pop("with_offsets", False):
            print("No offsets can be returned by NOTMAD.")
        betas = self.predict_params(
            C, uses_y=False, project_to_dag=project_to_dag, **kwargs
        )
        return betas

    def measure_mses(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Measures mean-squared errors.

        Args:
            C (np.ndarray): Contextual features for each sample (n_samples, n_context_features).
            X (np.ndarray): The data matrix (n_samples, n_features).
            individual_preds (bool, optional): If True, returns the MSEs for each bootstrap.
                Defaults to False.
            **kwargs: Keyword arguments for contextualized.dags.GraphTrainer.predict_params.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]:
                The mean-squared errors for each sample, or for each bootstrap if
                individual_preds is True (n_samples). Returns None on non-rank-0 under DDP.
        """
        betas = self.predict_networks(C, individual_preds=True, **kwargs)
        if betas is None:
            return None

        mses = np.zeros((len(betas), len(C)))  # n_bootstraps x n_samples
        for b in range(len(betas)):
            X_pred = dag_pred_np(X, betas[b])
            mses[b, :] = np.mean((X - X_pred) ** 2, axis=1)

        if individual_preds:
            return mses
        return np.mean(mses, axis=0)
