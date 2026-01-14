"""
This class contains tools for solving context-specific regression problems:

Y = g(beta(C)*X + mu(C))

C: Context
X: Explainable features
Y: Outcome, aka response (regreession) or labels (classification)
g: Link Function for contextualized generalized linear models.

Implemented with PyTorch Lightning
"""

from .datamodules import ContextualizedRegressionDataModule  

from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from contextualized.regression.regularizers import REGULARIZERS
from contextualized.regression.losses import MSE
from contextualized.functions import LINK_FUNCTIONS

from contextualized.regression.metamodels import (
    NaiveMetamodel,
    SubtypeMetamodel,
    MultitaskMetamodel,
    TasksplitMetamodel,
    SINGLE_TASK_METAMODELS,
    MULTITASK_METAMODELS,
)


def _resolve_registry_or_callable(maybe_obj, registry, name: str):
    """

    :param maybe_obj:
    :param registry:
    :param name:

    """
    if isinstance(maybe_obj, str):
        try:
            return registry[maybe_obj]
        except KeyError as e:
            raise KeyError(
                f"Unknown {name} '{maybe_obj}'. Valid keys: {list(registry.keys())}"
            ) from e
    if callable(maybe_obj):
        return maybe_obj
    raise TypeError(
        f"{name} must be a string key or a callable, got {type(maybe_obj).__name__}"
    )


def _resolve_loss(maybe_loss):
    """

    :param maybe_loss:

    """
    if isinstance(maybe_loss, str):
        if maybe_loss.lower() == "mse":
            return MSE
        raise KeyError(
            f"Unknown loss_fn '{maybe_loss}'. "
            "Pass a callable loss or the string 'mse'."
        )
    if callable(maybe_loss):
        return maybe_loss
    raise TypeError(
        f"loss_fn must be a string key or a callable, got {type(maybe_loss).__name__}"
    )


def _resolve_regularizer(maybe_reg):
    """

    :param maybe_reg:

    """
    if isinstance(maybe_reg, str):
        try:
            return REGULARIZERS[maybe_reg]
        except KeyError as e:
            raise KeyError(
                f"Unknown model_regularizer '{maybe_reg}'. "
                f"Valid keys: {list(REGULARIZERS.keys())}"
            ) from e
    if callable(maybe_reg):
        return maybe_reg
    raise TypeError(
        "model_regularizer must be a string key or a callable, got "
        f"{type(maybe_reg).__name__}"
    )


class ContextualizedRegressionBase(pl.LightningModule):
    """
    Abstract class for Contextualized Regression.
    """

    # def __init__(
    #     self,
    #     context_dim,
    #     x_dim,
    #     y_dim,
    #     univariate=False,
    #     num_archetypes=10,
    #     encoder_type="mlp",
    #     encoder_kwargs={
    #         "width": 25,
    #         "layers": 1,
    #         "link_fn": "identity",
    #     },
    #     learning_rate=1e-3,
    #     metamodel_type="subtype",
    #     fit_intercept=True,
    #     link_fn="identity",
    #     loss_fn="mse",
    #     model_regularizer="none",
    #     base_y_predictor=None,
    #     base_param_predictor=None,
    #     **kwargs,
    # ):
    #     super().__init__()
    #     self.learning_rate = learning_rate
    #     self.metamodel_type = metamodel_type
    #     self.fit_intercept = fit_intercept
    #     self.link_fn = LINK_FUNCTIONS[link_fn]
    #     if loss_fn == "mse":
    #         self.loss_fn = MSE
    #     else:
    #         raise ValueError("Supported loss_fn's: mse")
    #     self.model_regularizer = REGULARIZERS[model_regularizer]
    #     self.base_y_predictor = base_y_predictor
    #     self.base_param_predictor = base_param_predictor
    #     self._build_metamodel(
    #         context_dim,
    #         x_dim,
    #         y_dim,
    #         univariate,
    #         num_archetypes,
    #         encoder_type,
    #         encoder_kwargs,
    #         **kwargs,
    #     )

    # @abstractmethod
    # def _build_metamodel(
    #     self,
    #     context_dim,
    #     x_dim,
    #     y_dim,
    #     univariate,
    #     num_archetypes,
    #     encoder_type,
    #     encoder_kwargs,
    #     **kwargs
    # ):
    #     """

    #     :param *args:
    #     :param **kwargs:

    #     """
    #     # builds the metamodel
    #     self.metamodel = SINGLE_TASK_METAMODELS[self.metamodel_type](
    #         context_dim,
    #         x_dim,
    #         y_dim,
    #         univariate,
    #         num_archetypes,
    #         encoder_type,
    #         encoder_kwargs,
    #         **kwargs
    #     )

    # @abstractmethod
    # def dataloader(self, C, X, Y, batch_size=32):
    #     """

    #     :param C:
    #     :param X:
    #     :param Y:
    #     :param batch_size:  (Default value = 32)

    #     """
    #     # returns the dataloader for this class

    # @abstractmethod
    # def _batch_loss(self, batch, batch_idx):
    #     """

    #     :param batch:
    #     :param batch_idx:

    #     """
    #     # MSE loss by default

    # @abstractmethod
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     """

    #     :param batch:
    #     :param batch_idx:
    #     :param dataload_idx:

    #     """
    #     # returns predicted params on the given batch

    # @abstractmethod
    # def _params_reshape(self, beta_preds, mu_preds, dataloader):
    #     """

    #     :param beta_preds:
    #     :param mu_preds:
    #     :param dataloader:

    #     """
    #     # reshapes the batch parameter predictions into beta (y_dim, x_dim)

    # @abstractmethod
    # def _y_reshape(self, y_preds, dataloader):
    #     """

    #     :param y_preds:
    #     :param dataloader:

    #     """
    #     # reshapes the batch y predictions into a desirable format

    def forward(self, batch):
        """

        :param *args:

        """
        beta, mu = self.metamodel(batch["contexts"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        if self.base_param_predictor is not None:
            base_beta, base_mu = self.base_param_predictor.predict_params(
                batch["contexts"]
            )
            beta = beta + base_beta.to(beta.device)
            mu = mu + base_mu.to(mu.device)
        return beta, mu

    def configure_optimizers(self):
        """
        Set up optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _batch_size_from_batch(self, batch: dict) -> int:
        """

        :param batch:

        """
        if (
            isinstance(batch, dict)
            and "contexts" in batch
            and isinstance(batch["contexts"], torch.Tensor)
        ):
            return int(batch["contexts"].shape[0])
        return 1

    def _predict_payload(self, batch: dict, **outputs) -> dict:
        """

        :param batch:
        :param **outputs:

        """
        out = {}
        for k in (
            "idx",
            "orig_idx",
            "sample_idx",
            "outcome_idx",
            "predictor_idx",
            "contexts",
            "predictors",
        ):
            if isinstance(batch, dict) and k in batch:
                out[k] = batch[k]

        out.update(outputs)

        for k, v in list(out.items()):
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
        return out


    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        bs = self._batch_size_from_batch(batch)

        self.log(
            "train_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
            batch_size=bs,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=bs,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        bs = self._batch_size_from_batch(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        loss = self._batch_loss(batch, batch_idx)
        bs = self._batch_size_from_batch(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )
        return loss

    def _predict_from_models(self, X, beta_hat, mu_hat):
        """

        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        if isinstance(X, torch.Tensor) and X.dim() == 4 and X.shape[-1] == 1:
            X = X.to(device=beta_hat.device, dtype=beta_hat.dtype)

            if beta_hat.dim() == 3:
                beta_hat = beta_hat.unsqueeze(-1)
            if beta_hat.dim() != 4 or beta_hat.shape[-1] != 1:
                raise RuntimeError(
                    f"Univariate expects beta_hat (B,y,x,1); got {beta_hat.shape}"
                )

            if not isinstance(mu_hat, torch.Tensor):
                mu_hat = torch.as_tensor(
                    mu_hat, device=beta_hat.device, dtype=beta_hat.dtype
                )
            else:
                mu_hat = mu_hat.to(device=beta_hat.device, dtype=beta_hat.dtype)

            if mu_hat.dim() == 2:
                mu_hat = (
                    mu_hat.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, beta_hat.shape[1], beta_hat.shape[2], 1)
                )
            elif mu_hat.dim() == 3:
                if mu_hat.shape[-1] == 1:
                    mu_hat = mu_hat.unsqueeze(-1).expand(
                        -1, beta_hat.shape[1], beta_hat.shape[2], 1
                    )
                else:
                    mu_hat = mu_hat.unsqueeze(-1)
            elif mu_hat.dim() == 4 and mu_hat.shape[-1] == 1:
                pass
            else:
                raise RuntimeError(
                    f"Unsupported mu_hat shape for univariate: {mu_hat.shape}"
                )

            out = (beta_hat * X).sum(dim=-1, keepdim=True) + mu_hat
            return self.link_fn(out)

        if not isinstance(beta_hat, torch.Tensor):
            raise RuntimeError(f"beta_hat must be a tensor, got {type(beta_hat)}")

        if beta_hat.dim() == 4 and beta_hat.shape[-1] == 1:
            beta_hat = beta_hat.squeeze(-1)

        if beta_hat.dim() != 3:
            raise RuntimeError(
                f"_predict_from_models expects beta_hat with shape (B, y, x) "
                f"or (B, y, x, 1); got {beta_hat.shape}"
            )

        B, y_dim, x_dim = beta_hat.shape

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=beta_hat.device, dtype=beta_hat.dtype)
        else:
            X = X.to(device=beta_hat.device, dtype=beta_hat.dtype)

        if X.dim() == 2:
            if X.shape[0] != B:
                raise RuntimeError(
                    f"X batch dim {X.shape[0]} != beta_hat batch dim {B}. "
                    f"X.shape={X.shape}, beta_hat.shape={beta_hat.shape}"
                )
            if X.shape[1] != x_dim:
                raise RuntimeError(
                    f"X feature dim {X.shape[1]} != x_dim {x_dim}. "
                    f"X.shape={X.shape}, beta_hat.shape={beta_hat.shape}"
                )
            X = X.unsqueeze(1).expand(-1, y_dim, -1)

        elif X.dim() == 3:
            if X.shape[0] != B:
                raise RuntimeError(
                    f"X batch dim {X.shape[0]} != beta_hat batch dim {B}. "
                    f"X.shape={X.shape}, beta_hat.shape={beta_hat.shape}"
                )

            if X.shape[1] == y_dim and X.shape[2] == x_dim:
                pass
            elif X.shape[1] == 1 and X.shape[2] == x_dim:
                X = X.expand(-1, y_dim, -1)
            elif X.shape[1] == x_dim and X.shape[2] == y_dim and x_dim == y_dim:
                X = X.permute(0, 2, 1)
            else:
                raise RuntimeError(
                    f"Unexpected X shape {X.shape} for beta_hat {beta_hat.shape}. "
                    "Cannot safely align dimensions."
                )
        else:
            raise RuntimeError(
                f"Unsupported X.ndim={X.dim()} for _predict_from_models; "
                f"expected 2 or 3. X.shape={X.shape}, beta_hat.shape={beta_hat.shape}"
            )

        if not isinstance(mu_hat, torch.Tensor):
            mu_hat = torch.as_tensor(mu_hat, device=beta_hat.device, dtype=beta_hat.dtype)
        else:
            mu_hat = mu_hat.to(device=beta_hat.device, dtype=beta_hat.dtype)

        if mu_hat.dim() == 4 and mu_hat.shape[-1] == 1:
            mu_hat = mu_hat.squeeze(-1)

        if mu_hat.dim() == 2:
            mu_hat = mu_hat.unsqueeze(-1)
        elif mu_hat.dim() == 3:
            pass
        else:
            raise RuntimeError(
                f"Unsupported mu_hat.ndim={mu_hat.dim()} in _predict_from_models; "
                f"mu_hat.shape={mu_hat.shape}"
            )

        out = (beta_hat * X).sum(dim=-1, keepdim=True) + mu_hat
        return self.link_fn(out)

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        if self.base_y_predictor is not None:
            Y_base = self.base_y_predictor.predict_y(C, X)
            Y = Y + Y_base.to(Y.device)
        return Y

    # def _dataloader(self, C, X, Y, dataset_constructor, **kwargs):
    #     """

    #     :param C:
    #     :param X:
    #     :param Y:
    #     :param dataset_constructor:
    #     :param **kwargs:

    #     """
    #     kwargs["num_workers"] = kwargs.get("num_workers", 0)
    #     kwargs["batch_size"] = kwargs.get("batch_size", 32)
    #     return DataLoader(dataset=DataIterable(dataset_constructor(C, X, Y)), **kwargs)


class ContextualizedRegression(ContextualizedRegressionBase):
    """Supports SubtypeMetamodel and NaiveMetamodel, see selected metamodel for docs"""
    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        num_archetypes=10,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        metamodel_type="subtype",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
        base_y_predictor=None,
        base_param_predictor=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.base_y_predictor = base_y_predictor
        self.base_param_predictor = base_param_predictor
        if metamodel_type == "subtype":
            self.metamodel = SubtypeMetamodel(
                context_dim=context_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                univariate=False,
                num_archetypes=num_archetypes,
                encoder_type=encoder_type,
                encoder_kwargs=encoder_kwargs,
            )
        elif metamodel_type == "naive":
            if num_archetypes is not None:
                raise ValueError("NaiveMetamodel does not support num_archetypes.")
            self.metamodel = NaiveMetamodel(
                context_dim=context_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                univariate=False,
                encoder_type=encoder_type,
                encoder_kwargs=encoder_kwargs,
            )
        else:
            raise ValueError("Supported metamodel_type's: subtype, naive")

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class NaiveContextualizedRegression(ContextualizedRegression):
    """Handle for NaiveMetamodel usage of ContextualizedRegression.
    Does not use archetypes.
    """
    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
        base_y_predictor=None,
        base_param_predictor=None,
    ):
        super().__init__(
            context_dim=context_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            num_archetypes=None,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            learning_rate=learning_rate,
            metamodel_type="naive",
            fit_intercept=fit_intercept,
            link_fn=link_fn,
            loss_fn=loss_fn,
            model_regularizer=model_regularizer,
            base_y_predictor=base_y_predictor,
            base_param_predictor=base_param_predictor,
        )
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])


class MultitaskContextualizedRegression(ContextualizedRegressionBase):
    """See MultitaskMetamodel"""
    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        num_archetypes=10,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.metamodel = MultitaskMetamodel(
            context_dim=context_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            univariate=False,
            num_archetypes=num_archetypes,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
        )

    def forward(self, batch):
        """

        :param batch:

        """
        beta, mu = self.metamodel(batch["contexts"], batch["task"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        return beta, mu

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        return Y

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class TasksplitContextualizedRegression(ContextualizedRegressionBase):
    """See TasksplitMetamodel"""

    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        context_archetypes=10,
        context_encoder_type="mlp",
        context_encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        task_archetypes=10,
        task_encoder_type="mlp",
        task_encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        metamodel_type="tasksplit",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.metamodel_type = metamodel_type
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.metamodel = TasksplitMetamodel(
            context_dim=context_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            univariate=False,
            context_archetypes=context_archetypes,
            context_encoder_type=context_encoder_type,
            context_encoder_kwargs=context_encoder_kwargs,
            task_archetypes=task_archetypes,
            task_encoder_type=task_encoder_type,
            task_encoder_kwargs=task_encoder_kwargs,
        )

    def forward(self, batch):
        """

        :param batch:

        """
        beta, mu = self.metamodel(batch["contexts"], batch["task"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        return beta, mu

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        return Y

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class ContextualizedUnivariateRegression(ContextualizedRegressionBase):
    """Supports SubtypeMetamodel and NaiveMetamodel, see selected metamodel for docs"""
    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        num_archetypes=10,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        metamodel_type="subtype",
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
        base_y_predictor=None,
        base_param_predictor=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.base_y_predictor = base_y_predictor
        self.base_param_predictor = base_param_predictor
        if metamodel_type == "subtype":
            self.metamodel = SubtypeMetamodel(
                context_dim=context_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                univariate=True,
                num_archetypes=num_archetypes,
                encoder_type=encoder_type,
                encoder_kwargs=encoder_kwargs,
            )
        elif metamodel_type == "naive":
            if num_archetypes is not None:
                raise ValueError("NaiveMetamodel does not support num_archetypes.")
            self.metamodel = NaiveMetamodel(
                context_dim=context_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                univariate=True,
                encoder_type=encoder_type,
                encoder_kwargs=encoder_kwargs,
            )
        else:
            raise ValueError("Supported metamodel_type's: subtype, naive")

    def forward(self, batch):
        """

        :param *args:

        """
        beta, mu = self.metamodel(batch["contexts"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        return beta, mu

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class MultitaskContextualizedUnivariateRegression(ContextualizedRegressionBase):
    """See MultitaskMetamodel"""

    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        num_archetypes=10,
        encoder_type="mlp",
        encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.metamodel = MultitaskMetamodel(
            context_dim=context_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            univariate=True,
            num_archetypes=num_archetypes,
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
        )

    def forward(self, batch):
        """

        :param batch:

        """
        beta, mu = self.metamodel(batch["contexts"], batch["task"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        return beta, mu

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        return Y

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class TasksplitContextualizedUnivariateRegression(ContextualizedRegressionBase):
    """See TasksplitMetamodel"""

    def __init__(
        self,
        context_dim,
        x_dim,
        y_dim,
        context_archetypes=10,
        context_encoder_type="mlp",
        context_encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        task_archetypes=10,
        task_encoder_type="mlp",
        task_encoder_kwargs={
            "width": 25,
            "layers": 1,
            "link_fn": "identity",
        },
        learning_rate=1e-3,
        fit_intercept=True,
        link_fn="identity",
        loss_fn="mse",
        model_regularizer="none",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.link_fn = _resolve_registry_or_callable(link_fn, LINK_FUNCTIONS, "link_fn")
        self.loss_fn = _resolve_loss(loss_fn)

        self.model_regularizer = _resolve_regularizer(model_regularizer)

        self.metamodel = TasksplitMetamodel(
            context_dim=context_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            univariate=True,
            context_archetypes=context_archetypes,
            context_encoder_type=context_encoder_type,
            context_encoder_kwargs=context_encoder_kwargs,
            task_archetypes=task_archetypes,
            task_encoder_type=task_encoder_type,
            task_encoder_kwargs=task_encoder_kwargs,
        )

    def forward(self, batch):
        """

        :param batch:

        """
        beta, mu = self.metamodel(batch["contexts"], batch["task"])
        if not self.fit_intercept:
            mu = torch.zeros_like(mu)
        return beta, mu

    def _batch_loss(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        beta_hat, mu_hat = self(batch)
        pred_loss = self.loss_fn(
            batch["outcomes"],
            self._predict_y(batch["contexts"], batch["predictors"], beta_hat, mu_hat),
        )
        reg_loss = self.model_regularizer(beta_hat, mu_hat)
        return pred_loss + reg_loss

    def _predict_y(self, C, X, beta_hat, mu_hat):
        """

        :param C:
        :param X:
        :param beta_hat:
        :param mu_hat:

        """
        Y = self._predict_from_models(X, beta_hat, mu_hat)
        return Y

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class ContextualizedCorrelation(ContextualizedUnivariateRegression):
    """Using univariate contextualized regression to estimate Pearson's correlation
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        beta_hat = beta_hat.squeeze(-1)

        beta_hat_T = beta_hat.transpose(1, 2)
        signs = torch.sign(beta_hat)
        signs[signs != signs.transpose(1, 2)] = 0
        correlations = signs * torch.sqrt(torch.abs(beta_hat * beta_hat_T))

        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(
            batch, betas=beta_hat, mus=mu_hat, correlations=correlations
        )


class MultitaskContextualizedCorrelation(MultitaskContextualizedUnivariateRegression):
    """Using multitask univariate contextualized regression to estimate Pearson's correlation
    See TasksplitMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])


class TasksplitContextualizedCorrelation(TasksplitContextualizedUnivariateRegression):
    """Using multitask univariate contextualized regression to estimate Pearson's correlation
    See TasksplitMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])


class ContextualizedNeighborhoodSelection(ContextualizedRegression):
    """Using singletask multivariate contextualized regression to do edge-regression for
    estimating conditional dependencies
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(
        self,
        context_dim,
        x_dim,
        model_regularizer=REGULARIZERS["l1"](1e-3, mu_ratio=0),
        **kwargs,
    ):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(
            context_dim, x_dim, x_dim, model_regularizer=model_regularizer, **kwargs
        )
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.register_buffer("diag_mask", torch.ones(x_dim, x_dim) - torch.eye(x_dim))

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        beta_hat = beta_hat * self.diag_mask.expand(beta_hat.shape[0], -1, -1)

        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)


class ContextualizedMarkovGraph(ContextualizedRegression):
    """Using singletask multivariate contextualized regression to do edge-regression for
    estimating conditional dependencies
    See SubtypeMetamodel for assumptions and full docstring


    """

    def __init__(self, context_dim, x_dim, **kwargs):
        if "y_dim" in kwargs:
            del kwargs["y_dim"]
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
        self.save_hyperparameters(ignore=["base_y_predictor", "base_param_predictor"])

        self.register_buffer("diag_mask", torch.ones(x_dim, x_dim) - torch.eye(x_dim))

    def predict_step(self, batch, batch_idx):
        beta_hat, mu_hat = self(batch)
        beta_hat = beta_hat + beta_hat.transpose(1, 2)
        beta_hat = beta_hat * self.diag_mask.expand(beta_hat.shape[0], -1, -1)

        mu_hat = mu_hat if mu_hat.dim() >= 3 else mu_hat.unsqueeze(-1)
        return self._predict_payload(batch, betas=beta_hat, mus=mu_hat)
