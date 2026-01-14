"""
An sklearn-like wrapper for Contextualized models.
"""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from contextualized.functions import LINK_FUNCTIONS
from contextualized.regression import LOSSES, REGULARIZERS

# Prefer the new, DDP-safe DataModule path when available.
try:
    from contextualized.regression.datamodules import ContextualizedRegressionDataModule
except Exception:  
    ContextualizedRegressionDataModule = None  


DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_BOOTSTRAPS = 1
DEFAULT_ES_PATIENCE = 1
DEFAULT_VAL_BATCH_SIZE = 16
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_TEST_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_ENCODER_TYPE = "mlp"
DEFAULT_ENCODER_WIDTH = 25
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_ENCODER_LINK_FN = LINK_FUNCTIONS["identity"]
DEFAULT_NORMALIZE = False


def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    if _dist_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _is_main_process() -> bool:
    return _rank() == 0


def _world_size_env() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


class SKLearnWrapper:
    """
    An sklearn-like wrapper for Contextualized models.

    Args:
        base_constructor (callable/class): LightningModule constructor for the model.
        extra_model_kwargs (list[str] or set[str]): Extra kw names allowed in "model".
        extra_data_kwargs (list[str] or set[str]): Extra kw names allowed in "data".
        trainer_constructor (class): Trainer class (should provide predict_y / predict_params for DDP-safe inference).
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        loss_fn (torch.nn.Module, optional): Loss function. Defaults to LOSSES["mse"].
        link_fn (torch.nn.Module, optional): Link function. Defaults to LINK_FUNCTIONS["identity"].
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to
            context-specific parameters or context-specific offsets.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1
            vs l2 parameter norms.
        normalize (bool, optional): If True, automatically standardize inputs during training and inverse-transform
            predictions. Defaults to False.
    """

    def _set_defaults(self) -> None:
        self.default_learning_rate = DEFAULT_LEARNING_RATE
        self.default_n_bootstraps = DEFAULT_N_BOOTSTRAPS
        self.default_es_patience = DEFAULT_ES_PATIENCE
        self.default_train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
        self.default_test_batch_size = DEFAULT_TEST_BATCH_SIZE
        self.default_val_batch_size = DEFAULT_VAL_BATCH_SIZE
        self.default_val_split = DEFAULT_VAL_SPLIT
        self.default_encoder_width = DEFAULT_ENCODER_WIDTH
        self.default_encoder_layers = DEFAULT_ENCODER_LAYERS
        self.default_encoder_link_fn = DEFAULT_ENCODER_LINK_FN
        self.default_encoder_type = DEFAULT_ENCODER_TYPE
        self.default_normalize = DEFAULT_NORMALIZE

    def __init__(
        self,
        base_constructor,
        extra_model_kwargs,
        extra_data_kwargs,
        trainer_constructor,
        **kwargs,
    ):
        self._set_defaults()

        self.base_constructor = base_constructor
        self.trainer_constructor = trainer_constructor

        self._trainer_init_kwargs = kwargs.pop("trainer_kwargs", None)

        self.n_bootstraps: int = 1
        self.models: Optional[List[Any]] = None
        self.trainers: Optional[List[Any]] = None
        self.dataloaders: Optional[Dict[str, List[Any]]] = None

        self.normalize: bool = bool(kwargs.pop("normalize", self.default_normalize))
        self.scalers: Dict[str, Optional[StandardScaler]] = {"C": None, "X": None, "Y": None}

        self.context_dim: Optional[int] = None
        self.x_dim: Optional[int] = None
        self.y_dim: Optional[int] = None

        self.accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"

        self.acceptable_kwargs: Dict[str, List[str]] = {
            "data": [
                "train_batch_size",
                "val_batch_size",
                "test_batch_size",
                "predict_batch_size",
                "C_val",
                "X_val",
                "Y_val",
                "val_split",
                "random_state",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "drop_last",
                "shuffle_train",
                "shuffle_eval",
                "dtype",
            ],

            "model": [
                "loss_fn",
                "link_fn",
                "univariate",
                "encoder_type",
                "encoder_kwargs",
                "model_regularizer",
                "num_archetypes",
                "learning_rate",
                "context_dim",
                "x_dim",
                "y_dim",
                "width",
                "layers",
                "encoder_link_fn",
            ],
            "trainer": [
                "max_epochs",
                "check_val_every_n_epoch",
                "val_check_interval",
                "callbacks",
                "callback_constructors",
                "accelerator",
                "devices",
                "strategy",
                "plugins",
                "logger",
                "enable_checkpointing",
                "num_sanity_val_steps",
                "default_root_dir",
                "log_every_n_steps",
                "precision",
                "enable_progress_bar",
                "limit_val_batches",
            ],
            "fit": [],
            "wrapper": [
                "n_bootstraps",
                "es_patience",
                "es_monitor",
                "es_mode",
                "es_min_delta",
                "es_verbose",
                "normalize",
            ],
        }

        self._update_acceptable_kwargs("model", extra_model_kwargs)
        self._update_acceptable_kwargs("data", extra_data_kwargs)
        self._update_acceptable_kwargs(
            "model", kwargs.pop("remove_model_kwargs", []), acceptable=False
        )
        self._update_acceptable_kwargs(
            "data", kwargs.pop("remove_data_kwargs", []), acceptable=False
        )

        self.convenience_kwargs = [
            "alpha",
            "l1_ratio",
            "mu_ratio",
            "subtype_probabilities",
            "width",
            "layers",
            "encoder_link_fn",
        ]

        self.constructor_kwargs = self._organize_constructor_kwargs(**kwargs)

        if "encoder_kwargs" in self.constructor_kwargs:
            ek = self.constructor_kwargs["encoder_kwargs"]
            ek["width"] = kwargs.pop("width", ek.get("width", self.default_encoder_width))
            ek["layers"] = kwargs.pop("layers", ek.get("layers", self.default_encoder_layers))
            ek["link_fn"] = kwargs.pop(
                "encoder_link_fn", ek.get("link_fn", self.default_encoder_link_fn)
            )
        else:
            self.constructor_kwargs["width"] = kwargs.pop(
                "width", self.constructor_kwargs.get("width", self.default_encoder_width)
            )
            self.constructor_kwargs["layers"] = kwargs.pop(
                "layers", self.constructor_kwargs.get("layers", self.default_encoder_layers)
            )
            self.constructor_kwargs["encoder_link_fn"] = kwargs.pop(
                "encoder_link_fn",
                self.constructor_kwargs.get("encoder_link_fn", self.default_encoder_link_fn),
            )

        self.not_constructor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self.constructor_kwargs and k not in self.convenience_kwargs
        }

        self._init_kwargs, unrecognized = self._organize_kwargs(**self.not_constructor_kwargs)

        for k, v in self.constructor_kwargs.items():
            self._init_kwargs["model"][k] = v

        if isinstance(self._trainer_init_kwargs, dict):
            self._init_kwargs["trainer"].update(self._trainer_init_kwargs)

        recognized_private = set(self._parse_private_init_kwargs(**kwargs))
        for kw in unrecognized:
            if kw not in recognized_private:
                print(f"Received unknown keyword argument {kw}, probably ignoring.")

    def _parse_private_fit_kwargs(self, **kwargs) -> List[str]:
        """
        Parse private (model-specific) kwargs passed to fit function.
        Return the list of parsed kwargs.
        """
        return []

    def _parse_private_init_kwargs(self, **kwargs) -> List[str]:
        """
        Parse private (model-specific) kwargs passed to constructor.
        Return the list of parsed kwargs.
        """
        return []

    def _update_acceptable_kwargs(
        self, category, new_kwargs, acceptable: bool = True
    ) -> None:
        """
        Helper function to update the acceptable kwargs.

        If acceptable=True, the new kwargs will be added to the list of acceptable kwargs.
        If acceptable=False, the new kwargs will be removed from the list of acceptable kwargs.
        """
        new_kwargs = list(new_kwargs) if new_kwargs is not None else []
        if acceptable:
            self.acceptable_kwargs[category] = list(
                set(self.acceptable_kwargs[category]).union(set(new_kwargs))
            )
        else:
            self.acceptable_kwargs[category] = list(
                set(self.acceptable_kwargs[category]) - set(new_kwargs)
            )

    def _organize_kwargs(self, **kwargs) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Private helper function to organize kwargs passed to constructor or fit function.
        Organizes kwargs into data, model, trainer, fit, and wrapper categories.
        """
        out = {cat: {} for cat in self.acceptable_kwargs}
        unknown: List[str] = []
        for k, v in kwargs.items():
            placed = False
            for cat, allowed in self.acceptable_kwargs.items():
                if k in allowed:
                    out[cat][k] = v
                    placed = True
                    break
            if not placed:
                unknown.append(k)
        return out, unknown

    def _organize_constructor_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Helper function to set all the default constructor kwargs or changes allowed.
        """
        ctor: Dict[str, Any] = {}

        def maybe_add(kw, default_val):
            if kw in self.acceptable_kwargs["model"]:
                ctor[kw] = kwargs.get(kw, default_val)

        maybe_add("link_fn", LINK_FUNCTIONS["identity"])
        maybe_add("univariate", False)
        maybe_add("encoder_type", self.default_encoder_type)
        maybe_add("loss_fn", LOSSES["mse"])

        if "encoder_kwargs" in self.acceptable_kwargs["model"]:
            ctor["encoder_kwargs"] = kwargs.get(
                "encoder_kwargs",
                {
                    "width": kwargs.get("encoder_width", self.default_encoder_width),
                    "layers": kwargs.get("encoder_layers", self.default_encoder_layers),
                    "link_fn": kwargs.get("encoder_link_fn", self.default_encoder_link_fn),
                },
            )
            if kwargs.get("subtype_probabilities", False):
                ctor["encoder_kwargs"]["link_fn"] = LINK_FUNCTIONS["softmax"]
        else:
            maybe_add("width", self.default_encoder_width)
            maybe_add("layers", self.default_encoder_layers)
            maybe_add("encoder_link_fn", self.default_encoder_link_fn)
            if kwargs.get("subtype_probabilities", False):
                ctor["encoder_link_fn"] = LINK_FUNCTIONS["softmax"]

        if "model_regularizer" in self.acceptable_kwargs["model"]:
            alpha = float(kwargs.get("alpha", 0.0) or 0.0)
            if alpha > 0:
                ctor["model_regularizer"] = REGULARIZERS["l1_l2"](
                    alpha,
                    kwargs.get("l1_ratio", 1.0),
                    kwargs.get("mu_ratio", 0.5),
                )
            else:
                ctor["model_regularizer"] = kwargs.get(
                    "model_regularizer", REGULARIZERS["none"]
                )

        return ctor

    def _maybe_scale_C(self, C: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["C"] is not None:
            return self.scalers["C"].transform(C)
        return C

    def _maybe_scale_X(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.scalers["X"] is not None:
            return self.scalers["X"].transform(X)
        return X

    def _nanrobust_mean(self, arr: np.ndarray, axis: int = 0) -> np.ndarray:
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=axis)
        if np.isnan(mean).any():
            raise RuntimeError(
                "All bootstraps produced non-finite predictions for some items."
            )
        return mean

    def _default_num_workers(self, devices: int) -> int:
        try:
            n_cpu = os.cpu_count() or 0
        except Exception:
            n_cpu = 0
        if n_cpu <= 0:
            return 0
        if self.accelerator != "gpu":
            return min(2, n_cpu)

        world = max(1, _world_size_env() if _world_size_env() > 1 else devices)
        cpu_per_rank = max(1, n_cpu // world)
        return int(min(4, max(2, cpu_per_rank // 2)))

    def _safe_val_split(self, n: int, val_split: float) -> float:
        vs = float(val_split)
        if vs <= 0.0:
            return 0.0
        if int(round(n * vs)) < 2:
            return 0.0
        return vs

    def _resolve_train_val_arrays(
        self,
        C: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray],
        *,
        C_val: Optional[np.ndarray],
        X_val: Optional[np.ndarray],
        Y_val: Optional[np.ndarray],
        Y_required: bool,
        val_split: float,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        if (
            C_val is not None
            and X_val is not None
            and (not Y_required or Y_val is not None)
        ):
            n_tr = int(C.shape[0])
            C_all = np.concatenate([C, C_val], axis=0)
            X_all = np.concatenate([X, X_val], axis=0)

            if Y is None:
                Y_all = None
            else:
                if Y_val is None and Y_required:
                    raise ValueError("Y_val is required when Y is provided.")
                Y_all = np.concatenate([Y, Y_val], axis=0) if Y_val is not None else Y

            train_idx = np.arange(n_tr)
            val_idx = np.arange(n_tr, int(C_all.shape[0]))
            return C_all, X_all, Y_all, train_idx, val_idx

        n = int(C.shape[0])
        vs = self._safe_val_split(n, val_split)
        if vs <= 0.0:
            return C, X, Y, np.arange(n), None

        split_kwargs = dict(test_size=vs, shuffle=shuffle)
        if random_state is not None:
            split_kwargs["random_state"] = random_state

        tr_idx, va_idx = train_test_split(np.arange(n), **split_kwargs)
        return C, X, Y, tr_idx, va_idx


    def _build_datamodule(
        self,
        C: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray],
        *,
        train_idx: Optional[np.ndarray],
        val_idx: Optional[np.ndarray],
        test_idx: Optional[np.ndarray],
        predict_idx: Optional[np.ndarray],
        data_kwargs: Dict[str, Any],
        task_type: str,
    ):
        if ContextualizedRegressionDataModule is None:
            raise RuntimeError(
                "ContextualizedRegressionDataModule is not available in this installation."
            )

        dk = {
            "train_batch_size": self.default_train_batch_size,
            "val_batch_size": self.default_val_batch_size,
            "test_batch_size": self.default_test_batch_size,
            "predict_batch_size": self.default_val_batch_size,
            "num_workers": 0,
            "pin_memory": (self.accelerator == "gpu"),
            "persistent_workers": False,
            "drop_last": False,
            "shuffle_train": True,
            "shuffle_eval": False,
            "dtype": torch.float,
        }
        dk.update(data_kwargs or {})

        return ContextualizedRegressionDataModule(
            C=C,
            X=X,
            Y=Y,
            task_type=task_type,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            predict_idx=predict_idx,
            train_batch_size=dk["train_batch_size"],
            val_batch_size=dk["val_batch_size"],
            test_batch_size=dk["test_batch_size"],
            predict_batch_size=dk["predict_batch_size"],
            num_workers=dk["num_workers"],
            pin_memory=dk["pin_memory"],
            persistent_workers=dk["persistent_workers"],
            drop_last=dk["drop_last"],
            shuffle_train=dk["shuffle_train"],
            shuffle_eval=dk["shuffle_eval"],
            dtype=dk["dtype"],
        )

    def _use_datamodule_for_model(self, model: Any) -> bool:
        if ContextualizedRegressionDataModule is None:
            return False
        return not callable(getattr(model, "dataloader", None))

    def _organize_and_expand_fit_kwargs(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Private function to organize kwargs passed to constructor or fit function.
        """
        organized, unrecognized = self._organize_kwargs(**kwargs)
        recognized_private = set(self._parse_private_fit_kwargs(**kwargs))
        for kw in unrecognized:
            if kw not in recognized_private:
                print(f"Received unknown keyword argument {kw}, probably ignoring.")

        for category, cat_kwargs in self._init_kwargs.items():
            for k, v in cat_kwargs.items():
                organized[category].setdefault(k, v)

        def maybe_add(cat: str, k: str, default_val: Any) -> None:
            if k in self.acceptable_kwargs[cat]:
                organized[cat][k] = organized[cat].get(k, default_val)

        maybe_add("model", "learning_rate", self.default_learning_rate)
        maybe_add("model", "context_dim", self.context_dim)
        maybe_add("model", "x_dim", self.x_dim)
        maybe_add("model", "y_dim", self.y_dim)

        if organized["model"].get("num_archetypes", 1) == 0:
            organized["model"].pop("num_archetypes", None)

        maybe_add("data", "train_batch_size", self.default_train_batch_size)
        maybe_add("data", "val_batch_size", self.default_val_batch_size)
        maybe_add("data", "test_batch_size", self.default_test_batch_size)
        maybe_add(
            "data",
            "predict_batch_size",
            organized["data"].get("val_batch_size", self.default_val_batch_size),
        )

        maybe_add("trainer", "accelerator", self.accelerator)
        organized["trainer"].setdefault("enable_progress_bar", False)
        organized["trainer"].setdefault("logger", False)
        organized["trainer"].setdefault("num_sanity_val_steps", 0)

        world = _world_size_env()
        launched_externally = world > 1 and (
            os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None
        )

        if "devices" not in organized["trainer"]:
            organized["trainer"]["devices"] = 1 if launched_externally else (world if world > 1 else 1)

        devices_cfg = organized["trainer"].get("devices", 1)
        if isinstance(devices_cfg, int):
            devices = devices_cfg
        elif isinstance(devices_cfg, (list, tuple)):
            devices = len(devices_cfg)
        else:
            devices = 1

        if world > 1 and (not launched_externally) and devices != world:
            if _is_main_process():
                print(
                    f"[WARNING] WORLD_SIZE={world} but devices={devices}; "
                    f"overriding devices -> {world}."
                )
            organized["trainer"]["devices"] = world
            devices = world


        if "strategy" not in organized["trainer"]:
            if devices > 1 or world > 1:
                organized["trainer"]["strategy"] = DDPStrategy(
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                    process_group_backend="nccl" if torch.cuda.is_available() else "gloo",
                )
            else:
                organized["trainer"]["strategy"] = "auto"

        if self.accelerator == "gpu":
            organized["trainer"].setdefault("precision", "16-mixed")
        else:
            organized["trainer"].setdefault("precision", 32)

        maybe_add("data", "num_workers", self._default_num_workers(devices))
        maybe_add("data", "pin_memory", self.accelerator == "gpu")
        maybe_add(
            "data",
            "persistent_workers",
            organized["data"].get("num_workers", 0) > 0,
        )
        maybe_add("data", "drop_last", (devices > 1 or world > 1))
        maybe_add("data", "shuffle_train", True)
        maybe_add("data", "shuffle_eval", False)
        maybe_add("data", "dtype", torch.float)

        maybe_add("wrapper", "n_bootstraps", self.default_n_bootstraps)

        val_split = float(organized["data"].get("val_split", self.default_val_split))
        organized["data"]["val_split"] = val_split

        use_val = self._safe_val_split(10, val_split) > 0.0
        es_patience = organized["wrapper"].get("es_patience", self.default_es_patience)
        es_monitor = organized["wrapper"].get(
            "es_monitor", "val_loss" if use_val else "train_loss"
        )
        es_mode = organized["wrapper"].get("es_mode", "min")
        es_verbose = organized["wrapper"].get("es_verbose", False)
        es_min_delta = organized["wrapper"].get("es_min_delta", 0.0)

        cb_ctors = organized["trainer"].get("callback_constructors", None)
        if cb_ctors is None:
            cb_ctors = []

        organized["trainer"].setdefault("enable_checkpointing", True)

        if es_patience is not None and int(es_patience) > 0:
            cb_ctors.append(
                lambda i: EarlyStopping(
                    monitor=es_monitor,
                    mode=es_mode,
                    patience=int(es_patience),
                    verbose=bool(es_verbose),
                    min_delta=float(es_min_delta),
                )
            )

        if bool(organized["trainer"].get("enable_checkpointing", True)):
            cb_ctors.append(
                lambda i: ModelCheckpoint(
                    monitor=es_monitor,
                    dirpath=f"{kwargs.get('checkpoint_path', './lightning_logs')}/boot_{i}_checkpoints",
                    filename="{epoch}-{val_loss:.4f}",
                )
            )

        organized["trainer"]["callback_constructors"] = cb_ctors
        return organized

    def fit(self, *args, **kwargs) -> None:
        """
        Fit contextualized model to data.

        Args:
            C (np.ndarray): Context array of shape (n_samples, n_context_features)
            X (np.ndarray): Predictor array of shape (N, n_features)
            Y (np.ndarray, optional): Target array of shape (N, n_targets). Defaults to None.
            max_epochs (int, optional): Maximum number of epochs to train for. Defaults to 1.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-3.
            val_split (float, optional): Proportion of data to use for validation and early stopping. Defaults to 0.2.
            n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
            train_batch_size (int, optional): Batch size for training. Defaults to 1.
            val_batch_size (int, optional): Batch size for validation. Defaults to 16.
            test_batch_size (int, optional): Batch size for testing. Defaults to 16.
            es_patience (int, optional): Number of epochs to wait before early stopping. Defaults to 1.
            es_monitor (str, optional): Metric to monitor for early stopping. Defaults to "val_loss".
            es_mode (str, optional): Mode for early stopping. Defaults to "min".
            es_verbose (bool, optional): Whether to print early stopping updates. Defaults to False.
        """
        self.models, self.trainers = [], []
        self.dataloaders = {"train": [], "val": [], "test": []}

        if len(args) < 2:
            raise ValueError("fit expects at least (C, X) as positional args.")

        C = kwargs.pop("C", None)
        X = kwargs.pop("X", None)
        Y = kwargs.pop("Y", None)

        if C is None or X is None:
            C = args[0]
            X = args[1]
            if len(args) >= 3:
                Y = args[2]
        if C is None or X is None:
            raise ValueError("fit requires C and X.")

        C = np.asarray(C)
        X = np.asarray(X)
        if Y is not None:
            Y = np.asarray(Y)

        if self.normalize:
            if self.scalers["C"] is None:
                self.scalers["C"] = StandardScaler().fit(C)
            C = self.scalers["C"].transform(C)

            if self.scalers["X"] is None:
                self.scalers["X"] = StandardScaler().fit(X)
            X = self.scalers["X"].transform(X)

        self.context_dim = int(C.shape[-1])
        self.x_dim = int(X.shape[-1])

        if Y is None:
            Y = X
        else:
            if Y.ndim == 1:
                Y = np.expand_dims(Y, 1)

        if self.normalize and self.scalers["Y"] is not None:
            pass

        if self.normalize and not np.array_equal(np.unique(Y), np.array([0, 1])):
            if self.scalers["Y"] is None:
                self.scalers["Y"] = StandardScaler().fit(Y)
            Y = self.scalers["Y"].transform(Y)

        self.y_dim = int(Y.shape[-1])

        organized = self._organize_and_expand_fit_kwargs(**kwargs)
        self.n_bootstraps = int(
            organized["wrapper"].get("n_bootstraps", self.n_bootstraps)
        )

        val_split = float(organized["data"].get("val_split", self.default_val_split))
        val_split = self._safe_val_split(int(C.shape[0]), val_split)
        organized["data"]["val_split"] = val_split
        use_val = val_split > 0.0

        if not use_val:
            new_ctors = []
            for ctor in organized["trainer"].get("callback_constructors", []):

                def _wrap_ctor(_ctor):
                    def _inner(i):
                        cb = _ctor(i)
                        if (
                            isinstance(cb, EarlyStopping)
                            and isinstance(getattr(cb, "monitor", ""), str)
                            and cb.monitor.startswith("val_")
                        ):
                            return EarlyStopping(
                                monitor="train_loss",
                                mode=getattr(cb, "mode", "min"),
                                patience=getattr(cb, "patience", self.default_es_patience),
                                verbose=getattr(cb, "verbose", False),
                                min_delta=getattr(cb, "min_delta", 0.0),
                            )
                        if (
                            isinstance(cb, ModelCheckpoint)
                            and isinstance(getattr(cb, "monitor", ""), str)
                            and cb.monitor.startswith("val_")
                        ):
                            cb.monitor = None
                        return cb

                    return _inner

                new_ctors.append(_wrap_ctor(ctor))
            organized["trainer"]["callback_constructors"] = new_ctors
            organized["trainer"].setdefault("limit_val_batches", 0)

        C_val = organized["data"].get("C_val", None)
        X_val = organized["data"].get("X_val", None)
        Y_val = organized["data"].get("Y_val", None)

        univariate_flag = bool(organized["model"].get("univariate", False))
        task_type = "singletask_univariate" if univariate_flag else "singletask_multivariate"

        C_all, X_all, Y_all, train_idx, val_idx = self._resolve_train_val_arrays(
            C,
            X,
            Y,
            C_val=C_val,
            X_val=X_val,
            Y_val=Y_val,
            Y_required=True,
            val_split=val_split,
            random_state=organized["data"].get("random_state", None),
        )


        for b in range(self.n_bootstraps):
            model_kwargs = dict(organized["model"])
            model_kwargs.pop("univariate", None)

            model = self.base_constructor(**model_kwargs)

            use_dm = self._use_datamodule_for_model(model)

            trainer_kwargs = copy.deepcopy(organized["trainer"])
            cb_ctors = trainer_kwargs.pop("callback_constructors", [])
            callbacks = list(trainer_kwargs.get("callbacks", []))
            callbacks.extend([ctor(b) for ctor in cb_ctors])
            trainer_kwargs["callbacks"] = callbacks

            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    try:
                        os.makedirs(cb.dirpath, exist_ok=True)
                    except Exception:
                        pass

            from contextualized.regression.trainers import make_trainer_with_env

            trainer = make_trainer_with_env(self.trainer_constructor, **trainer_kwargs)

            if use_dm:
                dm = self._build_datamodule(
                    C=C_all,
                    X=X_all,
                    Y=Y_all,
                    train_idx=train_idx,
                    val_idx=val_idx if use_val else None,
                    test_idx=None,
                    predict_idx=None,
                    data_kwargs=organized["data"],
                    task_type=task_type,
                )

                if _is_main_process():
                    print(
                        f"[RANK {_rank()}] train_idx[:5]={train_idx[:5]}, "
                        f"val_idx[:5]={val_idx[:5] if val_idx is not None else None}"
                    )

                trainer.fit(model, datamodule=dm, **organized["fit"])

                try:
                    dm.setup("fit")
                    self.dataloaders["train"].append(dm.train_dataloader())
                    self.dataloaders["val"].append(dm.val_dataloader() if use_val else None)
                    self.dataloaders["test"].append(None)
                except Exception:
                    self.dataloaders["train"].append(None)
                    self.dataloaders["val"].append(None)
                    self.dataloaders["test"].append(None)

            else:
                train_data = (
                    [C_all[train_idx], X_all[train_idx], Y_all[train_idx]]
                    if Y_all is not None
                    else [C_all[train_idx], X_all[train_idx]]
                )

                val_data = None
                if use_val and val_idx is not None:
                    val_data = (
                        [C_all[val_idx], X_all[val_idx], Y_all[val_idx]]
                        if Y_all is not None
                        else [C_all[val_idx], X_all[val_idx]]
                    )

                train_dl = model.dataloader(
                    *train_data,
                    batch_size=organized["data"].get(
                        "train_batch_size", self.default_train_batch_size
                    ),
                )

                val_dl = None
                if val_data is not None:
                    val_dl = model.dataloader(
                        *val_data,
                        batch_size=organized["data"].get(
                            "val_batch_size", self.default_val_batch_size
                        ),
                    )

                try:
                    trainer.fit(model, train_dl, val_dl, **organized["fit"])
                except Exception:
                    trainer.fit(model, train_dl, **organized["fit"])

                self.dataloaders["train"].append(train_dl)
                self.dataloaders["val"].append(val_dl)
                self.dataloaders["test"].append(None)

            ckpt_cb = next(
                (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
                None,
            )
            if ckpt_cb is not None and getattr(ckpt_cb, "best_model_path", None):
                best_path = ckpt_cb.best_model_path
                if isinstance(best_path, str) and best_path and os.path.exists(best_path):
                    try:
                        best = torch.load(best_path, map_location="cpu")
                        if isinstance(best, dict) and "state_dict" in best:
                            model.load_state_dict(best["state_dict"])
                    except Exception:
                        pass

            self.models.append(model)
            self.trainers.append(trainer)

        return None

    def predict(
        self, C: np.ndarray, X: np.ndarray, individual_preds: bool = False, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """Predict outcomes from context C and predictors X.

        Args:
            C (np.ndarray): Context array of shape (n_samples, n_context_features)
            X (np.ndarray): Predictor array of shape (N, n_features)
            individual_preds (bool, optional): Whether to return individual predictions for each model. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray], None]: Predicted outcomes. If individual_preds is True, returns
            predictions for each bootstrap. Returns None if any trainer returns None.
        """
        if self.models is None or self.trainers is None:
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")

        C = np.asarray(C)
        X = np.asarray(X)
        Cq = self._maybe_scale_C(C)
        Xq = self._maybe_scale_X(X)

        preds_all: List[np.ndarray] = []
        saw_none = False

        for model, trainer in zip(self.models, self.trainers):
            if not hasattr(trainer, "predict_y"):
                raise RuntimeError(
                    "Trainer does not implement predict_y(). "
                    "Use contextualized.regression.trainers.RegressionTrainer (or a subclass)."
                )

            use_dm = self._use_datamodule_for_model(model)

            if use_dm:
                Yq = np.zeros((len(Cq), int(self.y_dim or 1)), dtype=np.float32)

                univariate_flag = bool(self._init_kwargs.get("model", {}).get("univariate", False))
                task_type = (
                    "singletask_univariate"
                    if univariate_flag
                    else "singletask_multivariate"
                )

                dm = self._build_datamodule(
                    C=Cq,
                    X=Xq,
                    Y=Yq,
                    train_idx=None,
                    val_idx=None,
                    test_idx=None,
                    predict_idx=np.arange(len(Cq)),
                    data_kwargs={**self._init_kwargs.get("data", {}), **kwargs},
                    task_type=task_type,
                )
                dm.setup("predict")
                dl = dm.predict_dataloader()
            else:
                dl = model.dataloader(
                    Cq,
                    Xq,
                    np.zeros((len(Cq), int(self.y_dim or 1))),
                    batch_size=kwargs.get(
                        "predict_batch_size", self.default_val_batch_size
                    ),
                )

            yhat = trainer.predict_y(model, dl, **kwargs)
            if yhat is None:
                saw_none = True
                continue

            preds_all.append(np.asarray(yhat, dtype=float))

        if saw_none:
            return None

        predictions = np.array(preds_all, dtype=float)

        if individual_preds:
            out = predictions
        else:
            bad = ~np.isfinite(predictions)
            if bad.any():
                num_bad_boots = np.unique(np.where(bad)[0]).size
                print(
                    f"Warning: {num_bad_boots}/{len(preds_all)} bootstraps produced "
                    f"non-finite predictions; excluding them from the ensemble."
                )
            out = self._nanrobust_mean(predictions, axis=0)

        if self.normalize and self.scalers["Y"] is not None:
            if individual_preds:
                out = np.array([self.scalers["Y"].inverse_transform(p) for p in out])
            else:
                out = self.scalers["Y"].inverse_transform(out)

        return out

    def predict_params(
        self,
        C: np.ndarray,
        individual_preds: bool = False,
        model_includes_mus: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[None, None], None]:
        """
        Predict context-specific model parameters from context C.

        Args:
            C (np.ndarray): Context array of shape (n_samples, n_context_features)
            individual_preds (bool, optional): Whether to return individual model predictions for each bootstrap.
                Defaults to False, averaging across bootstraps.
            model_includes_mus (bool, optional): Whether the model includes context-specific offsets (mu).
                Defaults to True.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[None, None], None]:
            If model_includes_mus is True, returns (betas, mus); otherwise returns betas.
            If individual_preds is True, returns arrays stacked over bootstraps.
            Returns (None, None) or None if any trainer returns None.
        """
        if self.models is None or self.trainers is None:
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")

        C = np.asarray(C)
        Cq = self._maybe_scale_C(C)

        uses_y = bool(kwargs.pop("uses_y", True))

        betas_list: List[np.ndarray] = []
        mus_list: List[np.ndarray] = []
        saw_none = False

        for model, trainer in zip(self.models, self.trainers):
            if not hasattr(trainer, "predict_params"):
                raise RuntimeError(
                    "Trainer does not implement predict_params(). "
                    "Use contextualized.regression.trainers.RegressionTrainer (or a subclass)."
                )

            use_dm = self._use_datamodule_for_model(model)

            if use_dm:
                X_zero = np.zeros((len(Cq), int(self.x_dim or 1)), dtype=np.float32)
                Y_zero = (
                    np.zeros((len(Cq), int(self.y_dim or 1)), dtype=np.float32)
                    if uses_y
                    else None
                )

                univariate_flag = bool(self._init_kwargs.get("model", {}).get("univariate", False))
                task_type = (
                    "singletask_univariate"
                    if univariate_flag
                    else "singletask_multivariate"
                )

                dm = self._build_datamodule(
                    C=Cq,
                    X=X_zero,
                    Y=Y_zero,
                    train_idx=None,
                    val_idx=None,
                    test_idx=None,
                    predict_idx=np.arange(len(Cq)),
                    data_kwargs={**self._init_kwargs.get("data", {}), **kwargs},
                    task_type=task_type,
                )
                dm.setup("predict")
                dl = dm.predict_dataloader()
            else:
                if uses_y:
                    dl = model.dataloader(
                        Cq,
                        np.zeros((len(Cq), int(self.x_dim or 1))),
                        np.zeros((len(Cq), int(self.y_dim or 1))),
                    )
                else:
                    dl = model.dataloader(
                        Cq,
                        np.zeros((len(Cq), int(self.x_dim or 1))),
                    )

            out = trainer.predict_params(model, dl, **kwargs)
            if out is None or (isinstance(out, tuple) and out[0] is None):
                saw_none = True
                continue

            if model_includes_mus:
                b, m = out
                betas_list.append(np.asarray(b))
                mus_list.append(np.asarray(m))
            else:
                betas_list.append(np.asarray(out))

        if saw_none:
            return (None, None) if model_includes_mus else None

        betas = np.array(betas_list)

        if model_includes_mus:
            mus = np.array(mus_list)
            if individual_preds:
                return betas, mus
            return np.mean(betas, axis=0), np.mean(mus, axis=0)

        if individual_preds:
            return betas
        return np.mean(betas, axis=0)
