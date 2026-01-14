# contextualized/regression/datamodules.py
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from .datasets import (
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)

TensorLike = Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
IndexLike = Optional[Union[Sequence[int], np.ndarray, torch.Tensor]]

TASK_TO_DATASET = {
    "singletask_multivariate": MultivariateDataset,
    "singletask_univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}


def _to_tensor(x: TensorLike, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, copy=False)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy(copy=False)
    return torch.as_tensor(x, dtype=dtype)


def _maybe_index(x: torch.Tensor, idx: IndexLike) -> torch.Tensor:
    if idx is None:
        return x
    if isinstance(idx, torch.Tensor):
        return x[idx]
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return x[idx]
    return x[torch.as_tensor(idx, dtype=torch.long)]


def _to_index_tensor(idx: IndexLike) -> Optional[torch.Tensor]:
    """Normalize an index-like into a 1D CPU LongTensor."""
    if idx is None:
        return None
    if isinstance(idx, torch.Tensor):
        out = idx.to(dtype=torch.long, device="cpu")
    elif isinstance(idx, np.ndarray):
        out = torch.as_tensor(idx, dtype=torch.long, device="cpu")
    else:
        out = torch.as_tensor(idx, dtype=torch.long, device="cpu")
    return out.view(-1)


class ContextualizedRegressionDataModule(pl.LightningDataModule):
    """
    DataModule that returns map-style datasets for contextualized regression,
    allowing Lightning's Trainer (DDP) to auto-attach DistributedSampler and shard data.

    give  ∈ {
        "singletask_multivariate",
        "singletask_univariate",
        "multitask_multivariate",
        "multitask_univariate",
    }
    """

    def __init__(
        self,
        C: TensorLike,
        X: TensorLike,
        Y: Optional[TensorLike],
        *,
        task_type: str,
        train_idx: IndexLike = None,
        val_idx: IndexLike = None,
        test_idx: IndexLike = None,
        predict_idx: IndexLike = None,
        splitter: Optional[
            Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                     Tuple[IndexLike, IndexLike, IndexLike]]
        ] = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        predict_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        shuffle_train: bool = True,
        shuffle_eval: bool = False,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        if task_type not in TASK_TO_DATASET:
            raise ValueError(
                f"Unknown task_type={task_type!r}. "
                f"Expected one of {list(TASK_TO_DATASET)}."
            )
        self.task_type = task_type

        self._C_raw = C
        self._X_raw = X
        self._Y_raw = Y

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.predict_idx = predict_idx
        self.splitter = splitter

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.predict_batch_size = predict_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = bool(persistent_workers and num_workers > 0)
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.shuffle_eval = shuffle_eval
        self.dtype = dtype

        self.C: Optional[torch.Tensor] = None
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.ds_predict = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        C = _to_tensor(self._C_raw, self.dtype)
        X = _to_tensor(self._X_raw, self.dtype)
        Y = None if self._Y_raw is None else _to_tensor(self._Y_raw, self.dtype)

        if self.train_idx is None and self.val_idx is None and self.test_idx is None:
            if self.splitter is not None:
                tr, va, te = self.splitter(C, X, Y)
                self.train_idx, self.val_idx, self.test_idx = tr, va, te

        if self.predict_idx is None:
            if self.test_idx is not None:
                self.predict_idx = self.test_idx
            else:
                self.predict_idx = torch.arange(C.shape[0], dtype=torch.long)

        def _mk_dataset(idx: IndexLike):
            if idx is None:
                return None

            idx_t = _to_index_tensor(idx)

            C_s = _maybe_index(C, idx_t)
            X_s = _maybe_index(X, idx_t)
            Y_s = None if (Y is None) else _maybe_index(Y, idx_t)
            ds_cls = TASK_TO_DATASET[self.task_type]

            if Y_s is None:
                Y_s = X_s

            return ds_cls(C_s, X_s, Y_s, orig_idx=idx_t, dtype=self.dtype)

        self.ds_train = _mk_dataset(self.train_idx)
        self.ds_val = _mk_dataset(self.val_idx)
        self.ds_test = _mk_dataset(self.test_idx)
        self.ds_predict = _mk_dataset(self.predict_idx)

        self.C, self.X, self.Y = C, X, Y

    def _common_dl_kwargs(self, batch_size: int, *, drop_last: Optional[bool] = None) -> Dict:
        return {
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.num_workers > 0 and self.persistent_workers),
            "drop_last": self.drop_last if drop_last is None else bool(drop_last),
        }

    def train_dataloader(self) -> DataLoader:
        if self.ds_train is None:
            raise RuntimeError("train dataset is not set; provide train_idx or splitter.")
        return DataLoader(
            dataset=self.ds_train,
            shuffle=self.shuffle_train,
            **self._common_dl_kwargs(self.train_batch_size, drop_last=self.drop_last),
        )

    def val_dataloader(self):
        if self.ds_val is None:
            return None
        return DataLoader(
            dataset=self.ds_val,
            shuffle=self.shuffle_eval,
            **self._common_dl_kwargs(self.val_batch_size, drop_last=False),
        )

    def test_dataloader(self) -> DataLoader:
        if self.ds_test is None:
            raise RuntimeError("test dataset is not set; provide test_idx or splitter.")
        return DataLoader(
            dataset=self.ds_test,
            shuffle=self.shuffle_eval,
            **self._common_dl_kwargs(self.test_batch_size, drop_last=False),
        )

    def predict_dataloader(self) -> DataLoader:
        if self.ds_predict is None:
            raise RuntimeError("predict dataset is not set; provide predict_idx/test_idx.")
        return DataLoader(
            dataset=self.ds_predict,
            shuffle=False,
            **self._common_dl_kwargs(self.predict_batch_size, drop_last=False),
        )
