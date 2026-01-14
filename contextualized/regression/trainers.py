"""
PyTorch-Lightning trainers used for Contextualized regression.
"""

from typing import Any, Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import LightningEnvironment
import os
from lightning.pytorch.strategies import DDPStrategy


def _stack_from_preds(preds: List[dict], key: str) -> torch.Tensor:
    """
    Concatenate a tensor field from the list of batch dicts returned by predict().
    """
    preds = _flatten_pl_predict_output(preds)
    parts = []
    for p in preds:
        val = p[key]
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        parts.append(val.detach().cpu())
    return torch.cat(parts, dim=0)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_distributed()) or dist.get_rank() == 0


def _flatten_pl_predict_output(preds):
    """
    Lightning can return:
    - list[dict] (single dataloader)
    - list[list[dict]] (multiple dataloaders)

    Normalize to list[dict].
    """
    if preds is None:
        return []
    if len(preds) > 0 and isinstance(preds[0], list):
        out = []
        for sub in preds:
            out.extend(sub)
        return out
    return preds


def _to_numpy_cpu(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pack_keys_from_preds(preds: list, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """
    Pack only requested keys from list[dict] predictions into numpy arrays.
    Concats on axis 0.
    """
    preds = _flatten_pl_predict_output(preds)
    if not preds:
        return {}

    packed: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    for p in preds:
        for k in keys:
            if k in p:
                v = _to_numpy_cpu(p[k])
                if v is not None:
                    packed[k].append(v)

    out: Dict[str, np.ndarray] = {}
    for k, parts in packed.items():
        if not parts:
            continue
        out[k] = np.concatenate(parts, axis=0)
    return out


def _gather_object_to_rank0(obj):
    """
    Gather arbitrary Python objects to rank 0.

    Returns:
    - list[obj] on rank 0
    - None on other ranks
    """
    if not _is_distributed():
        return [obj]

    world_size = dist.get_world_size()
    if world_size == 1:
        return [obj]

    if _is_main_process():
        gathered = [None for _ in range(world_size)]
        dist.gather_object(obj, object_gather_list=gathered, dst=0)
        return gathered
    else:
        dist.gather_object(obj, object_gather_list=None, dst=0)
        return None


def _merge_packed_payloads(
    payloads: List[Optional[Dict[str, np.ndarray]]],
) -> Dict[str, np.ndarray]:
    """
    Merge list[dict[str, np.ndarray]] -> dict[str, np.ndarray] by concatenation axis 0.
    """
    merged: Dict[str, np.ndarray] = {}
    payloads = [p for p in payloads if p]
    if not payloads:
        return merged

    keys = set()
    for p in payloads:
        keys.update(p.keys())

    for k in keys:
        chunks = [
            p[k]
            for p in payloads
            if (k in p) and (p[k] is not None) and (len(p[k]) > 0)
        ]
        if not chunks:
            continue
        merged[k] = np.concatenate(chunks, axis=0)
    return merged


def _stable_sort_and_dedupe(payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Sort payload arrays by dataset-local 'idx' when present (correct for subsets),
    else fall back to 'orig_idx'. Then dedupe (DistributedSampler may pad/duplicate).
    """
    if not payload:
        return payload

    key = "idx" if "idx" in payload else ("orig_idx" if "orig_idx" in payload else None)
    if key is None:
        return payload

    k = payload[key].astype(np.int64)
    if k.size == 0:
        return payload

    order = np.argsort(k, kind="mergesort")
    k_sorted = k[order]
    _, uniq_pos = np.unique(k_sorted, return_index=True)
    keep = order[np.sort(uniq_pos)]

    out: Dict[str, np.ndarray] = {}
    for name, v in payload.items():
        if isinstance(v, np.ndarray) and v.shape[0] == k.shape[0]:
            out[name] = v[keep]
        else:
            out[name] = v
    return out


def _gather_predict_payload(
    preds, keys: Tuple[str, ...]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Packs requested keys from local preds, gathers to rank0 under DDP, merges, and
    stable-sorts/dedupes by orig_idx (if present).

    Returns:
    - payload dict on rank 0
    - None on non-rank0 in DDP
    """
    local = _pack_keys_from_preds(preds, keys)

    gathered = _gather_object_to_rank0(local)
    if gathered is None:
        return None

    merged = _merge_packed_payloads(gathered)
    merged = _stable_sort_and_dedupe(merged)
    return merged


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression lightning_modules
    """

    @torch.no_grad()
    def predict_params(
        self, model: pl.LightningModule, dataloader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns context-specific regression models
        - beta (numpy.ndarray): (n, y_dim, x_dim)
        - mu (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)

        payload = _gather_predict_payload(preds, keys=("idx", "orig_idx", "betas", "mus"))
        if payload is None:
            return None, None

        if "betas" not in payload or "mus" not in payload:
            raise RuntimeError(
                "predict_params: predict_step must return 'betas' and 'mus' (and ideally 'orig_idx')."
            )

        return payload["betas"], payload["mus"]

    @torch.no_grad()
    def predict_y(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific predictions of the response Y
        - y_hat (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)

        payload = _gather_predict_payload(
            preds, keys=("idx", "orig_idx", "contexts", "predictors", "betas", "mus")
        )

        if payload is None:
            return None

        if "betas" not in payload or "mus" not in payload:
            raise RuntimeError("predict_y: predict_step must return 'betas' and 'mus'.")

        betas = torch.as_tensor(payload["betas"])
        mus = torch.as_tensor(payload["mus"])

        if ("contexts" in payload) and ("predictors" in payload):
            C = torch.as_tensor(payload["contexts"])
            X = torch.as_tensor(payload["predictors"])
        else:
            ds = getattr(dataloader, "dataset", None)
            if ds is None:
                raise RuntimeError(
                    "predict_y: dataloader has no .dataset; cannot reconstruct C/X."
                )

            idx_np = payload["idx"].astype(np.int64)
            idx_t = torch.as_tensor(idx_np, dtype=torch.long)

            if hasattr(ds, "dataset") and hasattr(ds, "indices"):
                base = ds.dataset
                if not (hasattr(base, "C") and hasattr(base, "X")):
                    raise RuntimeError("predict_y: Subset base dataset must expose .C and .X.")
                base_pos = np.asarray(ds.indices, dtype=np.int64)[idx_np]
                base_pos_t = torch.as_tensor(base_pos, dtype=torch.long)
                C = base.C[base_pos_t]
                X = base.X[base_pos_t]
            else:
                if not (hasattr(ds, "C") and hasattr(ds, "X")):
                    raise RuntimeError(
                        "predict_y: dataset must expose .C and .X tensors for Option A prediction."
                    )
                C = ds.C[idx_t]
                X = ds.X[idx_t]

            if torch.is_tensor(C):
                C = C.to(dtype=betas.dtype)
            else:
                C = torch.as_tensor(C, dtype=betas.dtype)

            if torch.is_tensor(X):
                X = X.to(dtype=betas.dtype)
            else:
                X = torch.as_tensor(X, dtype=betas.dtype)

        with torch.no_grad():
            yhat = model._predict_y(C, X, betas, mus).detach().cpu().numpy()

        return yhat


class CorrelationTrainer(RegressionTrainer):
    """
    Trains the contextualized.regression correlation lightning_modules
    """

    @torch.no_grad()
    def predict_correlation(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific correlation networks containing Pearson's correlation coefficient
        - correlation (numpy.ndarray): (n, x_dim, x_dim)
        """
        preds = super().predict(model, dataloader)
        preds_flat = _flatten_pl_predict_output(preds)

        if preds_flat and ("correlations" in preds_flat[0]):
            payload = _gather_predict_payload(preds, keys=("orig_idx", "correlations"))
            if payload is None:
                return None
            if "correlations" not in payload:
                raise RuntimeError(
                    "predict_correlation: predict_step returned no 'correlations'."
                )
            return payload["correlations"]

        betas, _ = self.predict_params(model, dataloader)
        if betas is None:
            return None

        signs = np.sign(betas)
        signs[signs != np.transpose(signs, (0, 2, 1))] = 0
        correlations = signs * np.sqrt(np.abs(betas * np.transpose(betas, (0, 2, 1))))
        return correlations


class MarkovTrainer(CorrelationTrainer):
    """
    Trains the contextualized.regression markov graph lightning_modules
    """

    @torch.no_grad()
    def predict_precision(self, model: pl.LightningModule, dataloader) -> np.ndarray:
        """
        Returns context-specific precision matrix under a Gaussian graphical model
        Assuming all diagonal precisions are equal and constant over context,
        this is equivalent to the negative of the multivariate regression coefficient.
        - precision (numpy.ndarray): (n, x_dim, x_dim)
        """
        return -super().predict_correlation(model, dataloader)


def choose_lightning_environment() -> LightningEnvironment:
    """
    Returns the Lightning environment plugin used for single-process runs.
    """
    return LightningEnvironment()


def make_trainer_with_env(trainer_cls, **trainer_kwargs):
    """
    Factory that respects caller-provided `devices` and `strategy`.
    Does not inject LightningEnvironment when torchrun is managing processes.
    """
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if "plugins" not in trainer_kwargs and world_size == 1:
        env = choose_lightning_environment()
        trainer_kwargs["plugins"] = [env]

    return trainer_cls(**trainer_kwargs)
