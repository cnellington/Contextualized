# HPC and DDP Usage Guide

This package supports single GPU, multi-GPU, and HPC clusters using PyTorch Lightning. The primary goal is consistent behavior across CPU, GPU, and DDP environments. The secondary goal is correct ordering of predictions under DDP.

## Core Principles

**Map-style datasets**: Lightning can shard data with DistributedSampler when using `torch.utils.data.Dataset`.

**LightningDataModule pattern**: Builds datasets from user arrays and manages train/val/test splits consistently.

**Stable prediction ordering**: Return prediction payloads that include stable indices, then gather and reorder on rank 0.

## Dataset and Batch Structure

Datasets are map-style (`torch.utils.data.Dataset`). Each `__getitem__` returns a dict with standard keys: `contexts`, `predictors`, `outcomes`, plus indexing keys for DDP-safe prediction assembly: `idx` (dataset local index), `orig_idx` (stable original row ID). For multitask variants, additional keys include `sample_idx`, `outcome_idx`, `predictor_idx`. DDP sharding can change sample order and pad the last batch, so these indices enable correct reconstruction.

## DataModule Usage

The DataModule converts numpy or pandas arrays into tensors and slices by split indices. It passes `orig_idx` into each dataset so every sample reports its original row ID.

**Split configuration**: Provide `train_idx`, `val_idx`, `test_idx`, `predict_idx` directly, or provide a `splitter(C, X, Y)` callable that returns `(train_idx, val_idx, test_idx)`. If `predict_idx` is not provided, it defaults to `test_idx` when present, otherwise defaults to the full range.

**Example instantiation**:
```python
from contextualized.regression.datamodules import ContextualizedRegressionDataModule

dm = ContextualizedRegressionDataModule(
    C=C, X=X, Y=Y,
    task_type="singletask_multivariate",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    predict_idx=predict_idx,
    train_batch_size=32,
    val_batch_size=64,
    test_batch_size=64,
    predict_batch_size=64,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    shuffle_train=True,
    shuffle_eval=False,
)

trainer.fit(model, datamodule=dm)
preds = trainer.predict(model, datamodule=dm)
```

If calling loaders manually, call `dm.setup(stage="predict")` before retrieving the dataloader.

## DDP Prediction Mechanics

Prediction assembly occurs only on rank 0. Non-rank-0 processes return `None`. This prevents duplicated outputs and keeps the API stable under DDP.

**Predict step payload**: Each `LightningModule.predict_step` returns a dict containing indices (`idx`, `orig_idx`, and optional task indices), batch content when needed (`contexts`, `predictors`), and model outputs (`betas`, `mus`, and sometimes `correlations`). Tensors are detached and moved to CPU inside `predict_step` to keep GPU memory stable.

**Gather and reorder process**: The trainer helper packs requested keys from local predict outputs, gathers to rank 0, merges payloads by concatenating on axis 0, stable sorts by `idx` when present (otherwise `orig_idx`), and deduplicates padded samples from `DistributedSampler`. If using `dist.gather_object`, ensure the collective backend supports object gathers (commonly Gloo). If your default process group is NCCL, use a Gloo group for object gather or switch to tensor all-gather.

## Training Configuration

**Single node, single GPU**: Use `Trainer(accelerator="gpu", devices=1, strategy="auto")`. Recommended data settings: `pin_memory=True`, `num_workers` tuned to CPU count and batch size, `persistent_workers=True` if `num_workers > 0`.

**Single node, multi-GPU**: Two supported patterns exist.

Pattern A (Lightning spawns processes): Use a normal Python launch and let Lightning spawn processes. Set `devices` to GPU count and `strategy="ddp"` or `DDPStrategy(...)`. Example for 2 GPUs: `Trainer(accelerator="gpu", devices=2, strategy="ddp")`.

Pattern B (torchrun launch, recommended for clusters): Use torchrun to launch one process per GPU. Each process should use `devices=1` (or omit devices) and set `strategy="ddp"` or `DDPStrategy(...)`. Example for 2 GPUs on one node:
```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 your_script.py
```

Recommended environment variables for NCCL:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
```

Set `NCCL_SOCKET_IFNAME` to your actual NIC. If you see hangs, confirm `WORLD_SIZE`, `RANK`, `LOCAL_RANK` are correct.

## Sklearn-Style Wrappers

The sklearn-style API uses the same DataModule path when available, keeping the public surface area stable.
```python
from contextualized.easy.wrappers import ContextualizedRegressor

m = ContextualizedRegressor(
    num_archetypes=8,
    encoder_type="mlp",
    encoder_kwargs={"width": 256, "layers": 2, "link_fn": "identity"},
    trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1,
        "max_epochs": 50,
    },
)

m.fit(C_train, X_train, Y_train, val_split=0.2)
yhat = m.predict(C_test, X_test)
betas, mus = m.predict_params(C_test)
```

**DDP behavior for wrappers**: Under DDP, rank 0 returns arrays and non-rank-0 returns `None`. If running under torchrun, only rank 0 should consume outputs. Do not stack outputs across ranks.

## Batch Sizing for Strong Scaling

Strong scaling means fixed global batch size. Per-GPU batch is `global_batch_size / world_size`. Set one global batch size and keep it fixed. As you add GPUs, let the per-GPU batch shrink. This reduces OOM risk when scaling up GPU count.

If you hit OOM: reduce global batch first, then reduce model width or layers, then reduce DataLoader workers if CPU memory becomes the bottleneck.

## Data Movement and Pinned Memory

Pinned CPU buffers improve host-to-device transfer speed. Recommended for GPU training: `pin_memory=True` in DataLoader, use pinned host tensors in synthetic or streaming benchmarks. For real datasets, use normal CPU tensors and let DataLoader pin memory.

## Common Pitfalls

**Wrong batch dict keys**: Models expect batch dict keys `contexts`, `predictors`, `outcomes`. If using custom datasets, match these names.

**Device mismatch under torchrun**: Under torchrun, processes already map to devices. Use `devices=1` per process (or omit devices), and each process uses `LOCAL_RANK` as its CUDA device.

**Dropping samples during eval**: Do not drop samples for validation, test, or predict. It breaks ordering assumptions. Use `drop_last=False` for eval loaders.

**Expecting prediction output on every rank**: Prediction helpers are rank-0-only by design. Non-rank-0 returns `None`.

## Minimal DDP Launch Recipe

Single node, 4 GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 train_script.py
```

Trainer settings that work well: `strategy=DDPStrategy(find_unused_parameters=False, broadcast_buffers=False)`, mixed precision on GPU if stable for your model, logging sync only on epoch metrics (not per step).

## Benchmark Pattern for Scaling

A good scaling benchmark uses fixed global batch size, uses a warmup window then measures steady state, uses already batched inputs to reduce DataLoader overhead, uses pinned CPU memory when measuring host-to-device transfer. Loss computation should be simple and shape-safe. Keep it in the benchmark harness. Avoid clever reshapes that depend on internal model conventions.

## Summary

HPC readiness comes from three components: map-style datasets and a DataModule so Lightning can shard correctly, prediction payloads that include stable indices and are CPU-friendly, rank-0 gather and reorder so predictions match user-expected order. Following these patterns ensures multi-GPU training and prediction are stable and repeatable.