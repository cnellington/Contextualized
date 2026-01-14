import os
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
import numpy as np

class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        torch.save(prediction, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}_{dataloader_idx}_{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))


import numpy as np
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter


class MmapEdgeWriter(BasePredictionWriter):
    """
    Store β and μ for every edge in one memory‑mapped array:
        shape = (n_samples, x_dim, y_dim, 2)
        last dim: 0 → beta, 1 → mu
    One file per rank:  <mmap_dir>/edges_rank{r}.dat
    """
    def __init__(
        self,
        mmap_dir: str,
        n_samples: int,
        x_dim: int,
        y_dim: int,
        dtype=np.float32,
        write_interval="batch",
    ):
        super().__init__(write_interval=write_interval)
        self.mmap_dir = Path(mmap_dir)
        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.dtype = dtype

        # Will be created in on_predict_start
        self.arr = None

    # ---------- life‑cycle hooks ------------------------------------------------
    def on_predict_start(self, trainer, pl_module):
        """Called *once* per rank, before the first batch."""
        self.mmap_dir.mkdir(parents=True, exist_ok=True)
        fname = self.mmap_dir / f"edges_rank{trainer.global_rank}.dat"

        if trainer.global_rank == 0 and fname.exists():
            fname.unlink()                      # replace stale file from a previous run
        trainer.strategy.barrier()              # make sure other ranks wait

        self.arr = np.memmap(
            fname,
            mode="w+",
            shape=(self.n_samples, self.x_dim, self.y_dim, 2),
            dtype=self.dtype,
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,       # the dict you returned in predict_step
        batch_indices,
        batch,            # original batch
        batch_idx,
        dataloader_idx,
    ):
        beta  = prediction["betas"].detach().cpu().numpy()
        mu  = prediction["mus"].detach().cpu().numpy()
        n  = prediction["sample_idx"].cpu().numpy()
        xi = prediction["predictor_idx"].cpu().numpy()
        yi = prediction["outcome_idx"].cpu().numpy()

        # vectorised write, no Python for‑loop
        self.arr[n, yi, xi, 0] = beta
        self.arr[n, yi, xi, 1] = mu

