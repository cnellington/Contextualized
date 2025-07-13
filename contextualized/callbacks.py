import os
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

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
