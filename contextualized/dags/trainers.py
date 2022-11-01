"""
PyTorch-Lightning trainers used for Contextualized graphs.
"""
import torch
import pytorch_lightning as pl


class GraphTrainer(pl.Trainer):
    """
    Trains the contextualized.graphs lightning_modules
    """

    def predict_params(self, model, dataloader, **kwargs):
        """
        Predict graph parameters with model-specific kwargs
        """
        preds = torch.cat(super().predict(model, dataloader), dim=0).numpy()
        return model._format_params(preds, **kwargs)
