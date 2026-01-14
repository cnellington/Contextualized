"""
Utility functions and simple helper predictors used across the library,
including saving/loading of contextualized models.
"""

from __future__ import annotations

import torch


def save(model, path: str) -> None:
    """Save a model object to disk."""
    with open(path, "wb") as out_file:
        torch.save(model, out_file)


def load(path: str):
    """Load a model object from disk."""
    with open(path, "rb") as in_file:
        # Newer torch supports weights_only; older versions do not.
        try:
            return torch.load(in_file, weights_only=False)
        except TypeError:
            return torch.load(in_file)


class DummyParamPredictor:
    """Predicts parameters as all zeros (for unit tests / baselines)."""

    def __init__(self, beta_dim, mu_dim):
        self.beta_dim = beta_dim
        self.mu_dim = mu_dim

    def predict_params(self, *args):
        n = len(args[0])
        return torch.zeros((n, *self.beta_dim)), torch.zeros((n, *self.mu_dim))


class DummyYPredictor:
    """Predicts Y values as all zeros (for unit tests / baselines)."""

    def __init__(self, y_dim):
        self.y_dim = y_dim

    def predict_y(self, *args):
        n = len(args[0])
        return torch.zeros((n, *self.y_dim))
