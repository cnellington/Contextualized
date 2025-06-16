"""
Utility functions, including saving/loading of contextualized models.
"""

import torch
import warnings


def save(model, path):
    """
    Saves model to path.
    :param model:
    :param path:

    """
    with open(path, "wb") as out_file:
        torch.save(model, out_file)


def load(path):
    """
    Loads model from path.
    :param path:

    """
    with open(path, "rb") as in_file:
        model = torch.load(in_file, weights_only=False)
    return model


class DummyParamPredictor:
    """
    Predicts Parameters as all zeros.
    """

    def __init__(self, beta_dim, mu_dim):
        self.beta_dim = beta_dim
        self.mu_dim = mu_dim

    def predict_params(self, *args):
        """

        :param *args:

        """
        n = len(args[0])
        return torch.zeros((n, *self.beta_dim)), torch.zeros((n, *self.mu_dim))


class DummyYPredictor:
    """
    Predicts Ys as all zeros.
    """

    def __init__(self, y_dim):
        self.y_dim = y_dim

    def predict_y(self, *args):
        """

        :param *args:

        """
        n = len(args[0])
        return torch.zeros((n, *self.y_dim))


def check_kwargs(kwargs, allowed_keys, context=""):
    """
    Check for unexpected keyword arguments and issue warnings.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments to validate.
    allowed_keys : set or list
        The expected valid keyword argument names.
    context : str
        Optional name of the function or class calling this check.

    Raises
    ------
    UserWarning for any unknown key in kwargs.
    """
    allowed_keys = set(allowed_keys)
    for key in kwargs:
        if key not in allowed_keys:
            warnings.warn(
                f"[{context}] Unexpected keyword argument: '{key}'",
                category=UserWarning,
                stacklevel=2
            )