"""
Utility functions, including saving/loading of contextualized models.
"""

import torch
import dill


def save(model, path):
    """
    Saves model to path.
    :param model:
    :param path:

    """
    with open(path, "wb") as out_file:
        torch.save(model, out_file, pickle_module=dill)


def load(path):
    """
    Loads model from path.
    :param path:

    """
    with open(path, "rb") as in_file:
        model = torch.load(in_file, pickle_module=dill)
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
