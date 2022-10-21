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
