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
    torch.save(model, open(path, "wb"), pickle_module=dill)


def load(path):
    """
    Loads model from path.
    :param path: 

    """
    model = torch.load(open(path, "rb"), pickle_module=dill)
    return model
