"""
Miscellaneous utility functions.
"""

import numpy as np


def convert_to_one_hot(col):
    """

    :param col: np array with observations

    returns col converted to one-hot values, and list of one-hot values.

    """
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals
