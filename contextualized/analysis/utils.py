"""
Miscellaneous utility functions.
"""

from typing import *

import numpy as np


def convert_to_one_hot(col: Collection[Any]) -> Tuple[np.ndarray, List[Any]]:
    """
    Converts a categorical variable to a one-hot vector.

    Args:
        col (Collection[Any]): The categorical variable.

    Returns:
        Tuple[np.ndarray, List[Any]]: The one-hot vector and the possible values.
    """
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals
