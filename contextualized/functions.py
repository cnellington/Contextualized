"""Utility torch mathematical functions which are used for many modules.
"""

import torch
import torch.nn.functional as F

zero_vector = lambda x, *args: torch.zeros((len(x), 1))
zero = lambda x: torch.zeros_like(x)
identity = lambda x: x
linear = lambda x, slope, intercept: x * slope + intercept
logistic = lambda x, slope, intercept: 1 / (1 + torch.exp(-x * slope - intercept))
linear_link = lambda x, slope, intercept: x * slope + intercept
identity_link = lambda x: x
softmax_link = lambda x, slope, intercept: F.softmax(x * slope + intercept, dim=1)


def make_fn(base_fn, **params):
    """
    Makes a single-parameter function from a base function class and a fixed
    set of extra parameters.
    :param base_fn:
    :param **params:

    """
    return lambda x: base_fn(x, **params)


def linear_constructor(slope=1, intercept=0):
    """
    Creates a single-parameter linear function with slope m and offset b.
    :param slope:  (Default value = 1)
    :param intercept:  (Default value = 0)

    """
    return make_fn(linear, slope=slope, intercept=intercept)


def logistic_constructor(slope=1, intercept=0):
    """
    Creates a single-parameter logistic function with slope m and offset b.
    :param slope:  (Default value = 1)
    :param intercept:  (Default value = 0)

    """
    return make_fn(logistic, slope=slope, intercept=intercept)


def identity_link_constructor():
    """
    Creates a single-parameter identity function.
    """
    return make_fn(identity_link)


def linear_link_constructor(slope=1, intercept=0):
    """
    Creates a single-parameter linear link function with slope m and offset b.
    :param slope:  (Default value = 1)
    :param intercept:  (Default value = 0)

    """
    return make_fn(linear_link, slope=slope, intercept=intercept)


def softmax_link_constructor(slope=1, intercept=0):
    """
    Creates a single-parameter softmax link function with slope m and offset b.
    :param slope:  (Default value = 1)
    :param intercept:  (Default value = 0)

    """
    return make_fn(softmax_link, slope=slope, intercept=intercept)


LINK_FUNCTIONS = {
    "identity": linear_link_constructor(),
    "logistic": logistic_constructor(),
    "softmax": softmax_link_constructor(),
}
