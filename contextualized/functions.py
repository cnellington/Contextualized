"""Utility torch mathematical functions which are used for many modules.
"""

import torch
import torch.nn.functional as F

zero_vector = lambda x, *args: torch.zeros((len(x), 1))
zero = lambda x: torch.zeros_like(x)
identity = lambda x: x
linear = lambda x, m, b: x * m + b
logistic = lambda x, m, b: 1 / (1 + torch.exp(-x * m - b))
linear_link = lambda x, m, b: x * m + b
identity_link = lambda x: x
softmax_link = lambda x, m, b: F.softmax(x * m + b, dim=1)


def make_fn(base_fn, **params):
    """
    Makes a single-parameter function from a base function class and a fixed
    set of extra parameters.
    :param base_fn:
    :param **params:

    """
    return lambda x: base_fn(x, **params)


def linear_constructor(m=1, b=0):
    """
    Creates a single-parameter linear function with slope m and offset b.
    :param m:  (Default value = 1)
    :param b:  (Default value = 0)

    """
    return make_fn(linear, m=m, b=b)


def logistic_constructor(m=1, b=0):
    """
    Creates a single-parameter logistic function with slope m and offset b.
    :param m:  (Default value = 1)
    :param b:  (Default value = 0)

    """
    return make_fn(logistic, m=m, b=b)


def identity_link_constructor():
    """
    Creates a single-parameter identity function.
    """
    return make_fn(identity_link)


def linear_link_constructor(m=1, b=0):
    """
    Creates a single-parameter linear link function with slope m and offset b.
    :param m:  (Default value = 1)
    :param b:  (Default value = 0)

    """
    return make_fn(linear_link, m=m, b=b)


def softmax_link_constructor(m=1, b=0):
    """
    Creates a single-parameter softmax link function with slope m and offset b.
    :param m:  (Default value = 1)
    :param b:  (Default value = 0)

    """
    return make_fn(softmax_link, m=m, b=b)


LINK_FUNCTIONS = {
    "identity": linear_link_constructor(),
    "logistic": logistic_constructor(),
    "softmax": softmax_link_constructor(),
}
