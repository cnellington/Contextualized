"""
Torch regularizers used for regression.
"""

import torch
from functools import partial


def no_reg_fn(beta, mu):
    return 0.0


def no_reg():
    """
    Function that returns an empty regularizer.
    """
    return no_reg_fn


def l1_reg_fn(alpha, mu_ratio, beta, mu):
    return (
        alpha
        * (
            mu_ratio * torch.norm(mu, p=1) + (1 - mu_ratio) * torch.norm(beta, p=1)
        ).mean()
    )


def l1_reg(alpha, mu_ratio=0.5):
    """

    :param alpha:
    :param mu_ratio:  (Default value = 0.5)

    """
    return partial(l1_reg_fn, alpha, mu_ratio)


def l2_reg_fn(alpha, mu_ratio, beta, mu):
    return (
        alpha
        * (
            mu_ratio * torch.norm(mu, p=2) + (1 - mu_ratio) * torch.norm(beta, p=2)
        ).mean()
    )


def l2_reg(alpha, mu_ratio=0.5):
    """

    :param alpha:
    :param mu_ratio:  (Default value = 0.5)

    """
    return partial(l2_reg_fn, alpha, mu_ratio)


def l1_l2_reg_fn(alpha, l1_ratio, mu_ratio, beta, mu):
    return (
        alpha
        * (
            l1_ratio
            * (mu_ratio * torch.norm(mu, p=1) + (1 - mu_ratio) * torch.norm(beta, p=1))
            + (1 - l1_ratio)
            * (mu_ratio * torch.norm(mu, p=2) + (1 - mu_ratio) * torch.norm(beta, p=2))
        ).mean()
    )


def l1_l2_reg(alpha, l1_ratio=0.5, mu_ratio=0.5):
    """

    :param alpha:
    :param l1_ratio:  (Default value = 0.5)
    :param mu_ratio:  (Default value = 0.5)

    """
    return partial(l1_l2_reg_fn, alpha, l1_ratio, mu_ratio)


REGULARIZERS = {"none": no_reg(), "l1": l1_reg, "l2": l2_reg, "l1_l2": l1_l2_reg}
