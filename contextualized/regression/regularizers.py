"""
Torch regularizers used for regression.
"""
import torch


def no_reg():
    """
    Function that returns an empty regularizer.
    """
    return lambda beta, mu: 0.0


def l1_reg(alpha, mu_ratio=0.5):
    """

    :param alpha:
    :param mu_ratio:  (Default value = 0.5)

    """
    return (
        lambda beta, mu: alpha
        * (
            mu_ratio * torch.norm(mu, p=1) + (1 - mu_ratio) * torch.norm(beta, p=1)
        ).mean()
    )


def l2_reg(alpha, mu_ratio=0.5):
    """

    :param alpha:
    :param mu_ratio:  (Default value = 0.5)

    """
    return (
        lambda beta, mu: alpha
        * (
            mu_ratio * torch.norm(mu, p=2) + (1 - mu_ratio) * torch.norm(beta, p=2)
        ).mean()
    )


def l1_l2_reg(alpha, l1_ratio=0.5, mu_ratio=0.5):
    """

    :param alpha:
    :param l1_ratio:  (Default value = 0.5)
    :param mu_ratio:  (Default value = 0.5)

    """
    return (
        lambda beta, mu: alpha
        * (
            l1_ratio
            * (mu_ratio * torch.norm(mu, p=1) + (1 - mu_ratio) * torch.norm(beta, p=1))
            + (1 - l1_ratio)
            * (mu_ratio * torch.norm(mu, p=2) + (1 - mu_ratio) * torch.norm(beta, p=2))
        ).mean()
    )


REGULARIZERS = {"none": no_reg(), "l1": l1_reg, "l2": l2_reg, "l1_l2": l1_l2_reg}
