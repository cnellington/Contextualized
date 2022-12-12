"""
Losses used in regression.
"""

import torch


def MSE(Y_true, Y_pred):
    """
    Returns
    - MSE (scalar torch.tensor): the mean squared-error or L2-error
        of multivariate and univariate regression problems. Default
        loss for contextualized.regression models.

    MV/UV: Multivariate/Univariate
    MT/ST: Multi-task/Single-task

    MV ST: beta (y_dim, x_dim),    mu (y_dim, 1),        x (y_dim, x_dim),    y (y_dim, 1)
    MV MT: beta (x_dim,),          mu (1,),              x (x_dim,),          y (1,)
    UV ST: beta (y_dim, x_dim, 1), mu (y_dim, x_dim, 1), x (y_dim, x_dim, 1), y (y_dim, x_dim, 1)
    UV MT: beta (1,),              mu (1,),              x (1,),              y (1,)
    """
    residual = Y_true - Y_pred
    return residual.pow(2).mean()


def BCELoss(Y_true, Y_pred):
    loss = -(
        Y_true * torch.log(Y_pred + 1e-8) + (1 - Y_true) * torch.log(1 - Y_pred + 1e-8)
    )
    return loss.mean()
