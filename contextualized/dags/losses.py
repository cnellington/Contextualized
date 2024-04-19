import numpy as np
import torch
from contextualized.dags.graph_utils import dag_pred, dag_pred_with_factors


def dag_loss_dagma_indiv(w, s=1):
    M = s * torch.eye(w.shape[-1]) - w * w
    return w.shape[-1] * np.log(s) - torch.slogdet(M)[1]


def dag_loss_dagma(W, s=1, alpha=0.0, **kwargs):
    """DAG loss on batched networks W using the
    DAGMA log-determinant
    """
    sample_losses = torch.Tensor([dag_loss_dagma_indiv(w, s) for w in W])
    return alpha * torch.mean(sample_losses)


def dag_loss_poly_indiv(w):
    d = w.shape[-1]
    return torch.trace(torch.eye(d) + (1 / d) * torch.matmul(w, w)) - d


def dag_loss_poly(W, **kwargs):
    """DAG loss on batched networks W using the
    h_poly form: h_poly(W) = Tr((I + 1/d(W*W)^d) - d
    """
    return torch.mean(torch.Tensor([dag_loss_poly_indiv(w) for w in W]))


def dag_loss_notears(W, alpha=0.0, rho=0.0, **kwargs):
    """
    DAG loss on batched networks W using the
    NOTEARS matrix exponential trace
    """
    m = torch.linalg.matrix_exp(W * W)
    h = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - W.shape[-1]
    return torch.mean(alpha * h + 0.5 * rho * h * h)


# Lasso (L1) regularization term
l1_loss = lambda w, l1: l1 * torch.sum(torch.abs(w))


# Mean squared error of y_true vs. y_pred
mse_loss = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()


def linear_sem_loss_with_factors(x_true, w_pred, factor_mat):
    """Computes MSE loss between true x and predicted linear structural equation model,
    for torch tensors. Works on batches only.

    Args:
        x_true (torch.FloatTensor): Vector of True features x
        w_pred (torch.FloatTensor): Predicted linear structural equation model
        factor_mat (torch.FloatTensor): Factor matrix, size latent factors x x.shape[-1]

    Returns:
        torch.tensor: MSE loss for data features and predicted SEM.
    """
    x_prime = dag_pred_with_factors(x_true, w_pred, factor_mat)
    return (0.5 / x_true.shape[0]) * torch.square(torch.linalg.norm(x_true - x_prime))


def linear_sem_loss(x_true, w_pred):
    """Computes MSE loss between true x and predicted linear structural equation model,
    for torch tensors. Works on batches only.

    Args:
        x_true (torch.FloatTensor): Vector of True features x
        w_pred (torch.FloatTensor): Predicted linear structural equation model

    Returns:
        torch.tensor: MSE loss for data features and predicted SEM.
    """
    x_prime = dag_pred(x_true, w_pred)
    return (0.5 / x_true.shape[0]) * torch.square(torch.linalg.norm(x_true - x_prime))
