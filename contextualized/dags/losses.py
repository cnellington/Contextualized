import torch
from contextualized.dags.graph_utils import dag_pred


def dag_loss(w, alpha, rho):
    """
    DAG loss on batched networks W using the
    NOTEARS matrix exponential trace
    """
    m = torch.linalg.matrix_exp(w * w)
    h = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - w.shape[-1]
    return torch.mean(alpha * h + 0.5 * rho * h * h)


# Lasso (L1) regularization term
l1_loss = lambda w, l1: l1 * torch.sum(torch.abs(w))


# Mean squared error of y_true vs. y_pred
mse_loss = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()


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


def NOTEARS_loss(x_true, w_pred, l1_lambda, alpha, rho):
    """Computes NOTEARS loss between true x and predicted network, for torch tensors.
    Works on batches only.

    Args:
        w_pred (torch.FloatTensor): _description_
        alpha (float): Alpha DAG loss param
        rho (float): Rho DAG loss param

    Returns:
        torch.tensor: NOTEARS loss for data features and predicted network (batches only).
    """
    mse_term = linear_sem_loss(x_true, w_pred)
    l1_term = l1_loss(w_pred, l1_lambda)
    dag_term = dag_loss(w_pred, alpha, rho)
    notears = mse_term + l1_term + dag_term
    return notears
