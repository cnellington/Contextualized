import torch


def DAG_loss(w, alpha, rho):
    """
    DAG loss on batched networks W using the
    NOTEARS matrix exponential trace
    """
    m = torch.linalg.matrix_exp(w * w)
    h = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - w.shape[-1]
    return alpha * h + 0.5 * rho * h * h


# Lasso (L1) regularization term
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)


# Mean squared error of y_true vs. y_pred
mse_loss = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()
