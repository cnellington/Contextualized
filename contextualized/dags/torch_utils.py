import torch


def DAG_loss(w_pred, alpha, rho):
    """Computes the DAG loss via torch functions, for torch tensors.
    Used by torch NOTMAD for NOTEARS loss, archetype loss, and DynamicAlphaRho callback.
    Works on both batches of predicted networks and single networks.

    Args:
        w_pred (torch.FloatTensor): Predicted DAG for features
        alpha (float): Alpha DAG loss param
        rho (float): Rho DAG loss param

    Returns:
        torch.tensor: DAG loss for given network or batch of networks
    """
    if len(w_pred.shape) == 2:  # archetype loss
        d = w_pred.size()[0]
        m = torch.linalg.matrix_exp(w_pred * w_pred)
        h = torch.trace(m) - d
        loss = alpha * h + 0.5 * rho * h * h
    else:  # batch loss
        d = w_pred.size()[1]
        m = torch.linalg.matrix_exp(w_pred * w_pred)
        h = torch.sum(torch.diagonal(m, dim1=-2, dim2=-1), axis=1) - d  # batch trace
        loss = torch.mean(alpha * h + 0.5 * rho * h * h)

    return loss

def DAG_loss_np(w, alpha, rho):
    """
    Computes DAG loss on w which is np array.
    """
    d = w.shape[-1]
    m = torch.linalg.matrix_exp(w * w)
    h = torch.trace(m) - d
    return alpha * h + 0.5 * rho * h * h

dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)
mse_loss = lambda y_true, y_pred: ((y_true - y_pred)**2).mean()



def dag_mse(x_true, w_pred):
    """Computes MSE loss between true x and predicted network, for torch tensors.
    Works on batches only.

    Args:
        w_pred (torch.FloatTensor): Predicted DAG for features
        x_true (torch.FloatTensor): Vector of True features x

    Returns:
        torch.tensor: MSE loss for data features and predicted network.
    """
    x_prime = torch.matmul(x_true.unsqueeze(1), w_pred).squeeze(
        1
    )  # use squeezing to match dims of x_true and w_pred
    loss = (0.5 / x_true.shape[0]) * torch.square(torch.linalg.norm(x_true - x_prime))
    return loss


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
    mse = dag_mse(x_true, w_pred)
    loss = mse + l1_lambda * torch.norm(w_pred, p=1) + DAG_loss(w_pred, alpha, rho)
    return loss
