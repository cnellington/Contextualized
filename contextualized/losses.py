

def MSE(beta, mu, x, y, link_fn=lambda x: x):
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
    y_hat = link_fn((beta * x).sum(axis=-1).unsqueeze(-1) + mu)
    residual = y_hat - y
    return residual.pow(2).mean()
