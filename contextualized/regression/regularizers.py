import torch


def no_reg():
    return lambda beta, mu: 0.


def l1_reg(alpha, mu_ratio=0.5):
    return lambda beta, mu: alpha*(
        mu_ratio*torch.norm(mu, p=1) + (1-mu_ratio)*torch.norm(beta, p=1)).mean()


def l2_reg(alpha, mu_ratio=0.5):
    return lambda beta, mu: alpha*(
        mu_ratio*torch.norm(mu, p=2) + (1-mu_ratio)*torch.norm(beta, p=2)).mean()


def l1_l2_reg(alpha, l1_ratio=0.5, mu_ratio=0.5):
    return lambda beta, mu: alpha*(
        l1_ratio*(mu_ratio*torch.norm(mu, p=1) + (1-mu_ratio)*torch.norm(beta, p=1)) +
        (1-l1_ratio)*(mu_ratio*torch.norm(mu, p=2) + (1-mu_ratio)*torch.norm(beta, p=2))).mean()


def correlation_reg(alpha):
    # Regularize self-regression coefficients to 1
    def reg(beta, mu, T):
        diag_beta = torch.diagonal(beta, dim1=1, dim2=2)
        return alpha * torch.norm(diag_beta - torch.ones_like(diag_beta), p=2)
    return reg


def tasksplit_self_reg(alpha):
    # Regularize taskspit self-regression coefficients to 1
    def reg(beta, mu, T):
        tsize = T.shape[-1] // 2
        T_diff = torch.eq(T[:,:tsize], T[:,tsize:])
        diag_idx = torch.logical_and(T_diff[:,0], T_diff[:,1]).unsqueeze(-1)
        diag_beta = beta[diag_idx]
        return alpha * torch.norm(diag_beta - torch.ones_like(diag_beta), p=2)
    return reg