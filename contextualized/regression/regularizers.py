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
    return l1_ratio*l1_reg(alpha, mu_ratio) + (1-l1_ratio)*l2_reg(alpha, mu_ratio)
