import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def loss(beta, xi, xj):
    return (beta.unsqueeze(-1) * xi - xj).pow(2)


class ContextualCorrelator(nn.Module):
    """
    rho(c) = beta(c) * beta'(c)
    beta(c) = sigma(A @ f(c) + b)
    
    beta_{a_i, b_j} = sigma(A(t_i, t_j) @ f(c) + b)
    f(c) = sigma(dense(c))
    A(t_i, t_j) = <g(t_i, t_j), A_{1..K}>
    g(t_i, t_j) = softmax(dense(t_i, t_j))
    """

    def __init__(self, context_shape, task_shape, num_archetypes=2, final_dense_size=10):
        super(ContextualCorrelator, self).__init__()
        self.context_encoder_in_shape = (context_shape[-1], 1)
        self.context_encoder_out_shape = (final_dense_size,)
        task_encoder_in_shape = (task_shape[-1] * 2, 1)
        task_encoder_out_shape = (final_dense_size,)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_shape[-1], 25), 
            nn.ReLU(),
            nn.Linear(25, final_dense_size), 
            nn.ReLU(), 
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_shape[-1], 25), 
            nn.ReLU(),
            nn.Linear(25, num_archetypes),
            nn.Softmax(dim=1), 
        )
        init_archetypes = torch.rand(num_archetypes, final_dense_size)
        self.archetypes = nn.parameter.Parameter(init_archetypes, requires_grad=True)
        self.flatten = nn.Flatten(0, 1)

    def forward(self, c, t):
        Z = self.context_encoder(c).unsqueeze(-1)
        A = self.task_encoder(t).unsqueeze(1)
        batch_size = A.shape[0]
        batch_archetypes = self.archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
        W = torch.bmm(A, batch_archetypes)
        beta = torch.bmm(W, Z)
        return self.flatten(beta)
        
