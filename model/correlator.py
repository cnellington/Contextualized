import numpy as np
import torch
from torch.nn.parameter import Parameter

def loss(beta, xi, xj):
    return (beta.unsqueeze(-1) * xi - xj).pow(2)

class ContextualCorrelator(torch.nn.Module):
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
        
        self.context_encoder = torch.nn.Sequential(
            torch.nn.Linear(context_shape[-1], final_dense_size), 
            torch.nn.ReLU(), 
        )
        self.task_encoder = torch.nn.Sequential(
            torch.nn.Linear(task_shape[-1], num_archetypes), 
            torch.nn.Softmax(dim=1), 
        )
        init_archetypes = torch.rand(num_archetypes, final_dense_size)
        self.archetypes = torch.nn.parameter.Parameter(init_archetypes, requires_grad=True)
        self.flatten = torch.nn.Flatten(0, 1)
        """
        self.task_dense = torch.nn.Linear(task_encoder_in_shape, 1)
        self.task_softmax = torch.nn.Softmax(dim=1)
        self.archetypes = torch.tensor(np.random.random((final_dense_size, 1)), requires_grad=True)
        self.final_dense = torch.bmm(self.archetypes, self.task_softmax)
        self.final_dense.T @ self.linear1(c)
        relu()
        """

    def forward(self, c, t):
        Z = self.context_encoder(c).unsqueeze(-1)
        A = self.task_encoder(t).unsqueeze(1)
        batch_size = A.shape[0]
        batch_archetypes = self.archetypes.unsqueeze(0).repeat(batch_size, 1, 1)
        W = torch.bmm(A, batch_archetypes)
        beta = torch.bmm(W, Z)
        return self.flatten(beta)
        
