import numpy as np
import torch
from torch.nn.parameter import Parameter

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

        self.linear1 = torch.nn.Linear(context_shape[-1], final_dense_size)
        self.linear2 = torch.nn.Linear(final_dense_size, 1)
        self.flatten = torch.nn.Flatten(0, 1)

    def forward(self, c):
        x = torch.nn.functional.relu(self.linear1(c))
        x = torch.nn.functional.relu(self.linear2(x))
        return self.flatten(x) 
        

#     model = torch.nn.Sequential(
#         torch.nn.Linear(2, 20),
#         torch.nn.ReLU(),
#         torch.nn.Linear(20, 1),
#         torch.nn.ReLU(),
#         torch.nn.Flatten(0, 1)
#     )
# 
#     def loss(beta, xi, xj):
#         loss_val = (beta.unsqueeze(-1) * xi - xj).pow(2)
#         return loss_val
# 
#     lr = 1e-3
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     for _ in range(1000):
#         beta_pred = model(X_in)
#         loss_val = loss(beta_pred, Xi, Xj)
#         model.zero_grad()
#         loss_val.sum().backward()
#         print(loss_val.sum().detach())
#         opt.step()
#     
#     print(model.parameters())
       


