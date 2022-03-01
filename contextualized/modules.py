import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSelect(nn.Module):
    """
    Parameter sharing for multiple context encoders:
    Batched computation for mapping many subtypes onto d-dimensional archetypes
    """
    def __init__(self, in_dims, out_shape):
        super(SoftSelect, self).__init__()
        init_mat = torch.rand(list(out_shape) + list(in_dims)) * 2e-2 - 1e-2
        self.archetypes = nn.parameter.Parameter(init_mat, requires_grad=True)

    def forward(self, *batch_weights):
        batch_size = batch_weights[0].shape[0]
        expand_dims = [batch_size] + [-1 for _ in range(len(self.archetypes.shape))]
        batch_archetypes = self.archetypes.unsqueeze(0).expand(expand_dims)
        for batch_w in batch_weights[::-1]:
            batch_w = batch_w.unsqueeze(-1)
            d = len(batch_archetypes.shape) - len(batch_w.shape)
            for _ in range(d):
                batch_w = batch_w.unsqueeze(1)
            batch_archetypes = torch.matmul(batch_archetypes, batch_w).squeeze(-1)
        return batch_archetypes


class Explainer(nn.Module):
    """
    2D subtype-archetype parameter sharing
    """
    def __init__(self, k, out_shape):
        super(Explainer, self).__init__()
        self.softselect = SoftSelect((k, ), out_shape)

    def forward(self, batch_subtypes):
        return self.softselect(batch_subtypes)


class NGAM(nn.Module):
    """
    Neural generalized additive model
    """
    def __init__(self, input_dim, output_dim, width, layers, activation=nn.ReLU):
        super(NGAM, self).__init__()
        self.intput_dim = input_dim
        self.output_dim = output_dim
        hidden_layers = lambda: [layer for _ in range(0, layers - 2) for layer in (nn.Linear(width, width), activation())]
        nam_layers = lambda: [nn.Linear(1, width), activation()] + hidden_layers() + [nn.Linear(width, output_dim)]
        self.nams = nn.ModuleList([nn.Sequential(*nam_layers()) for _ in range(input_dim)])

    def forward(self, x):
        batch_size = x.shape[0]
        ret = torch.zeros((batch_size, self.output_dim))
        for i, nam in enumerate(self.nams):
            ret += nam(x[:, i].unsqueeze(-1))
        return ret

    
class MLP(nn.Module):
    """
    Multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim, width, layers, activation=nn.ReLU):
        super(MLP, self).__init__()
        hidden_layers = lambda: [layer for _ in range(0, layers - 2) for layer in (nn.Linear(width, width), activation())]
        mlp_layers = [nn.Linear(input_dim, width), activation()] + hidden_layers() + [nn.Linear(width, output_dim)]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        ret = self.mlp(x)
        return ret