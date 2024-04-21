"""
PyTorch modules which are used as building blocks of Contextualized models.
"""

import torch
from torch import nn

from contextualized.functions import identity_link


class SoftSelect(nn.Module):
    """
    Parameter sharing for multiple context encoders:
    Batched computation for mapping many subtypes onto d-dimensional archetypes
    """

    def __init__(self, in_dims, out_shape):
        super().__init__()
        self.in_dims = in_dims
        self.out_shape = out_shape
        init_mat = torch.rand(list(out_shape) + list(in_dims)) * 2e-2 - 1e-2
        self.archetypes = nn.parameter.Parameter(init_mat, requires_grad=True)

    def forward(self, *batch_weights):
        """Torch Forward pass."""
        batch_size = batch_weights[0].shape[0]
        expand_dims = [batch_size] + [-1 for _ in range(len(self.archetypes.shape))]
        batch_archetypes = self.archetypes.unsqueeze(0).expand(expand_dims)
        for batch_w in batch_weights[::-1]:
            batch_w = batch_w.unsqueeze(-1)
            empty_dims = len(batch_archetypes.shape) - len(batch_w.shape)
            for _ in range(empty_dims):
                batch_w = batch_w.unsqueeze(1)
            batch_archetypes = torch.matmul(batch_archetypes, batch_w).squeeze(-1)
        return batch_archetypes

    def _cycle_dims(self, tensor, n_steps):
        """
        Cycle tensor dimensions from front to back for n steps
        """
        for _ in range(n_steps):
            tensor = tensor.unsqueeze(0).transpose(0, -1).squeeze(-1)
        return tensor

    def get_archetypes(self):
        """
        Returns archetype parameters: (*in_dims, *out_shape)
        """
        return self._cycle_dims(self.archetypes, len(self.in_dims))

    def set_archetypes(self, archetypes):
        """
        Sets archetype parameters

        Requires archetypes.shape == (*in_dims, *out_shape)
        """
        self.archetypes = nn.parameter.Parameter(
            self._cycle_dims(archetypes, len(self.out_shape)), requires_grad=True
        )


class Explainer(SoftSelect):
    """
    2D subtype-archetype parameter sharing
    """

    def __init__(self, k, out_shape):
        super().__init__((k,), out_shape)


class MLP(nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        width,
        layers,
        activation=nn.ReLU,
        link_fn=identity_link,
    ):
        super().__init__()
        if layers > 0:
            mlp_layers = [nn.Linear(input_dim, width), activation()]
            for _ in range(layers - 1):
                mlp_layers += [nn.Linear(width, width), activation()]
            mlp_layers.append(nn.Linear(width, output_dim))
        else:  # Linear encoder
            mlp_layers = [nn.Linear(input_dim, output_dim)]
        self.mlp = nn.Sequential(*mlp_layers)
        self.link_fn = link_fn

    def forward(self, X):
        """Torch Forward pass."""
        ret = self.mlp(X)
        return self.link_fn(ret)


class NGAM(nn.Module):
    """
    Neural generalized additive model
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        width,
        layers,
        activation=nn.ReLU,
        link_fn=identity_link,
    ):
        super().__init__()
        self.intput_dim = input_dim
        self.output_dim = output_dim
        self.nams = nn.ModuleList(
            [
                MLP(
                    1,
                    output_dim,
                    width,
                    layers,
                    activation=activation,
                    link_fn=identity_link,
                )
                for _ in range(input_dim)
            ]
        )
        self.link_fn = link_fn

    def forward(self, X):
        """Torch Forward pass."""
        ret = self.nams[0](X[:, 0].unsqueeze(-1))
        for i, nam in enumerate(self.nams[1:]):
            ret += nam(X[:, i].unsqueeze(-1))
        return self.link_fn(ret)


class Linear(nn.Module):
    """
    Linear encoder
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = MLP(
            input_dim, output_dim, width=output_dim, layers=0, activation=None
        )

    def forward(self, X):
        """Torch Forward pass."""
        return self.linear(X)


ENCODERS = {"mlp": MLP, "ngam": NGAM, "linear": Linear}
