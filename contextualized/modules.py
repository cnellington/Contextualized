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


ENCODERS = {
    "mlp": MLP,
    "ngam": NGAM,
}


if __name__ == "__main__":
    N_SAMPLES = 100
    X_DIM = 10
    Y_DIM = 5
    K_ARCHETYPES = 3
    WIDTH = 50
    LAYERS = 5
    X_data = torch.rand((N_SAMPLES, X_DIM))

    mlp = MLP(X_DIM, Y_DIM, WIDTH, LAYERS)
    mlp(X_data)

    ngam = NGAM(X_DIM, Y_DIM, WIDTH, LAYERS)
    ngam(X_data)

    IN_DIMS = (3, 4)
    OUT_SHAPE = (5, 6)
    Z1 = torch.randn(N_SAMPLES, IN_DIMS[0])
    Z2 = torch.randn(N_SAMPLES, IN_DIMS[1])
    softselect = SoftSelect(IN_DIMS, OUT_SHAPE)
    softselect(Z1, Z2)

    precycle_vals = softselect.archetypes
    assert precycle_vals.shape == (*OUT_SHAPE, *IN_DIMS)
    postcycle_vals = softselect.get_archetypes()
    assert postcycle_vals.shape == (*IN_DIMS, *OUT_SHAPE)
    softselect.set_archetypes(torch.randn(*IN_DIMS, *OUT_SHAPE))
    assert (softselect.archetypes != precycle_vals).any()
    softselect.set_archetypes(postcycle_vals)
    assert (softselect.archetypes == precycle_vals).all()

    IN_DIMS = (3,)
    explainer = Explainer(IN_DIMS[0], OUT_SHAPE)
    explainer(Z1)

    precycle_vals = explainer.archetypes
    assert precycle_vals.shape == (*OUT_SHAPE, *IN_DIMS)
    postcycle_vals = explainer.get_archetypes()
    assert postcycle_vals.shape == (*IN_DIMS, *OUT_SHAPE)
    explainer.set_archetypes(torch.randn(*IN_DIMS, *OUT_SHAPE))
    assert (explainer.archetypes != precycle_vals).any()
    explainer.set_archetypes(postcycle_vals)
    assert (explainer.archetypes == precycle_vals).all()
