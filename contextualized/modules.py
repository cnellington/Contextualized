import torch
import torch.nn as nn

from contextualized.functions import identity_link, identity


class SoftSelect(nn.Module):
    """
    Parameter sharing for multiple context encoders:
    Batched computation for mapping many subtypes onto d-dimensional archetypes
    """
    def __init__(self, in_dims, out_shape):
        super(SoftSelect, self).__init__()
        self.in_dims = in_dims
        self.out_shape = out_shape
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

    def _cycle_dims(self, tensor, n):
        """
        Cycle tensor dimensions from front to back for n steps
        """
        for _ in range(n):
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
        self.archetypes = nn.parameter.Parameter(self._cycle_dims(archetypes, len(self.out_shape)), requires_grad=True)


class Explainer(SoftSelect):
    """
    2D subtype-archetype parameter sharing
    """
    def __init__(self, k, out_shape):
        super().__init__((k, ), out_shape)


class MLP(nn.Module):
    """
    Multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim, width, layers, activation=nn.ReLU, link_fn=identity_link):
        super(MLP, self).__init__()
        if layers > 0:
            mlp_layers = [nn.Linear(input_dim, width), activation()]
            for _ in range(layers - 1):
                mlp_layers += [nn.Linear(width, width), activation()]
            mlp_layers.append(nn.Linear(width, output_dim))
        else:  # Linear encoder
            mlp_layers = [nn.Linear(input_dim, output_dim)]
        self.mlp = nn.Sequential(*mlp_layers)
        self.link_fn = link_fn

    def forward(self, x):
        ret = self.mlp(x)
        return self.link_fn(ret)


class NGAM(nn.Module):
    """
    Neural generalized additive model
    """
    def __init__(self, input_dim, output_dim, width, layers, activation=nn.ReLU, link_fn=identity_link):
        super(NGAM, self).__init__()
        self.intput_dim = input_dim
        self.output_dim = output_dim
        self.nams = nn.ModuleList([MLP(1, output_dim, width, layers, activation=activation, link_fn=identity_link) for _ in range(input_dim)])
        self.link_fn = link_fn

    def forward(self, x):
        batch_size = x.shape[0]
        ret = torch.zeros((batch_size, self.output_dim))
        for i, nam in enumerate(self.nams):
            ret += nam(x[:, i].unsqueeze(-1))
        return self.link_fn(ret)


if __name__ == '__main__': 
    n = 100
    x_dim = 10
    y_dim = 5
    k = 3
    width = 50
    layers = 5
    x = torch.rand((n, x_dim))
    
    mlp = MLP(x_dim, y_dim, width, layers)
    mlp(x)

    ngam = NGAM(x_dim, y_dim, width, layers)
    ngam(x)
    
    in_dims = (3, 4)
    out_shape = (5, 6)
    z1 = torch.randn(100, 3)
    z2 = torch.randn(100, 4)
    softselect = SoftSelect(in_dims, out_shape)
    softselect(z1, z2)

    precycle_vals = softselect.archetypes
    assert precycle_vals.shape == (*out_shape, *in_dims)
    postcycle_vals = softselect.get_archetypes()
    assert postcycle_vals.shape == (*in_dims, *out_shape)
    softselect.set_archetypes(torch.randn(*in_dims, *out_shape))
    assert (softselect.archetypes != precycle_vals).any()
    softselect.set_archetypes(postcycle_vals)
    assert (softselect.archetypes == precycle_vals).all()

    in_dims = (3, )
    explainer = Explainer(in_dims[0], out_shape)
    explainer(z1)
    
    precycle_vals = explainer.archetypes
    assert precycle_vals.shape == (*out_shape, *in_dims)
    postcycle_vals = explainer.get_archetypes()
    assert postcycle_vals.shape == (*in_dims, *out_shape)
    explainer.set_archetypes(torch.randn(*in_dims, *out_shape))
    assert (explainer.archetypes != precycle_vals).any()
    explainer.set_archetypes(postcycle_vals)
    assert (explainer.archetypes == precycle_vals).all()

