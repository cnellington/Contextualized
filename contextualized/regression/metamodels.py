"""
Metamodels which generate context-specific models.
"""

import torch
from torch import nn

from contextualized.modules import ENCODERS, Explainer, SoftSelect
from contextualized.functions import LINK_FUNCTIONS

# Multivariate variants (explicit classes below)

class NaiveMultivariateMetamodel(nn.Module):
    """Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) --> {beta, mu} --> (X, Y)


    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        y_dim: int,
        encoder_type: str = "mlp",
        width: int = 25,
        layers: int = 1,
        link_fn: callable = LINK_FUNCTIONS["identity"],
    ):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels

        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        encoder_type (str: mlp): encoder module to use
        width (int: 25): width of the MLP encoder
        layers (int: 1): number of hidden layers in the MLP encoder
        link_fn (callable: identity): link function to apply to the output of the encoder
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        # multivariate: mu is scalar per y
        self.mu_dim = 1
        out_dim = (x_dim + self.mu_dim) * y_dim
        if encoder_type == "linear":
            self.context_encoder = encoder(context_dim, out_dim)
        else:
            self.context_encoder = encoder(
                context_dim, out_dim, width=width, layers=layers, link_fn=link_fn
            )

    def forward(self, C):
        """

        :param C:

        """
        W = self.context_encoder(C)
        W = torch.reshape(W, (W.shape[0], self.y_dim, self.x_dim + self.mu_dim))
        beta = W[:, :, : self.x_dim]
        mu = W[:, :, self.x_dim :]
        return beta, mu


class SubtypeMultivariateMetamodel(nn.Module):
    """Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z} --> {beta, mu} --> (X)

    Z: latent variable, causal parent of both the context and regression model


    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        y_dim: int,
        num_archetypes: int = 10,
        encoder_type: str = "mlp",
        width: int = 25,
        layers: int = 1,
        link_fn: callable = LINK_FUNCTIONS["identity"],
    ):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels

        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        num_archetypes (int: 10): number of atomic regression models in {Z}
        encoder_type (str: mlp): encoder module to use
        width (int: 25): width of the MLP encoder
        layers (int: 1): number of hidden layers in the MLP encoder
        link_fn (callable: identity): link function to apply to the output of the encoder
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        # multivariate: out_shape is (y_dim, x_dim + 1)
        out_shape = (y_dim, x_dim + 1)
        if encoder_type == "linear":
            self.context_encoder = encoder(context_dim, num_archetypes)
        else:
            self.context_encoder = encoder(
                context_dim, num_archetypes, width=width, layers=layers, link_fn=link_fn
            )
        self.explainer = Explainer(num_archetypes, out_shape)

    def forward(self, C):
        """

        :param C:

        """
        Z = self.context_encoder(C)
        W = self.explainer(Z)
        beta = W[:, :, : self.x_dim]
        mu = W[:, :, self.x_dim :]
        return beta, mu


class MultitaskMultivariateMetamodel(nn.Module):
    """Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z} --> {beta, mu} --> (X)
    (T) <---/

    Z: latent variable, causal parent of the context, regression model, and task (T)


    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        y_dim: int,
        num_archetypes: int = 10,
        encoder_type: str = "mlp",
        width: int = 25,
        layers: int = 1,
        link_fn: callable = LINK_FUNCTIONS["identity"],
    ):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels

        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        num_archetypes (int: 10): number of atomic regression models in {Z}
        encoder_type (str: mlp): encoder module to use
        width (int: 25): width of the MLP encoder
        layers (int: 1): number of hidden layers in the MLP encoder
        link_fn (callable: identity): link function to apply to the output of the encoder
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        encoder = ENCODERS[encoder_type]
        # multivariate multitask: beta_dim = x_dim, task_dim = y_dim
        beta_dim = x_dim
        task_dim = y_dim
        if encoder_type == "linear":
            self.context_encoder = encoder(context_dim + task_dim, num_archetypes)
        else:
            self.context_encoder = encoder(
                context_dim + task_dim,
                num_archetypes,
                width=width,
                layers=layers,
                link_fn=link_fn,
            )
        self.explainer = Explainer(num_archetypes, (beta_dim + 1,))

    def forward(self, C, T):
        """

        :param C:
        :param T:

        """
        CT = torch.cat((C, T), 1)
        Z = self.context_encoder(CT)
        W = self.explainer(Z)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu


class TasksplitMultivariateMetamodel(nn.Module):
    """Probabilistic assumptions as a graphical model (observed) {unobserved}:
    (C) <-- {Z_c} --> {beta, mu} --> (X)
    (T) <-- {Z_t} ----^

    Z_c: latent context variable, causal parent of the context and regression model
    Z_t: latent task variable, causal parent of the task and regression model


    """

    def __init__(
        self,
        context_dim: int,
        x_dim: int,
        y_dim: int,
        context_archetypes: int = 10,
        task_archetypes: int = 10,
        context_encoder_type: str = "mlp",
        context_width: int = 25,
        context_layers: int = 1,
        context_link_fn: callable = LINK_FUNCTIONS["softmax"],
        task_encoder_type: str = "mlp",
        task_width: int = 25,
        task_layers: int = 1,
        task_link_fn: callable = LINK_FUNCTIONS["identity"],
    ):
        """
        context_dim (int): dimension of flattened context
        x_dim (int): dimension of flattened features
        y_dim (int): dimension of flattened labels

        key-word args:
        univariate (bool: False): flag to solve a univariate regression problem instead
            of the standard multivariate problem
        context_archetypes (int: 10): number of atomic regression models in {Z_c}
        task_archetypes (int: 10): number of atomic regression models in {Z_t}
        context_encoder_type (str: mlp): context encoder module to use
        context_width (int: 25): width of the MLP context encoder
        context_layers (int: 1): number of hidden layers in the MLP context encoder
        context_link_fn (callable: softmax): link function to apply to the output of the context encoder
        task_encoder_type (str: mlp): task encoder module to use
        task_width (int: 25): width of the MLP task encoder
        task_layers (int: 1): number of hidden layers in the MLP task encoder
        task_link_fn (callable: identity): link function to apply to the output of the task encoder
        """
        super().__init__()
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        context_encoder = ENCODERS[context_encoder_type]
        task_encoder = ENCODERS[task_encoder_type]
        # multivariate tasksplit: beta_dim = x_dim, task_dim = y_dim
        beta_dim = x_dim
        task_dim = y_dim
        self.context_encoder = context_encoder(
            context_dim,
            context_archetypes,
            width=context_width,
            layers=context_layers,
            link_fn=context_link_fn,
        )
        self.task_encoder = task_encoder(
            task_dim,
            task_archetypes,
            width=task_width,
            layers=task_layers,
            link_fn=task_link_fn,
        )
        self.explainer = SoftSelect(
            (context_archetypes, task_archetypes), (beta_dim + 1,)
        )

    def forward(self, C, T):
        """

        :param C:
        :param T:

        """
        Z_c = self.context_encoder(C)
        Z_t = self.task_encoder(T)
        W = self.explainer(Z_c, Z_t)
        beta = W[:, :-1]
        mu = W[:, -1:]
        return beta, mu


SINGLE_TASK_METAMODELS = {
    "naive": NaiveMultivariateMetamodel,
    "subtype": SubtypeMultivariateMetamodel,
}

MULTITASK_METAMODELS = {
    "multitask": MultitaskMultivariateMetamodel,
    "tasksplit": TasksplitMultivariateMetamodel,
}

# Backwards compatible aliases for the original class names
NaiveMetamodel = NaiveMultivariateMetamodel
SubtypeMetamodel = SubtypeMultivariateMetamodel
MultitaskMetamodel = MultitaskMultivariateMetamodel
TasksplitMetamodel = TasksplitMultivariateMetamodel

# Univariate variants (explicit classes below)


class NaiveUnivariateMetamodel(NaiveMetamodel):
    """Univariate version of NaiveMetamodel where mu has dimension x_dim."""

    def __init__(self, context_dim, x_dim, y_dim, encoder_type="mlp", width=25, layers=1, link_fn=LINK_FUNCTIONS["identity"]):
        super().__init__(context_dim, x_dim, y_dim, encoder_type=encoder_type, width=width, layers=layers, link_fn=link_fn)
        # override mu_dim and reshaping behavior
        self.mu_dim = x_dim
        out_dim = (x_dim + self.mu_dim) * y_dim
        # rebuild encoder to match new out_dim
        encoder = ENCODERS[encoder_type]
        if encoder_type == "linear":
            self.context_encoder = encoder(context_dim, out_dim)
        else:
            self.context_encoder = encoder(context_dim, out_dim, width=width, layers=layers, link_fn=link_fn)


class SubtypeUnivariateMetamodel(SubtypeMetamodel):
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=10, encoder_type="mlp", width=25, layers=1, link_fn=LINK_FUNCTIONS["identity"]):
        super().__init__(context_dim, x_dim, y_dim, num_archetypes=num_archetypes, encoder_type=encoder_type, width=width, layers=layers, link_fn=link_fn)
        # adjust explainer out shape for univariate behavior
        out_shape = (y_dim, x_dim * 2, 1)
        self.explainer = Explainer(num_archetypes, out_shape)


class MultitaskUnivariateMetamodel(MultitaskMetamodel):
    def __init__(self, context_dim, x_dim, y_dim, num_archetypes=10, encoder_type="mlp", width=25, layers=1, link_fn=LINK_FUNCTIONS["identity"]):
        # For univariate multitask, beta is scalar per task and task_dim = y_dim + x_dim
        super().__init__(context_dim, x_dim, y_dim, num_archetypes=num_archetypes, encoder_type=encoder_type, width=width, layers=layers, link_fn=link_fn)
        beta_dim = 1
        task_dim = y_dim + x_dim
        encoder = ENCODERS[encoder_type]
        if encoder_type == "linear":
            self.context_encoder = encoder(context_dim + task_dim, num_archetypes)
        else:
            self.context_encoder = encoder(context_dim + task_dim, num_archetypes, width=width, layers=layers, link_fn=link_fn)
        self.explainer = Explainer(num_archetypes, (beta_dim + 1,))


class TasksplitUnivariateMetamodel(TasksplitMetamodel):
    def __init__(self, context_dim, x_dim, y_dim, context_archetypes=10, task_archetypes=10, context_encoder_type="mlp", context_width=25, context_layers=1, context_link_fn=LINK_FUNCTIONS["softmax"], task_encoder_type="mlp", task_width=25, task_layers=1, task_link_fn=LINK_FUNCTIONS["identity"]):
        super().__init__(context_dim, x_dim, y_dim, context_archetypes=context_archetypes, task_archetypes=task_archetypes, context_encoder_type=context_encoder_type, context_width=context_width, context_layers=context_layers, context_link_fn=context_link_fn, task_encoder_type=task_encoder_type, task_width=task_width, task_layers=task_layers, task_link_fn=task_link_fn)
        beta_dim = 1
        task_dim = y_dim + x_dim
        context_encoder = ENCODERS[context_encoder_type]
        task_encoder = ENCODERS[task_encoder_type]
        self.context_encoder = context_encoder(context_dim, context_archetypes, width=context_width, layers=context_layers, link_fn=context_link_fn)
        self.task_encoder = task_encoder(task_dim, task_archetypes, width=task_width, layers=task_layers, link_fn=task_link_fn)
        self.explainer = SoftSelect((context_archetypes, task_archetypes), (beta_dim + 1,))


SINGLE_TASK_UNIVARIATE_METAMODELS = {
    "naive": NaiveUnivariateMetamodel,
    "subtype": SubtypeUnivariateMetamodel,
}

MULTITASK_UNIVARIATE_METAMODELS = {
    "multitask": MultitaskUnivariateMetamodel,
    "tasksplit": TasksplitUnivariateMetamodel,
}
