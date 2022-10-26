"""
sklearn-like interface to Contextualized Regressors.
"""
from contextualized.regression import (
    NaiveContextualizedRegression,
    ContextualizedRegression,
)
from contextualized.regression import REGULARIZERS, LOSSES
from contextualized.functions import LINK_FUNCTIONS

from contextualized.easy.wrappers import SKLearnPredictorWrapper

# TODO: Multitask metamodels
# TODO: Task-specific link functions.
# TODO: Easier early stopping (right now, have to pass in 'callback_constructors' kwarg.


class ContextualizedRegressor(SKLearnPredictorWrapper):
    """
    sklearn-like interface to Contextualized Regression.
    """

    def __init__(self, **kwargs):
        self.num_archetypes = kwargs.get("num_archetypes", 0)
        if self.num_archetypes == 0:
            self.constructor = NaiveContextualizedRegression
        elif self.num_archetypes > 0:
            self.constructor = ContextualizedRegression
        else:
            print(
                f"""
                Was told to construct a ContextualizedRegressor with {self.num_archetypes}
                archetypes, but this should be a non-negative integer."""
            )
        constructor_kwargs, convenience_kwargs = self._organize_constructor_kwargs(
            **kwargs
        )
        not_constructor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in constructor_kwargs and k not in convenience_kwargs
        }
        super().__init__(self.constructor, **not_constructor_kwargs)
        for key, value in constructor_kwargs.items():
            self._init_kwargs["model"][key] = value

    def _organize_constructor_kwargs(self, **kwargs):
        """
        Helper function to set all the default constructor or changes allowed
        by ContextualizedRegressor.
        """
        constructor_kwargs = {}
        convenience_kwargs = ["subtype_probabilities", "alpha", "l1_ratio", "mu_ratio"]
        constructor_kwargs["link_fn"] = kwargs.get(
            "link_fn", LINK_FUNCTIONS["identity"]
        )
        constructor_kwargs["univariate"] = kwargs.get("univariate", False)
        constructor_kwargs["encoder_type"] = kwargs.get("encoder_type", "mlp")
        constructor_kwargs["loss_fn"] = kwargs.get("loss_fn", LOSSES["mse"])
        constructor_kwargs["encoder_kwargs"] = kwargs.get(
            "encoder_kwargs",
            {"width": 25, "layers": 2, "link_fn": LINK_FUNCTIONS["identity"]},
        )
        if kwargs.get("subtype_probabilities", False):
            constructor_kwargs["encoder_kwargs"]["link_fn"] = LINK_FUNCTIONS["softmax"]

        # Make regularizer
        if "alpha" in kwargs and kwargs["alpha"] > 0:
            constructor_kwargs["model_regularizer"] = REGULARIZERS["l1_l2"](
                kwargs["alpha"],
                kwargs.get("l1_ratio", 1.0),
                kwargs.get("mu_ratio", 0.5),
            )
        else:
            constructor_kwargs["model_regularizer"] = kwargs.get(
                "model_regularizer", REGULARIZERS["none"]
            )
        return constructor_kwargs, convenience_kwargs
