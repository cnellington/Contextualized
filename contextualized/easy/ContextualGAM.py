"""
Contextual Generalized Additive Model.
See https://www.sciencedirect.com/science/article/pii/S1532046422001022
for more details.
"""

from contextualized.easy import ContextualizedClassifier, ContextualizedRegressor


class ContextualGAMClassifier(ContextualizedClassifier):
    """
    A GAM as context encoder with a classifier on top.
    """

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)


class ContextualGAMRegressor(ContextualizedRegressor):
    """
    A GAM as context encoder with a regressor on top.
    """

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)
