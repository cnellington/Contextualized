"""
Contextual Generalized Additive Model.
See https://www.sciencedirect.com/science/article/pii/S1532046422001022
for more details.
"""

from contextualized.easy import ContextualizedClassifier, ContextualizedRegressor


class ContextualGAMClassifier(ContextualizedClassifier):
    """
    The Contextual GAM Classifier separates and interprets the effect of context in context-varying decisions and classifiers, such as heterogeneous disease diagnoses.
    Implemented as a Contextual Generalized Additive Model with a classifier on top.
    Always uses a Neural Additive Model ("ngam") encoder for interpretability.
    See `this paper <https://www.sciencedirect.com/science/article/pii/S1532046422001022>`__
    for more details.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 0, which used the NaiveMetaModel. If > 0, uses archetypes in the ContextualizedMetaModel.
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)


class ContextualGAMRegressor(ContextualizedRegressor):
    """
    The Contextual GAM Regressor separates and interprets the effect of context in context-varying relationships, such as heterogeneous treatment effects.
    Implemented as a Contextual Generalized Additive Model with a linear regressor on top.
    Always uses a Neural Additive Model ("ngam") encoder for interpretability.
    See `this paper <https://www.sciencedirect.com/science/article/pii/S1532046422001022>`__
    for more details.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 0, which used the NaiveMetaModel. If > 0, uses archetypes in the ContextualizedMetaModel.
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)
