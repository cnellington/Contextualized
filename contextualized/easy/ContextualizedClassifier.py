"""
sklearn-like interface to Contextualized Classifiers.
"""

import numpy as np

from contextualized.functions import LINK_FUNCTIONS
from contextualized.easy import ContextualizedRegressor


class ContextualizedClassifier(ContextualizedRegressor):
    """
    sklearn-like interface to Contextualized Classifiers.
    """

    def __init__(self, **kwargs):
        kwargs["link_fn"] = LINK_FUNCTIONS["logistic"]
        super().__init__(**kwargs)

    def predict(self, C, X, individual_preds=False, **kwargs):
        """
        Predict outcomes from context C and predictors X.

        :param C:
        :param X:
        :param individual_preds:
        :param **kwargs:

        """
        return np.round(super().predict(C, X, individual_preds, **kwargs))

    def predict_proba(self, C, X, **kwargs):
        """
        Predict probabilities of outcomes from context C and predictors X.

        :param C:
        :param X:
        :param **kwargs:

        """
        # Returns a np array of shape N samples, K outcomes, 2.
        probs = super().predict(C, X, **kwargs)
        return np.array([1 - probs, probs]).T.swapaxes(0, 1)
