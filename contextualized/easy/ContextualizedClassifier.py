import numpy as np

from contextualized.regression import LINK_FUNCTIONS
from contextualized.easy import ContextualizedRegressor


def inv_sigmoid(x, slope=1):
    return (1./slope)*np.log(x / (1-x))


class ContextualizedClassifier(ContextualizedRegressor):
    def __init__(self, **kwargs):
        kwargs['link_fn'] = LINK_FUNCTIONS['logistic']
        super().__init__(**kwargs)

    def predict(self, C, X, **kwargs):
        return np.round(super().predict(C, X, **kwargs))

    def predict_proba(self, C, X, **kwargs):
        # Returns a np array of shape N samples, K outcomes, 2.
        probs = super().predict(C, X, **kwargs)
        return np.array([1-probs, probs]).T.swapaxes(0, 1)
