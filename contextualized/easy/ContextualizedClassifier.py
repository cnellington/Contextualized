"""
sklearn-like interface to Contextualized Classifiers.
"""

import numpy as np

from contextualized.functions import LINK_FUNCTIONS
from contextualized.easy import ContextualizedRegressor
from contextualized.regression import LOSSES


class ContextualizedClassifier(ContextualizedRegressor):
    """
    Contextualized Logistic Regression reveals context-dependent decisions and decision boundaries.
    Implemented as a ContextualizedRegressor with logistic link function and binary cross-entropy loss.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 0, which used the NaiveMetaModel. If > 0, uses archetypes in the ContextualizedMetaModel.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        kwargs["link_fn"] = LINK_FUNCTIONS["logistic"]
        kwargs["loss_fn"] = LOSSES["bceloss"]
        super().__init__(**kwargs)

    def predict(self, C, X, individual_preds=False, **kwargs):
        """Predict binary outcomes from context C and predictors X.

        Args:
            C (np.ndarray): Context array of shape (n_samples, n_context_features)
            X (np.ndarray): Predictor array of shape (N, n_features)
            individual_preds (bool, optional): Whether to return individual predictions for each model. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The binary outcomes predicted by the context-specific models (n_samples, y_dim). Returned as lists of individual bootstraps if individual_preds is True.
        """
        return np.round(super().predict(C, X, individual_preds, **kwargs))

    def predict_proba(self, C, X, **kwargs):
        """
        Predict probabilities of outcomes from context C and predictors X.

        Args:
            C (np.ndarray): Context array of shape (n_samples, n_context_features)
            X (np.ndarray): Predictor array of shape (N, n_features)
            individual_preds (bool, optional): Whether to return individual predictions for each model. Defaults to False.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: The outcome probabilities predicted by the context-specific models (n_samples, y_dim, 2). Returned as lists of individual bootstraps if individual_preds is True.
        """
        # Returns a np array of shape N samples, K outcomes, 2.
        probs = super().predict(C, X, **kwargs)
        return np.array([1 - probs, probs]).T.swapaxes(0, 1)
