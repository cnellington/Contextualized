"""
sklearn-like interface to Contextualized Regressors.
"""

from contextualized.regression import (
    NaiveContextualizedRegression,
    ContextualizedRegression,
)
from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression import RegressionTrainer

# TODO: Multitask metamodels
# TODO: Task-specific link functions.


class ContextualizedRegressor(SKLearnWrapper):
    """
    Contextualized Linear Regression quantifies context-varying linear relationships.

    Args:
        n_bootstraps (int, optional): Number of bootstraps to use. Defaults to 1.
        num_archetypes (int, optional): Number of archetypes to use. Defaults to 0, which used the NaiveMetaModel. If > 0, uses archetypes in the ContextualizedMetaModel.
        encoder_type (str, optional): Type of encoder to use ("mlp", "ngam", "linear"). Defaults to "mlp".
        loss_fn (torch.nn.Module, optional): Loss function. Defaults to LOSSES["mse"].
        link_fn (torch.nn.Module, optional): Link function. Defaults to LINK_FUNCTIONS["identity"].
        alpha (float, optional): Regularization strength. Defaults to 0.0.
        mu_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization applies to context-specific parameters or context-specific offsets. Defaults to 0.0.
        l1_ratio (float, optional): Float in range (0.0, 1.0), governs how much the regularization penalizes l1 vs l2 parameter norms. Defaults to 0.0.
    """

    def __init__(self, **kwargs):
        self.num_archetypes = kwargs.get("num_archetypes", 0)
        if self.num_archetypes == 0:
            constructor = NaiveContextualizedRegression
        elif self.num_archetypes > 0:
            constructor = ContextualizedRegression
        else:
            print(
                f"""
                Was told to construct a ContextualizedRegressor with {self.num_archetypes}
                archetypes, but this should be a non-negative integer."""
            )

        extra_model_kwargs = ["base_param_predictor", "base_y_predictor", "y_dim"]
        extra_data_kwargs = ["Y_val"]
        trainer_constructor = RegressionTrainer
        super().__init__(
            constructor,
            extra_model_kwargs,
            extra_data_kwargs,
            trainer_constructor,
            **kwargs,
        )

    def _split_train_data(self, C, X, Y=None, Y_required=False, **kwargs):
        return super()._split_train_data(C, X, Y, Y_required=True, **kwargs)
