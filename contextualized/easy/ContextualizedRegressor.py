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
    sklearn-like interface to Contextualized Regression.
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

        extra_model_kwargs = [
            "base_param_predictor",
            "base_y_predictor",
            "y_dim"
        ]
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
