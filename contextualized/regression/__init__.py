from contextualized.regression.datasets import (
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)

DATASETS = {
    "multivariate": MultivariateDataset,
    "univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}

from contextualized.regression.losses import MSE

LOSSES = {"mse": MSE}

from contextualized.regression.regularizers import REGULARIZERS

from contextualized.regression.lightning_modules import (
    NaiveContextualizedRegression,
    ContextualizedRegression,
    MultitaskContextualizedRegression,
    TasksplitContextualizedRegression,
    ContextualizedUnivariateRegression,
    TasksplitContextualizedUnivariateRegression,
)

MODELS = ["multivariate", "univariate"]
METAMODELS = ["simple", "subtype", "multitask", "tasksplit"]

from contextualized.regression.trainers import RegressionTrainer

TRAINERS = {"regression_trainer": RegressionTrainer}
