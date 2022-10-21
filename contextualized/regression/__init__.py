"""
Contextualized Regression models.
"""
from contextualized.regression.datasets import (
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)
from contextualized.regression.losses import MSE
from contextualized.regression.regularizers import REGULARIZERS
from contextualized.regression.lightning_modules import (
    NaiveContextualizedRegression,
    ContextualizedRegression,
    MultitaskContextualizedRegression,
    TasksplitContextualizedRegression,
    ContextualizedUnivariateRegression,
    TasksplitContextualizedUnivariateRegression,
)
from contextualized.regression.trainers import RegressionTrainer

DATASETS = {
    "multivariate": MultivariateDataset,
    "univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}
LOSSES = {"mse": MSE}
MODELS = ["multivariate", "univariate"]
METAMODELS = ["simple", "subtype", "multitask", "tasksplit"]
TRAINERS = {"regression_trainer": RegressionTrainer}
