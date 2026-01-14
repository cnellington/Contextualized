"""
Contextualized Regression models.
"""

from contextualized.regression.datasets import (
    MultivariateDataset,
    UnivariateDataset,
    MultitaskMultivariateDataset,
    MultitaskUnivariateDataset,
)
from contextualized.regression.losses import MSE, BCELoss
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
from contextualized.regression.datamodules import (
    ContextualizedRegressionDataModule,
    TASK_TO_DATASET,
)

DATASETS = {
    "multivariate": MultivariateDataset,
    "univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}

LOSSES = {"mse": MSE, "bceloss": BCELoss}

MODELS = ["multivariate", "univariate"]

METAMODELS = ["simple", "subtype", "multitask", "tasksplit"]

TRAINERS = {"regression_trainer": RegressionTrainer}

# New exports for distributed-ready data handling
DATAMODULES = {
    "regression": ContextualizedRegressionDataModule,
}

__all__ = [
    # datasets
    "MultivariateDataset",
    "UnivariateDataset",
    "MultitaskMultivariateDataset",
    "MultitaskUnivariateDataset",
    "DATASETS",
    # datamodules
    "ContextualizedRegressionDataModule",
    "TASK_TO_DATASET",
    "DATAMODULES",
    # losses/regularizers
    "MSE",
    "BCELoss",
    "REGULARIZERS",
    "LOSSES",
    # models
    "NaiveContextualizedRegression",
    "ContextualizedRegression",
    "MultitaskContextualizedRegression",
    "TasksplitContextualizedRegression",
    "ContextualizedUnivariateRegression",
    "TasksplitContextualizedUnivariateRegression",
    "MODELS",
    "METAMODELS",
    # trainers
    "RegressionTrainer",
    "TRAINERS",
]