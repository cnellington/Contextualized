from contextualized.modules import MLP, NGAM
ENCODERS = {
    'mlp': MLP,
    'ngam': NGAM,
}

from contextualized.functions import *
LINK_FUNCTIONS = {
    'identity': linear_link_constructor(),
    'logistic': logistic_constructor(),
    'softmax': softmax_link_constructor()
}

from contextualized.regression.datasets import MultivariateDataset, UnivariateDataset, MultitaskMultivariateDataset, MultitaskUnivariateDataset
DATASETS = {
    "multivariate": MultivariateDataset,
    "univariate": UnivariateDataset,
    "multitask_multivariate": MultitaskMultivariateDataset,
    "multitask_univariate": MultitaskUnivariateDataset,
}

from contextualized.regression.losses import MSE
LOSSES = {
    'mse': MSE
}

from contextualized.regression.regularizers import *
REGULARIZERS = {
    'none': no_reg(),
    'l1': l1_reg,
    'l2': l2_reg,
    'l1_l2': l1_l2_reg
}

from contextualized.regression.lightning_modules import NaiveContextualizedRegression, ContextualizedRegression, MultitaskContextualizedRegression, TasksplitContextualizedRegression, ContextualizedUnivariateRegression, TasksplitContextualizedUnivariateRegression
MODELS = ['multivariate', 'univariate']
METAMODELS = ['simple', 'subtype', 'multitask', 'tasksplit']

from contextualized.regression.trainers import RegressionTrainer
TRAINERS = {
    'regression_trainer': RegressionTrainer
}
