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

from contextualized.regression.datasets import UnivariateDataset, MultitaskUnivariateDataset
DATASETS = {
    "univariate": UnivariateDataset,
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

from contextualized.networks.lightning_modules import ContextualizedCorrelation, MultitaskContextualizedCorrelation
from contextualized.networks.notmad import NOTMAD
MODELS = ['correlation', 'bayesian']
METAMODELS = ['subtype', 'tasksplit']

from contextualized.networks.trainers import CorrelationTrainer
TRAINERS = {
    'correlation_trainer': CorrelationTrainer
}
