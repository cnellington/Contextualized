from contextualized.modules import MLP, NGAM
ENCODERS = {
    'mlp': MLP,
    'ngam': NGAM,
}

from contextualized.dags.notmad import NOTMAD
MODELS = ['bayesian']
METAMODELS = ['subtype']
