"""
Contextualized Directed Acyclic Graphs (DAGs).
"""

from contextualized.modules import MLP, NGAM
from contextualized.dags.torch_notmad import NOTMAD_model

ENCODERS = {
    "mlp": MLP,
    "ngam": NGAM,
}

MODELS = ["bayesian"]
METAMODELS = ["subtype"]