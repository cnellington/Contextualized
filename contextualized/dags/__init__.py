"""
Contextualized Directed Acyclic Graphs (DAGs).
"""

from contextualized.modules import MLP, NGAM
from contextualized.dags.notmad import NOTMAD

ENCODERS = {
    "mlp": MLP,
    "ngam": NGAM,
}

MODELS = ["bayesian"]
METAMODELS = ["subtype"]
