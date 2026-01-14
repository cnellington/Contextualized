"""Contextualized.ML: A statistical machine learning toolbox for estimating
	models, distributions, and functions with context-specific parameters.
	For more details, please refer to contextualized.ml.
"""
import torch

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high") 
    except Exception:
        pass
from contextualized import analysis
from contextualized import dags
from contextualized import easy
from contextualized import regression
from contextualized import baselines
from contextualized import utils
from contextualized.utils import *


