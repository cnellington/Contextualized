"""
Utilities for analyzing contextualized models.
"""

from contextualized.analysis.accuracy_split import print_acc_by_covars
from contextualized.analysis.bootstraps import select_good_bootstraps
from contextualized.analysis.embeddings import (
    plot_lowdim_rep,
    plot_embedding_for_all_covars,
)
from contextualized.analysis.effects import (
    plot_homogeneous_context_effects,
    plot_homogeneous_predictor_effects,
    plot_heterogeneous_predictor_effects,
)
from contextualized.analysis.pvals import (
    calc_homogeneous_context_effects_pvals,
    calc_homogeneous_predictor_effects_pvals,
    calc_heterogeneous_predictor_effects_pvals,
    test_each_context,
    get_possible_pvals,
)
