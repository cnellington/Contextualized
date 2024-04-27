Analysis
==============

``contextualized.analysis`` contains functions to analyze and plot the results of contextualized models.
All functions can be loaded directly from the module, e.g. ``from contextualized.analysis import plot_heterogeneous_predictor_effects``.

.. currentmodule:: contextualized.analysis

.. autosummary:: 
    :nosignatures:
 
    pvals.calc_homogeneous_context_effects_pvals
    pvals.calc_homogeneous_predictor_effects_pvals
    pvals.calc_heterogeneous_predictor_effects_pvals
    pvals.test_each_context
    pvals.get_possible_pvals
    accuracy_split.print_acc_by_covars
    bootstraps.select_good_bootstraps
    embeddings.plot_lowdim_rep
    embeddings.plot_embedding_for_all_covars
    effects.plot_homogeneous_context_effects
    effects.plot_homogeneous_predictor_effects
    effects.plot_heterogeneous_predictor_effects

.. autofunction:: contextualized.analysis.pvals.calc_homogeneous_context_effects_pvals
.. autofunction:: contextualized.analysis.pvals.calc_homogeneous_predictor_effects_pvals
.. autofunction:: contextualized.analysis.pvals.calc_heterogeneous_predictor_effects_pvals
.. autofunction:: contextualized.analysis.pvals.test_each_context
.. autofunction:: contextualized.analysis.pvals.get_possible_pvals
.. autofunction:: contextualized.analysis.accuracy_split.print_acc_by_covars
.. autofunction:: contextualized.analysis.bootstraps.select_good_bootstraps
.. autofunction:: contextualized.analysis.embeddings.plot_lowdim_rep
.. autofunction:: contextualized.analysis.embeddings.plot_embedding_for_all_covars
.. autofunction:: contextualized.analysis.effects.plot_homogeneous_context_effects
.. autofunction:: contextualized.analysis.effects.plot_homogeneous_predictor_effects
.. autofunction:: contextualized.analysis.effects.plot_heterogeneous_predictor_effects