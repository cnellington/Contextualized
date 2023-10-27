"""
Miscellaneous utility functions.
"""

import numpy as np
import pandas as pd

from contextualized.analysis.pvals import (
    calc_homogeneous_context_effects_pvals,
)


def convert_to_one_hot(col):
    """

    :param col: np array with observations

    returns col converted to one-hot values, and list of one-hot values.

    """
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals

def test_sequential_contexts(model_constructor, C, X, Y, **kwargs):
    """
    Sequentially test each feature in C using calc_homogeneous_context_effects_pvals.

    Parameters
    ----------
    model_constructor : contextualized.models.Model constructor
        The model to be tested.
    C : pandas DataFrame
        The context data with multiple features.
    X : pandas DataFrame
        The input training data.
    Y : pandas DataFrame
        The output training data.
    **kwargs : 
        Additional arguments for the model constructor.

    Returns
    -------
    pvals_list : list
        A list of p-values for each feature.
    """

    default_fit_params = {
        'encoder_type': 'mlp',
        'max_epochs': 3,
        'learning_rate': 1e-2
    }
    fit_params = {**default_fit_params, **kwargs}
    
    pvals_list = []

    for column in C.columns:
        feature_subset = C[[column]].values
        model = model_constructor(**fit_params)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()

        model.fit(feature_subset, X, Y, **fit_params)

        curr_pval = calc_homogeneous_context_effects_pvals(model, feature_subset)
        pvals_list.append(curr_pval.item())

    return pvals_list
