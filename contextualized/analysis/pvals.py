"""
Analysis tools for generating pvalues from bootstrap replicates.

"""

from typing import *

import numpy as np
import pandas as pd

from contextualized.analysis.effects import (
    get_homogeneous_context_effects,
    get_homogeneous_predictor_effects,
    get_heterogeneous_predictor_effects,
)
from contextualized.easy.wrappers import SKLearnWrapper


def get_possible_pvals(num_bootstraps: int) -> list:
    """
    Get the range of possible p-values based on the number of bootstraps.

    Args:
        num_bootstraps (int): The number of bootstraps.

    Returns:
        list: The minimum and maximum possible p-values.
    """
    min_pval = 1 / (num_bootstraps + 1)
    max_pval = num_bootstraps / (num_bootstraps + 1)
    return [min_pval, max_pval]


def _validate_args(n_bootstraps: int, verbose: bool = False) -> None:
    """
    Check that the test has a sufficient number of bootstraps.

    Args:
        num_bootstraps (int): The number of bootstraps.

    Raises:
        ValueError: If the number of bootstraps is less than 2.
    """
    if n_bootstraps < 2:
        raise ValueError(
            f"P-values are not well defined without multiple bootstrap samples."
        )
    min_pval, max_pval = get_possible_pvals(n_bootstraps)
    if verbose:
        print(
            "########################################################################################\n"
            f"You are testing a model which contains {n_bootstraps} bootstraps.\n"
            f"The minimum possible p-value is {min_pval}.\n"
            f"To allow for lower p-values, increase the model's n_bootstraps.\n"
            "########################################################################################"
        )


def calc_pval_bootstraps_one_sided(estimates, thresh=0, laplace_smoothing=1):
    """
    Calculate p-values from bootstrapped estimates.

    Parameters
    ----------
    estimates : np.ndarray
        Bootstrapped estimates of the test statistic.
    thresh : float, optional
    laplace_smoothing : int, optional
    """

    return (laplace_smoothing + np.sum(estimates < thresh)) / (
        estimates.shape[0] + laplace_smoothing
    )


def calc_pval_bootstraps_one_sided_mean(estimates, laplace_smoothing=1):
    """
    Calculate p-values from bootstrapped estimates.
    The p-value is calculated as the proportion of bootstrapped estimates that are:
        less than 0 if the mean of the estimates is positive,
        greater than 0 if the mean of the estimates is negative.

    Parameters
    ----------
    estimates : np.ndarray
        Bootstrapped estimates of the test statistic.
    laplace_smoothing : int, optional
    """

    return calc_pval_bootstraps_one_sided(
        estimates * np.sign(np.mean(estimates)), 0, laplace_smoothing
    )


def calc_homogeneous_context_effects_pvals(
    model: SKLearnWrapper, C: np.ndarray, verbose: bool = True, **kwargs
) -> np.ndarray:
    """
    Calculate p-values for the effects of context directly on the outcome.

    Args:
        model (SKLearnWrapper): Model to analyze.
        C (np.ndarray): Contexts to analyze.
        verbose (bool): Whether to print the range of possible p-values.

    Returns:
        np.ndarray: P-values of shape (n_contexts, n_outcomes) testing whether the
            sign of the direct effect of context on outcomes is consistent across bootstraps.

    Raises:
        ValueError: If the model's n_bootstraps is less than 2.
    """
    _validate_args(model.n_bootstraps, verbose=verbose)
    _, effects = get_homogeneous_context_effects(model, C, **kwargs)
    # effects.shape: (n_contexts, n_bootstraps, n_context_vals, n_outcomes)
    diffs = effects[:, :, -1] - effects[:, :, 0]  # Test whether the sign is consistent
    pvals = np.array(
        [
            np.array(
                [
                    calc_pval_bootstraps_one_sided_mean(
                        diffs[i, :, j],
                        laplace_smoothing=kwargs.get("laplace_smoothing", 1),
                    )
                    for j in range(diffs.shape[2])  # n_outcomes
                ]
            )
            for i in range(diffs.shape[0])  # n_contexts
        ]
    )
    return pvals


def calc_homogeneous_predictor_effects_pvals(
    model: SKLearnWrapper, C: np.ndarray, verbose: bool = True, **kwargs
) -> np.ndarray:
    """
    Calculate p-values for the context-invariant effects of predictors.

    Args:
        model (SKLearnWrapper): Model to analyze.
        C (np.ndarray): Contexts to analyze.
        verbose (bool): Whether to print the range of possible p-values.

    Returns:
        np.ndarray: P-values of shape (n_predictors, n_outcomes) testing whether the
            sign of the context-invariant predictor effects are consistent across bootstraps.

    Raises:
        ValueError: If the model's n_bootstraps is less than 2.
    """
    _validate_args(model.n_bootstraps, verbose=verbose)
    _, effects = get_homogeneous_predictor_effects(model, C, **kwargs)
    # effects.shape: (n_predictors, n_bootstraps, n_outcomes)
    pvals = np.array(
        [
            np.array(
                [
                    calc_pval_bootstraps_one_sided_mean(
                        effects[i, :, j],
                        laplace_smoothing=kwargs.get("laplace_smoothing", 1),
                    )
                    for j in range(effects.shape[2])  # n_outcomes
                ]
            )
            for i in range(effects.shape[0])  # n_predictors
        ]
    )
    return pvals


def calc_heterogeneous_predictor_effects_pvals(
    model, C: np.ndarray, verbose: bool = True, **kwargs
) -> np.ndarray:
    """
    Calculate p-values for the heterogeneous (context-dependent) effects of predictors.

    Args:
        model (SKLearnWrapper): Model to analyze.
        C (np.ndarray): Contexts to analyze.
        verbose (bool): Whether to print the range of possible p-values.

    Returns:
        np.ndarray: P-values of shape (n_contexts, n_predictors, n_outcomes) testing whether the
            context-varying parameter range is consistent across bootstraps.

    Raises:
        ValueError: If the model's n_bootstraps is less than 2.
    """
    _validate_args(model.n_bootstraps, verbose=verbose)
    _, effects = get_heterogeneous_predictor_effects(model, C, **kwargs)
    # effects.shape is (n_contexts, n_predictors, n_bootstraps, n_context_vals, n_outcomes)
    diffs = (
        effects[:, :, :, -1] - effects[:, :, :, 0]
    )  # Test whether the sign is consistent
    # diffs.shape is (n_contexts, n_predictors, n_bootstraps, n_outcomes)
    pvals = np.array(
        [
            np.array(
                [
                    np.array(
                        [
                            calc_pval_bootstraps_one_sided_mean(
                                diffs[i, j, :, k],
                                laplace_smoothing=kwargs.get("laplace_smoothing", 1),
                            )
                            for k in range(diffs.shape[3])
                        ]
                    )  # n_outcomes
                    for j in range(diffs.shape[1])
                ]
            )  # n_predictors
            for i in range(diffs.shape[0])  # n_contexts
        ]
    )
    return pvals


def test_each_context(
    model_constructor: Type[SKLearnWrapper],
    C: pd.DataFrame,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    verbose: bool = True,
    model_kwargs: Dict = {"encoder_type": "linear"},
    fit_kwargs: Dict = {"max_epochs": 3, "learning_rate": 1e-2, "n_bootstraps": 20},
) -> pd.DataFrame:
    """
    Test heterogeneous predictor effects attributed to every individual context feature.
    Applies test_heterogeneous_predictor_effects to a model learned for a single context feature in C, and does this sequentially for every context feature.

    Args:
        model_constructor (SKLearnWrapper): The constructor of the model to be tested, currently either ContextualizedRegressor or ContextualizedClassifier.
        C (pd.DataFrame): The context dataframe (n_samples, n_contexts).
        X (pd.DataFrame): The predictor dataframe (n_samples, n_predictors).
        Y (pd.DataFrame): The outcome, target, or label dataframe (n_samples, n_outcomes).
        verbose (bool): Whether to print the range of possible p-values.
        **kwargs: Additional arguments for the model constructor.

    Returns:
        pd.DataFrame: A DataFrame of p-values for each (context, predictor, outcome) combination, describing how much the predictor's effect on the outcome varies across the context.

    Raises:
        ValueError: If the model's n_bootstraps is less than 2.
    """
    pvals_dict = {
        "Context": [],
        "Predictor": [],
        "Target": [],
        "Pvals": [],
    }
    _validate_args(fit_kwargs["n_bootstraps"], verbose=verbose)
    for context in C.columns:
        context_col = C[[context]].values
        model = model_constructor(**model_kwargs)
        model.fit(context_col, X.values, Y.values, **fit_kwargs)
        pvals = calc_heterogeneous_predictor_effects_pvals(
            model, context_col, verbose=False
        )
        for i, predictor in enumerate(X.columns):
            for j, outcome in enumerate(Y.columns):
                pvals_dict["Context"].append(context)
                pvals_dict["Predictor"].append(predictor)
                pvals_dict["Target"].append(outcome)
                pvals_dict["Pvals"].append(pvals[0, i, j])

    return pd.DataFrame.from_dict(pvals_dict)
