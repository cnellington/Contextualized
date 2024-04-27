import numpy as np
from contextualized.easy.wrappers import SKLearnWrapper


def select_good_bootstraps(
    sklearn_wrapper: SKLearnWrapper, train_errs: np.ndarray, tol: float = 2
) -> SKLearnWrapper:
    """
    Prune any divergent or bad bootstraps with mean training errors below tol * min(training errors).

    Args:
        sklearn_wrapper (contextualized.easy.wrappers.SKLearnWrapper): Wrapper for the sklearn model.
        train_errs (np.ndarray): Training errors for each bootstrap (n_bootstraps, n_samples, n_outcomes).
        tol (float): Only bootstraps with mean train_errs below tol * min(train_errs) are kept.

    Returns:
        contextualized.easy.wrappers.SKLearnWrapper: The input model with only selected bootstraps.
    """
    if len(train_errs.shape) == 2:
        train_errs = train_errs[:, :, None]

    train_errs_by_bootstrap = np.mean(train_errs, axis=(1, 2))
    train_errs_min = np.min(train_errs_by_bootstrap)
    sklearn_wrapper.models = [
        model
        for train_err, model in zip(train_errs_by_bootstrap, sklearn_wrapper.models)
        if train_err < train_errs_min * tol
    ]
    sklearn_wrapper.n_bootstraps = len(sklearn_wrapper.models)
    return sklearn_wrapper
