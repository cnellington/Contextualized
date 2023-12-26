# Utility functions for bootstraps

def select_good_bootstraps(sklearn_wrapper, train_errs, tol=2, **kwargs):
    """
    Select bootstraps that are good for a given model.

    Parameters
    ----------
    sklearn_wrapper : contextualized.easy.wrappers.SKLearnWrapper
    train_errs : np.ndarray of shape (n_bootstraps, n_samples, n_outcomes)
    tol : float tolerance for the mean of the train_errs

    Returns
    -------
    sklearn_wrapper : sklearn_wrapper with only selected bootstraps
    """
    if len(train_errs.shape) == 2:
        train_errs = train_errs[:, :, None]

    train_errs_by_bootstrap = np.mean(train_errs, axis=(1, 2))
    sklearn_wrapper.models = sklearn_wrapper.models[
         train_errs_by_bootstrap < tol*np.min(train_errs_by_bootstrap)]
    return sklearn_wrapper
