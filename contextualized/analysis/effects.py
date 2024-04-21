"""
Utilities for plotting effects learned by Contextualized models.
"""

from typing import *

import numpy as np
import matplotlib.pyplot as plt

from contextualized.easy.wrappers import SKLearnWrapper


def simple_plot(
    x_vals: List[Union[float, int]],
    y_vals: List[Union[float, int]],
    **kwargs,
) -> None:
    """
    Simple plotting of y vs x with kwargs passed to matplotlib helpers.

    Args:
        x_vals: x values to plot
        y_vals: y values to plot
        **kwargs: kwargs passed to matplotlib helpers (fill_alpha, fill_color, y_lowers, y_uppers, x_label, y_label, x_ticks, x_ticklabels, y_ticks, y_ticklabels)

    Returns:
        None
    """
    plt.figure(figsize=kwargs.get("figsize", (8, 8)))
    if "y_lowers" in kwargs and "y_uppers" in kwargs:
        plt.fill_between(
            x_vals,
            np.squeeze(kwargs["y_lowers"]),
            np.squeeze(kwargs["y_uppers"]),
            alpha=kwargs.get("fill_alpha", 0.2),
            color=kwargs.get("fill_color", "blue"),
        )
    plt.plot(x_vals, y_vals)
    plt.xlabel(kwargs.get("x_label", "X"))
    plt.ylabel(kwargs.get("y_label", "Y"))
    if (
        kwargs.get("x_ticks", None) is not None
        and kwargs.get("x_ticklabels", None) is not None
    ):
        plt.xticks(kwargs["x_ticks"], kwargs["x_ticklabels"])
    if (
        kwargs.get("y_ticks", None) is not None
        and kwargs.get("y_ticklabels", None) is not None
    ):
        plt.yticks(kwargs["y_ticks"], kwargs["y_ticklabels"])
    plt.show()


def plot_effect(x_vals, y_means, y_lowers=None, y_uppers=None, **kwargs):
    """Plots a single effect."""
    min_val = np.min(y_means)
    y_means -= min_val
    if y_lowers is not None and y_uppers is not None:
        y_lowers -= min_val
        y_uppers -= min_val
    if kwargs.get("should_exponentiate", False):
        y_means = np.exp(y_means)
        if y_lowers is not None and y_uppers is not None:
            y_lowers = np.exp(y_lowers)
            y_uppers = np.exp(y_uppers)
    try:
        if "x_encoder" in kwargs and kwargs["x_encoder"] is not None:
            x_classes = kwargs["x_encoder"].classes_
            # Line up class values with centered values.
            x_ticks = np.array(list(range(len(x_classes))))
            if (
                kwargs.get("x_means", None) is not None
                and kwargs.get("x_stds", None) is not None
            ):
                x_ticks = (x_ticks - kwargs["x_means"]) / kwargs["x_stds"]
        else:
            x_ticks = None
            x_classes = None
    except:
        x_classes = None
        x_ticks = None

    if np.max(y_means) > kwargs.get("min_effect_size", 0.0):
        simple_plot(
            x_vals,
            y_means,
            x_label=kwargs.get("xlabel", "X"),
            y_label=kwargs.get("ylabel", "Odds Ratio of Outcome"),
            y_lowers=y_lowers,
            y_uppers=y_uppers,
            x_ticks=x_ticks,
            x_ticklabels=x_classes,
        )


def get_homogeneous_context_effects(
    model: SKLearnWrapper, C: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the homogeneous (context-invariant) effects of context.

    Args:
        model (SKLearnWrapper): a fitted ``contextualized.easy`` model
        C: the context values to use to estimate the effects
        verbose (bool, optional): print progess. Default True.
        individual_preds (bool, optional): whether to use plot each bootstrap. Default True.
        C_vis (np.ndarray, optional): Context bins used to visualize context (n_vis, n_contexts). Default None to construct anew.
        n_vis (int, optional): Number of bins to use to visualize context. Default 1000.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            c_vis: the context values that were used to estimate the effects
            effects: array of effects, one for each context. Each homogeneous effect is a matrix of shape:
                (n_bootstraps, n_context_vals, n_outcomes).
    """
    if kwargs.get("verbose", True):
        print("Estimating Homogeneous Contextual Effects.")
    c_vis = maybe_make_c_vis(C, **kwargs)

    effects = []
    for j in range(C.shape[1]):
        c_j = np.zeros_like(c_vis)
        c_j[:, j] = c_vis[:, j]
        try:
            (_, mus) = model.predict_params(
                c_j, individual_preds=kwargs.get("individual_preds", True)
            )
        except ValueError:
            (_, mus) = model.predict_params(c_j)
        effects.append(mus)
    return c_vis, np.array(effects)


def get_homogeneous_predictor_effects(model, C, **kwargs):
    """
    Get the homogeneous (context-invariant) effects of predictors.
    :param model:
    :param C:

    returns:
        c_vis: the context values that were used to estimate the effects
        effects: np array of effects, one for each predictor. Each homogeneous effect is a matrix of shape:
            (n_bootstraps, n_outcomes).

    """
    if kwargs.get("verbose", True):
        print("Estimating Homogeneous Predictor Effects.")
    c_vis = maybe_make_c_vis(C, **kwargs)
    c_idx = 0
    try:
        (betas, _) = model.predict_params(
            c_vis, individual_preds=kwargs.get("individual_preds", True)
        )
        # bootstraps x C_vis x outcomes x predictors
        if len(betas.shape) == 4:
            c_idx = 1
    except ValueError:
        (betas, _) = model.predict_params(c_vis)
    betas = np.mean(
        betas, axis=c_idx
    )  # homogeneous predictor effect is context-invariant
    return c_vis, np.transpose(betas, (2, 0, 1))


def get_heterogeneous_predictor_effects(model, C, **kwargs):
    """
    Get the heterogeneous (context-variant) effects of predictors.
    :param model:
    :param C:

    returns:
        c_vis: the context values that were used to estimate the effects
        effects: np array of effects, one for each context x predictor pair.
            Each heterogeneous effect is a matrix of shape:
                (n_predictors, n_bootstraps, n_context_vals, n_outcomes).
    """
    if kwargs.get("verbose", True):
        print("Estimating Heterogeneous Predictor Effects.")
    c_vis = maybe_make_c_vis(C, **kwargs)

    effects = []
    for j in range(C.shape[1]):
        c_j = np.zeros_like(c_vis)
        c_j[:, j] = c_vis[:, j]
        c_idx = 0
        try:
            (betas, _) = model.predict_params(
                c_j, individual_preds=kwargs.get("individual_preds", True)
            )
            # bootstraps x C_vis x outcomes x predictors
            if len(betas.shape) == 4:
                c_idx = 1
        except ValueError:
            (betas, _) = model.predict_params(c_j)
        # Heterogeneous Effects are mean-centered wrt C
        effect = np.transpose(
            np.transpose(betas, (0, 2, 3, 1))
            - np.tile(
                np.expand_dims(np.mean(betas, axis=c_idx), -1),
                (1, 1, 1, betas.shape[1]),
            ),
            (0, 3, 1, 2),
        )
        effects.append(effect)
    effects = np.array(effects)
    if len(effects.shape) == 5:
        effects = np.transpose(
            effects, (0, 4, 1, 2, 3)
        )  # (n_contexts, n_predictors, n_bootstraps, n_context_vals, n_outcomes)
    else:
        effects = np.transpose(
            effects, (0, 3, 1, 2)
        )  # (n_contexts, n_predictors, n_context_vals, n_outcomes)
    return c_vis, effects


def plot_boolean_vars(names, y_mean, y_err, **kwargs):
    """
    Plots Boolean variables.
    """
    plt.figure(figsize=kwargs.get("figsize", (12, 8)))
    sorted_i = np.argsort(y_mean)
    if kwargs.get("classification", True):
        y_mean = np.exp(y_mean)
        y_err = np.exp(y_err)
    for counter, i in enumerate(sorted_i):
        plt.bar(
            counter,
            y_mean[i],
            width=0.5,
            color=kwargs.get("fill_color", "blue"),
            edgecolor=kwargs.get("edge_color", "black"),
            yerr=y_err,
        )
    plt.xticks(
        range(len(names)),
        np.array(names)[sorted_i],
        rotation=60,
        fontsize=kwargs.get("boolean_x_ticksize", 18),
        ha="right",
    )
    plt.ylabel(
        kwargs.get("ylabel", "Odds Ratio of Outcome"),
        fontsize=kwargs.get("ylabel_fontsize", 32),
    )
    plt.yticks(fontsize=kwargs.get("ytick_fontsize", 18))
    if kwargs.get("bool_figname", None) is not None:
        plt.savefig(kwargs.get("bool_figname"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_homogeneous_context_effects(
    model: SKLearnWrapper,
    C: np.ndarray,
    **kwargs,
) -> None:
    """
    Plot the direct effect of context on outcomes, disregarding other features.
    This context effect is homogeneous in that it is a static function of context (context-invariant).

    Args:
        model (SKLearnWrapper): a fitted ``contextualized.easy`` model
        C: the context values to use to estimate the effects
        verbose (bool, optional): print progess. Default True.
        individual_preds (bool, optional): whether to use plot each bootstrap. Default True.
        C_vis (np.ndarray, optional): Context bins used to visualize context (n_vis, n_contexts). Default None to construct anew.
        n_vis (int, optional): Number of bins to use to visualize context. Default 1000.
        lower_pct (int, optional): Lower percentile for bootstraps. Default 2.5.
        upper_pct (int, optional): Upper percentile for bootstraps. Default 97.5.
        classification (bool, optional): Whether to exponentiate the effects. Default True.
        C_encoders (List[sklearn.preprocessing.LabelEncoder], optional): encoders for each context. Default None.
        C_means (np.ndarray, optional): means for each context. Default None.
        C_stds (np.ndarray, optional): standard deviations for each context. Default None.
        xlabel_prefix (str, optional): prefix for x label. Default "".
        figname (str, optional): name of figure to save. Default None.

    Returns:
        None
    """
    c_vis, effects = get_homogeneous_context_effects(model, C, **kwargs)
    # effects.shape is (n_context, n_bootstraps, n_context_vals, n_outcomes)
    for outcome in range(effects.shape[-1]):
        for j in range(effects.shape[0]):
            try:
                mus = effects[j, :, :, outcome]
                means = np.mean(mus, axis=0)
                lowers = np.percentile(mus, kwargs.get("lower_pct", 2.5), axis=0)
                uppers = np.percentile(mus, kwargs.get("upper_pct", 97.5), axis=0)
            except ValueError:
                mus = effects[j, :, outcome]
                means = mus  # no bootstraps were provided.
                lowers, uppers = None, None

            if "C_encoders" in kwargs:
                encoder = kwargs["C_encoders"][j]
            else:
                encoder = None
            if "C_means" in kwargs:
                c_means = kwargs["C_means"][j]
            else:
                c_means = None
            if "C_stds" in kwargs:
                c_stds = kwargs["C_stds"][j]
            else:
                c_stds = None
            plot_effect(
                c_vis[:, j],
                means,
                lowers,
                uppers,
                should_exponentiate=kwargs.get("classification", True),
                x_encoder=encoder,
                x_means=c_means,
                x_stds=c_stds,
                xlabel=C.columns.tolist()[j],
                **kwargs,
            )


def plot_homogeneous_predictor_effects(
    model: SKLearnWrapper,
    C: np.ndarray,
    X: np.ndarray,
    **kwargs,
) -> None:
    """
    Plot the effect of predictors on outcomes that do not change with context (homogeneous).

    Args:
        model (SKLearnWrapper): a fitted ``contextualized.easy`` model
        C: the context values to use to estimate the effects
        X: the predictor values to use to estimate the effects
        max_classes_for_discrete (int, optional): maximum number of classes to treat as discrete. Default 10.
        min_effect_size (float, optional): minimum effect size to plot. Default 0.003.
        ylabel (str, optional): y label for plot. Default "Influence of ".
        xlabel_prefix (str, optional): prefix for x label. Default "".
        X_names (List[str], optional): names of predictors. Default None.
        X_encoders (List[sklearn.preprocessing.LabelEncoder], optional): encoders for each predictor. Default None.
        X_means (np.ndarray, optional): means for each predictor. Default None.
        X_stds (np.ndarray, optional): standard deviations for each predictor. Default None.
        verbose (bool, optional): print progess. Default True.
        lower_pct (int, optional): Lower percentile for bootstraps. Default 2.5.
        upper_pct (int, optional): Upper percentile for bootstraps. Default 97.5.
        classification (bool, optional): Whether to exponentiate the effects. Default True.
        figname (str, optional): name of figure to save. Default None.

    Returns:
        None
    """
    c_vis = np.zeros_like(C.values)
    x_vis = make_grid_mat(X.values, 1000)
    (betas, _) = model.predict_params(
        c_vis, individual_preds=True
    )  # bootstraps x C_vis x outcomes x predictors
    homogeneous_betas = np.mean(betas, axis=1)  # bootstraps x outcomes x predictors
    for outcome in range(homogeneous_betas.shape[1]):
        betas = homogeneous_betas[:, outcome, :]  # bootstraps x predictors
        my_avg_betas = np.mean(betas, axis=0)
        lowers = np.percentile(betas, kwargs.get("lower_pct", 2.5), axis=0)
        uppers = np.percentile(betas, kwargs.get("upper_pct", 97.5), axis=0)
        max_impacts = []
        # Calculate the max impact of each effect.
        for k in range(my_avg_betas.shape[0]):
            effect_range = my_avg_betas[k] * np.ptp(x_vis[:, k])
            max_impacts.append(effect_range)
        effects_by_desc_impact = np.argsort(max_impacts)[::-1]

        boolean_vars = [j for j in range(X.shape[-1]) if len(set(X.values[:, j])) == 2]
        if len(boolean_vars) > 0:
            plot_boolean_vars(
                [X.columns[j] for j in boolean_vars],
                [max_impacts[j] for j in boolean_vars],
                [np.max(uppers[j]) - max_impacts[j] for j in boolean_vars],
                **kwargs,
            )
        for j in effects_by_desc_impact:
            if j in boolean_vars:
                continue
            means = my_avg_betas[j] * x_vis[:, j]
            my_lowers = lowers[j] * x_vis[:, j]
            my_uppers = uppers[j] * x_vis[:, j]
            if "X_encoders" in kwargs:
                encoder = kwargs["X_encoders"][j]
            else:
                encoder = None
            if "X_means" in kwargs:
                x_means = kwargs["X_means"][j]
            else:
                x_means = None
            if "X_stds" in kwargs:
                x_stds = kwargs["X_stds"][j]
            else:
                x_stds = None

            plot_effect(
                x_vis[:, j],
                means,
                my_lowers,
                my_uppers,
                should_exponentiate=kwargs.get("classification", True),
                x_encoder=encoder,
                x_means=x_means,
                x_stds=x_stds,
                xlabel=f"{kwargs.get('xlabel_prefix',  '')} {X.columns[j]}",
                **kwargs,
            )


def plot_heterogeneous_predictor_effects(model, C, X, **kwargs):
    """
    Plot how the effect of predictors on outcomes changes with context (heterogeneous).

    Args:
        model (SKLearnWrapper): a fitted ``contextualized.easy`` model
        C: the context values to use to estimate the effects
        X: the predictor values to use to estimate the effects
        max_classes_for_discrete (int, optional): maximum number of classes to treat as discrete. Default 10.
        min_effect_size (float, optional): minimum effect size to plot. Default 0.003.
        y_prefix (str, optional): y prefix for plot. Default "Influence of ".
        X_names (List[str], optional): names of predictors. Default None.
        verbose (bool, optional): print progess. Default True.
        individual_preds (bool, optional): whether to use plot each bootstrap. Default True.
        C_vis (np.ndarray, optional): Context bins used to visualize context (n_vis, n_contexts). Default None to construct anew.
        n_vis (int, optional): Number of bins to use to visualize context. Default 1000.
        lower_pct (int, optional): Lower percentile for bootstraps. Default 2.5.
        upper_pct (int, optional): Upper percentile for bootstraps. Default 97.5.
        classification (bool, optional): Whether to exponentiate the effects. Default True.
        C_encoders (List[sklearn.preprocessing.LabelEncoder], optional): encoders for each context. Default None.
        C_means (np.ndarray, optional): means for each context. Default None.
        C_stds (np.ndarray, optional): standard deviations for each context. Default None.
        xlabel_prefix (str, optional): prefix for x label. Default "".
        figname (str, optional): name of figure to save. Default None.

    Returns:
        None
    """
    c_vis = maybe_make_c_vis(C, **kwargs)
    n_vis = c_vis.shape[0]
    # c_names = C.columns.tolist()
    for j in range(C.shape[1]):
        c_j = c_vis.copy()
        c_j[:, :j] = 0.0
        c_j[:, j + 1 :] = 0.0
        (models, _) = model.predict_params(
            c_j, individual_preds=True
        )  # n_bootstraps x n_vis x outcomes x predictors
        homogeneous_effects = np.mean(
            models, axis=1
        )  # n_bootstraps x outcomes x predictors
        heterogeneous_effects = models.copy()
        for i in range(n_vis):
            heterogeneous_effects[:, i] -= homogeneous_effects
        # n_bootstraps x n_vis x outcomes x predictors

        for outcome in range(heterogeneous_effects.shape[2]):
            my_effects = heterogeneous_effects[
                :, :, outcome, :
            ]  # n_bootstraps x n_vis x predictors
            means = np.mean(my_effects, axis=0)  # n_vis x predictors
            my_lowers = np.percentile(my_effects, kwargs.get("lower_pct", 2.5), axis=0)
            my_uppers = np.percentile(my_effects, kwargs.get("upper_pct", 97.5), axis=0)

            x_ticks = None
            x_ticklabels = None
            try:
                x_classes = kwargs["encoders"][j].classes_
                if len(x_classes) <= kwargs.get("max_classes_for_discrete", 10):
                    x_ticks = np.array(list(range(len(x_classes))))
                    if "c_means" in kwargs:
                        x_ticks -= kwargs["c_means"][j]
                    if "c_stds" in kwargs:
                        x_ticks /= kwargs["c_stds"][j]
                    x_ticklabels = x_classes
            except KeyError:
                pass
            for k in range(my_effects.shape[-1]):
                if np.max(heterogeneous_effects[:, k]) > kwargs.get(
                    "min_effect_size", 0.0
                ):
                    simple_plot(
                        c_vis[:, j],
                        means[:, k],
                        x_label=C.columns[j],
                        y_label=f"{kwargs.get('y_prefix', 'Influence of')} {X.columns[k]}",
                        y_lowers=my_lowers[:, k],
                        y_uppers=my_uppers[:, k],
                        x_ticks=x_ticks,
                        x_ticklabels=x_ticklabels,
                        **kwargs,
                    )


def make_grid_mat(observation_mat, n_vis):
    """

    :param observation_mat: defines the domain for each feature.
    :param n_vis:

    returns a matrix of n_vis x n_features that can be used to visualize the effects of the features.

    """
    ar_vis = np.zeros((n_vis, observation_mat.shape[1]))
    for j in range(observation_mat.shape[1]):
        ar_vis[:, j] = np.linspace(
            np.min(observation_mat[:, j]), np.max(observation_mat[:, j]), n_vis
        )
    return ar_vis


def make_c_vis(C, n_vis):
    """

    :param C:
    :param n_vis:

    returns a matrix of n_vis x n_contexts that can be used to visualize the effects of the context variables.

    """
    if isinstance(C, np.ndarray):
        return make_grid_mat(C, n_vis)
    return make_grid_mat(C.values, n_vis)


def maybe_make_c_vis(C, **kwargs):
    """

    :param C:
    :param n_vis:

    returns a matrix of n_vis x n_contexts that can be used to visualize the effects of the context variables.
    if C_vis is supplied, then we use that instead.

    """
    if kwargs.get("C_vis", None) is None:
        if kwargs.get("verbose", True):
            print(
                """Generating datapoints for visualization by assuming the encoder is
            an additive model and thus doesn't require sampling on a manifold.
            If the encoder has interactions, please supply C_vis so that we
            can visualize these effects on the correct data manifold."""
            )
        return make_c_vis(C, kwargs.get("n_vis", 1000))
    return kwargs["C_vis"]
