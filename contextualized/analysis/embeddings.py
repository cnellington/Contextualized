"""
Utilities for plotting embeddings of fitted Contextualized models.
"""

from typing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


def convert_to_one_hot(col: Collection[Any]) -> Tuple[np.ndarray, List[Any]]:
    """
    Converts a categorical variable to a one-hot vector.

    Args:
        col (Collection[Any]): The categorical variable.

    Returns:
        Tuple[np.ndarray, List[Any]]: The one-hot vector and the possible values.
    """
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals


def plot_embedding_for_all_covars(
    reps: np.ndarray,
    covars_df: pd.DataFrame,
    covars_stds: np.ndarray = None,
    covars_means: np.ndarray = None,
    covars_encoders: List[Callable] = None,
    **kwargs,
) -> None:
    """
    Plot embeddings of representations for all covariates in a Pandas dataframe.

    Args:
        reps (np.ndarray): Embeddings of shape (n_samples, n_dims).
        covars_df (pd.DataFrame): DataFrame of covariates.
        covars_stds (np.ndarray, optional): Standard deviations of covariates. Defaults to None.
        covars_means (np.ndarray, optional): Means of covariates. Defaults to None.
        covars_encoders (List[LabelEncoder], optional): Encoders for covariates. Defaults to None.
        kwargs: Keyword arguments for plotting.

    Returns:
        None
    """
    for i, covar in enumerate(covars_df.columns):
        my_labels = covars_df.iloc[:, i].values
        if covars_stds is not None:
            my_labels *= covars_stds
        if covars_means is not None:
            my_labels += covars_means
        if covars_encoders is not None:
            my_labels = covars_encoders[i].inverse_transform(my_labels.astype(int))
        if kwargs.get("dithering_pct", 0.0) > 0:
            reps[:, 0] += np.random.normal(
                0, kwargs["dithering_pct"] * np.std(reps[:, 0]), size=reps[:, 0].shape
            )
            reps[:, 1] += np.random.normal(
                0, kwargs["dithering_pct"] * np.std(reps[:, 1]), size=reps[:, 1].shape
            )
        try:
            plot_lowdim_rep(
                reps[:, :2],
                my_labels,
                cbar_label=covar,
                **kwargs,
            )
        except TypeError:
            print(f"Error with covar {covar}")


def plot_lowdim_rep(
    low_dim: np.ndarray,
    labels: np.ndarray,
    max_classes_for_discrete: int = 10,
    figsize: Tuple[int, int] = (12, 12),
    min_samples: int = 0,
    alpha: float = 1.0,
    plot_nan: bool = True,
    xlabel: str = "X",
    xlabel_fontsize: int = 48,
    ylabel: str = "Y",
    ylabel_fontsize: int = 48,
    title: str = "",
    title_fontsize: int = 52,
    cbar_label: Optional[str] = None,
    cbar_fontsize: int = 32,
    figname: Optional[str] = None,
):
    """
    Plot a low-dimensional representation of a dataset.

    Args:
        low_dim (np.ndarray): Low-dimensional representation of shape (n_samples, 2).
        labels (np.ndarray): Labels of shape (n_samples,).
        max_classes_for_discrete (int, optional): Maximum number of classes to treat labels as discrete. Default is 10.
        figsize (tuple, optional): Size of the figure. Default is (12, 12).
        min_samples (int, optional): Minimum number of samples required to include a class. Default is 0.
        alpha (float, optional): Alpha blending value for scatter plot. Default is 1.0.
        plot_nan (bool, optional): Whether to plot NaN values in a separate color. Default is True.
        xlabel (str, optional): Label for the x-axis. Default is 'X'.
        xlabel_fontsize (int, optional): Font size for x-axis label. Default is 48.
        ylabel (str, optional): Label for the y-axis. Default is 'Y'.
        ylabel_fontsize (int, optional): Font size for y-axis label. Default is 48.
        title (str, optional): Title of the plot. Default is an empty string.
        title_fontsize (int, optional): Font size for the title. Default is 52.
        cbar_label (str, optional): Label for the colorbar. Default is None.
        cbar_fontsize (int, optional): Font size for the colorbar label. Default is 32.
        figname (str, optional): If provided, saves the figure to this name (with .pdf extension). Default is None.

    Returns:
        None
    """

    if len(set(labels)) < max_classes_for_discrete:  # discrete labels
        discrete = True
        cmap = plt.cm.jet
    else:
        discrete = False
        tag = labels
        norm = None
        cmap = plt.cm.coolwarm
    fig = plt.figure(figsize=figsize)
    if discrete:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", [cmap(i) for i in range(cmap.N)], cmap.N
        )
        tag, tag_names = convert_to_one_hot(labels)
        order = np.argsort(tag_names)
        tag_names = np.array(tag_names)[order]
        tag = np.array([list(order).index(int(x)) for x in tag])
        good_tags = [
            np.sum(tag == i) > min_samples
            for i in range(len(tag_names))
        ]
        tag_names = np.array(tag_names)[good_tags]
        good_idxs = np.array([good_tags[int(tag[i])] for i in range(len(tag))])
        tag = tag[good_idxs]
        tag, _ = convert_to_one_hot(tag)
        bounds = np.linspace(0, len(tag_names), len(tag_names) + 1)
        try:
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        except ValueError:
            print(
                "Not enough values for a colorbar (needs at least 2 values), quitting."
            )
            return
        plt.scatter(
            low_dim[good_idxs, 0],
            low_dim[good_idxs, 1],
            c=tag,
            alpha=alpha,
            s=100,
            cmap=cmap,
            norm=norm,
        )
    else:
        # plot valid points first
        mask_nan = np.isnan(labels)
        mask_valid = ~mask_nan
        plt.scatter(
            low_dim[mask_valid, 0],
            low_dim[mask_valid, 1],
            c=labels[mask_valid],
            alpha=alpha,
            s=100,
            cmap=cmap,
        )

        # then users decide whether or not to plot NaN points
        if mask_nan.any() and plot_nan:
            plt.scatter(
                low_dim[mask_nan, 0],
                low_dim[mask_nan, 1],
                c="green",  # For continuous labels, colorbar is coolwarm, so green is a good choice to show NaN
                marker="s",
                alpha=alpha,
                s=100,
            )

    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=title_fontsize)

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    if discrete:
        color_bar = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            spacing="proportional",
            ticks=bounds[:-1] + 0.5,  # boundaries=bounds,
            format="%1i",
        )

        # enhancement of the above code, accepting strings as labels
        try:
            tag_labels = np.round(tag_names)
        except TypeError:
            tag_labels = [str(x) for x in tag_names]
        color_bar.ax.set(yticks=bounds[:-1] + 0.5, yticklabels=tag_labels)

    else:
        color_bar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, format="%.1f")
        if mask_nan.any() and plot_nan:
            nan_legend = Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="NaN",
                markerfacecolor="green",
                markersize=10,
                alpha=1,
            )
            plt.legend(handles=[nan_legend], loc="best")

    if cbar_label is not None:
        color_bar.ax.set_ylabel(
            cbar_label, fontsize=cbar_fontsize
        )
    if figname is not None:
        plt.savefig(f"{figname}.pdf", dpi=300, bbox_inches="tight")
