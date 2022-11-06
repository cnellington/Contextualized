"""
Utilities for plotting embeddings of fitted Contextualized models.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from contextualized.analysis import utils


def plot_embedding_for_all_covars(
    reps, covars_df, covars_stds=None, covars_means=None, covars_encoders=None, **kwargs
):
    """
    Plot embeddings of representations for all covariates in a Pandas dataframe.
    :param reps:
    :param covars_df:
    :param covars_stds: Used to project back to readable values. (Default value = None)
    :param covars_means: Used to project back to readable values. (Default value = None)
    :param covars_encoders: Used to project back to readable values. (Default value = None)
    :param kwargs: Keyword arguments for plotting.
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
    low_dim,
    labels,
    **kwargs,
):
    """

    :param low_dim:
    :param labels:
    :param kwargs:
        Keyword arguments.

    """

    if len(set(labels)) < kwargs.get("max_classes_for_discrete", 10):  # discrete labels
        discrete = True
        cmap = plt.cm.jet
    else:
        discrete = False
        tag = labels
        norm = None
        cmap = plt.cm.coolwarm
    fig = plt.figure(figsize=kwargs.get("figsize", (12, 12)))
    if discrete:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", [cmap(i) for i in range(cmap.N)], cmap.N
        )
        tag, tag_names = utils.convert_to_one_hot(labels)
        order = np.argsort(tag_names)
        tag_names = np.array(tag_names)[order]
        tag = np.array([list(order).index(int(x)) for x in tag])
        good_tags = [
            np.sum(tag == i) > kwargs.get("min_samples", 0)
            for i in range(len(tag_names))
        ]
        tag_names = np.array(tag_names)[good_tags]
        good_idxs = np.array([good_tags[int(tag[i])] for i in range(len(tag))])
        tag = tag[good_idxs]
        tag, _ = utils.convert_to_one_hot(tag)
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
            alpha=kwargs.get("alpha", 1.0),
            s=100,
            cmap=cmap,
            norm=norm,
        )
    else:
        plt.scatter(
            low_dim[:, 0],
            low_dim[:, 1],
            c=labels,
            alpha=kwargs.get("alpha", 1.0),
            s=100,
            cmap=cmap,
        )
    plt.xlabel(kwargs.get("xlabel", "X"), fontsize=kwargs.get("xlabel_fontsize", 48))
    plt.ylabel(kwargs.get("ylabel", "Y"), fontsize=kwargs.get("ylabel_fontsize", 48))
    plt.xticks([])
    plt.yticks([])
    plt.title(kwargs.get("title", ""), fontsize=kwargs.get("title_fontsize", 52))

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
        try:
            color_bar.ax.set(yticks=bounds[:-1] + 0.5, yticklabels=np.round(tag_names))
        except ValueError:
            color_bar.ax.set(yticks=bounds[:-1] + 0.5, yticklabels=tag_names)
    else:
        color_bar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, format="%.1f")
    if kwargs.get("cbar_label", None) is not None:
        color_bar.ax.set_ylabel(
            kwargs["cbar_label"], fontsize=kwargs.get("cbar_fontsize", 32)
        )
    if "figname" in kwargs:
        plt.savefig(f"{kwargs['figname']}.pdf", dpi=300, bbox_inches="tight")
