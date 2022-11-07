"""
Utilities for post-hoc analysis of trained Contextualized models.
"""

import numpy as np
from sklearn.metrics import roc_auc_score as roc


def get_roc(Y_true, Y_pred):
    """Measures ROC. Return np.nan if no valid ROC value."""
    try:
        return roc(Y_true, Y_pred)
    except (IndexError, ValueError):
        return np.nan


def print_acc_by_covars(Y_true, Y_pred, covar_df, **kwargs):
    """
    Prints Accuracy for different data splits with covariates.
    Assume Y_true and Y_pred are np arrays.
    Allows train_idx and test_idx as Boolean locators.
    """
    Y_true = np.squeeze(Y_true)
    Y_pred = np.squeeze(Y_pred)
    for i, covar in enumerate(covar_df.columns):
        my_labels = covar_df.values[:, i]
        if len(set(my_labels)) > kwargs.get("max_classes", 20):
            continue
        if kwargs.get("covar_stds", None) is not None:
            my_labels *= kwargs["covar_stds"][i]
        if kwargs.get("covar_means", None) is not None:
            my_labels += kwargs["covar_means"][i]
        if kwargs.get("covar_encoders", None) is not None:
            try:
                my_labels = kwargs["covar_encoders"][i].inverse_transform(
                    my_labels.astype(int)
                )
            except (AttributeError, TypeError, ValueError):
                pass
        print("=" * 20)
        print(covar)
        print("-" * 10)

        for label in sorted(set(my_labels)):
            label_idxs = my_labels == label
            if (
                kwargs.get("train_idx", None) is not None
                and kwargs.get("test_idx", None) is not None
            ):
                my_train_idx = np.logical_and(label_idxs, kwargs["train_idx"])
                my_test_idx = np.logical_and(label_idxs, kwargs["test_idx"])
                train_roc = get_roc(Y_true[my_train_idx], Y_pred[my_train_idx])
                test_roc = get_roc(Y_true[my_test_idx], Y_pred[my_test_idx])
                print(
                    f"{label}:\t Train ROC: {train_roc:.2f}, Test ROC: {test_roc:.2f}"
                )
            else:
                overall_roc = get_roc(Y_true[label_idxs], Y_pred[label_idxs])
                print(f"{label}:\t ROC: {overall_roc:.2f}")
        print("=" * 20)
