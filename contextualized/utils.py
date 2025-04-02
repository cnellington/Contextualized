"""
Utility functions, including saving/loading of contextualized models.
"""

import torch


def save(model, path):
    """
    Saves model to path.
    :param model:
    :param path:

    """
    with open(path, "wb") as out_file:
        torch.save(model, out_file)


def load(path):
    """
    Loads model from path.
    :param path:

    """
    with open(path, "rb") as in_file:
        model = torch.load(in_file)
    return model


class DummyParamPredictor:
    """
    Predicts Parameters as all zeros.
    """

    def __init__(self, beta_dim, mu_dim):
        self.beta_dim = beta_dim
        self.mu_dim = mu_dim

    def predict_params(self, *args):
        """

        :param *args:

        """
        n = len(args[0])
        return torch.zeros((n, *self.beta_dim)), torch.zeros((n, *self.mu_dim))


class DummyYPredictor:
    """
    Predicts Ys as all zeros.
    """

    def __init__(self, y_dim):
        self.y_dim = y_dim

    def predict_y(self, *args):
        """

        :param *args:

        """
        n = len(args[0])
        return torch.zeros((n, *self.y_dim))
    
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_data(C, X, return_scaler=False):
    """
    Normalize C and X 
    """
    scaler_C = StandardScaler()
    scaler_X = StandardScaler()

    C_norm = scaler_C.fit_transform(C)
    X_norm = scaler_X.fit_transform(X)

    if isinstance(C, pd.DataFrame):
        C_norm = pd.DataFrame(C_norm, columns=C.columns, index=C.index)
    if isinstance(X, pd.DataFrame):
        X_norm = pd.DataFrame(X_norm, columns=X.columns, index=X.index)

    if return_scaler:
        return C_norm, X_norm, scaler_C, scaler_X
    return C_norm, X_norm

# def inverse_transform_preserve_df(scaler, df):
#     X_inv = scaler.inverse_transform(df)
#     if isinstance(df, pd.DataFrame):
#         return pd.DataFrame(X_inv, columns=df.columns, index=df.index)
#     return X_inv

def inverse_transform_preserve_df(scaler, df, reference=None):
    X_inv = scaler.inverse_transform(df)

    if isinstance(df, pd.DataFrame):
        return pd.DataFrame(X_inv, columns=df.columns, index=df.index)

    if isinstance(reference, pd.DataFrame):
        return pd.DataFrame(X_inv, columns=reference.columns, index=reference.index)

    return X_inv
