"""
Utility functions for 'easy'
"""

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


def inverse_transform_preserve_df(scaler, df, reference=None):
    X_inv = scaler.inverse_transform(df)

    if isinstance(df, pd.DataFrame):
        return pd.DataFrame(X_inv, columns=df.columns, index=df.index)

    if isinstance(reference, pd.DataFrame):
        return pd.DataFrame(X_inv, columns=reference.columns, index=reference.index)

    return X_inv