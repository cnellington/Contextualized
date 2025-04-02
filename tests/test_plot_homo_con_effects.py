import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from contextualized.easy import ContextualizedRegressor
from contextualized.analysis.effects import plot_homogeneous_context_effects


def test_plot_homogeneous_context_effects():
    
    # make up some data
    np.random.seed(42)
    C = np.random.rand(100, 2)
    X = np.random.rand(100, 3)
    Y = X @ np.array([1.0, -0.5, 2.0]) + C[:, 0] + np.random.randn(100)
    Y = Y.reshape(-1, 1)

    C_df = pd.DataFrame(C, columns=["context_1", "context_2"])  # original C

    # construct model and train
    model = ContextualizedRegressor(normalize=True, max_epochs=3)
    model.fit(C_df, X, Y)

    # plot test for original C
    print("🔍 Plotting with original (normalized internally) C...")
    plot_homogeneous_context_effects(model, C_df, inverse_transform=True)

    # # plot test for normalized C
    # print("🔍 Plotting with manually scaled C and original_C override...")
    # C_scaled = model.scaler_C.transform(C_df)
    # plot_homogeneous_context_effects(model, C_scaled, inverse_transform=True, original_C=C_df) # C_df still needed for column names

if __name__ == "__main__":
    test_plot_homogeneous_context_effects()
