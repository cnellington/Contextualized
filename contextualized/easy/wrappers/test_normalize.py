"""
test for issue #250

"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("../../.."))

from contextualized.easy import ContextualizedRegressor

# generate random un-normalized C and X
np.random.seed(42)
C = np.random.rand(100, 5) * 10
X = np.random.rand(100, 3) * 10
Y = np.random.rand(100, 1)

# split training/testing data
C_train, C_test = C[:80], C[80:]
X_train, X_test = X[:80], X[80:]
Y_train, Y_test = Y[:80], Y[80:]

# initialized with normalize=True
model = ContextualizedRegressor(normalize=True)

# train model on training data
model.fit(C_train, X_train, Y_train)

assert np.allclose(
    model.scaler_C.mean_, np.mean(C_train, axis=0), atol=1e-1
), "C normalization failed!"
assert np.allclose(
    model.scaler_C.scale_, np.std(C_train, axis=0), atol=1e-1
), "C normalization failed!"
assert np.allclose(
    model.scaler_X.mean_, np.mean(X_train, axis=0), atol=1e-1
), "X normalization failed!"
assert np.allclose(
    model.scaler_X.scale_, np.std(X_train, axis=0), atol=1e-1
), "X normalization failed!"
print("\n✅ fit() successfully normalized C_train and X_train!")
# predict on test data by trained model
Y_pred = model.predict(C_test, X_test)

assert Y_pred is not None, "predict() failed!"
assert Y_pred.shape == Y_test.shape, "predict() has wrong output shape!"

print(
    "\n✅ predict() ran successfully, un-normalized C_test and X_test were correctly treated!"
)

print("predicted value:", Y_pred)
