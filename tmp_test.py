import numpy as np

C = np.random.normal(0, 1, (100, 2))
X = np.random.normal(0, 1, (100, 2))
Y = np.random.normal(0, 1, 100)

from contextualized.easy import ContextualizedRegressor

model = ContextualizedRegressor(encoder_type='linear')

model.fit(C, X, Y)
