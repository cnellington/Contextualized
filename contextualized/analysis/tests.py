"""
Unit tests for analysis utilities.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from contextualized.analysis import (
	test_each_context
)

from contextualized.easy import ContextualizedRegressor


class TestTestEachContext(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def setUp(self):
		"""
		Shared data setup.
		"""
		n_samples = 1000
		C = np.random.uniform(0, 1, size=(n_samples, 2))
		X = np.random.uniform(0, 1, size=(n_samples, 2))
		beta = np.concatenate([np.ones((n_samples, 1)), C], axis=1)
		Y = np.sum(beta[:, :2] * X, axis=1)

		self.C_train_df = pd.DataFrame(C, columns=['C0', 'C1'])
		self.X_train_df = pd.DataFrame(X, columns=['X0', 'X1'])
		self.Y_train_df = pd.DataFrame(Y, columns=['Y'])

		self.pvals = test_each_context(ContextualizedRegressor, self.C_train_df, self.X_train_df, self.Y_train_df, model_kwargs={'encoder_type': 'mlp', 'layers': 0}, fit_kwargs={'max_epochs': 1, 'learning_rate': 1e-2, 'n_bootstraps': 40})

	def test_output_shape(self):
		"""
		Test that the output shape of the test_each_context function is as expected.
		"""
		expected_shape = (self.C_train_df.shape[1] * self.X_train_df.shape[1] * self.Y_train_df.shape[1], 4)
		self.assertEqual(self.pvals.shape, expected_shape)
	
	def test_valid_pval_range(self):
		"""
		Test that all pvals are in the valid range.
		"""
		self.assertTrue(all(0 <= pval <= 1 for pval in self.pvals['Pvals']))

	def test_expected_significant_pval(self):
		"""
		Test that expected significant pvals are in fact significant.
		"""
		pval_c0_x1 = self.pvals.loc[1, 'Pvals']
		self.assertTrue(pval_c0_x1 < 0.05, "C0 X1 p-value is not significant.")

		other_pvals = self.pvals.drop(1)
		self.assertTrue(all(pval >= 0.05 for pval in other_pvals['Pvals']), "Other p-values are significant.")


if __name__ == '__main__':
	unittest.main()
