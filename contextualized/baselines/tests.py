"""
Unit tests for baseline models.
"""

import unittest
import numpy as np

from contextualized.baselines import (
    CorrelationNetwork,
    MarkovNetwork,
    BayesianNetwork,
    GroupedNetworks,
)


class TestBaselineNetworks(unittest.TestCase):
    """
    Test that the baseline networks can be fit and predict the correct shape.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Shared data setup.
        """
        self.n_samples, self.x_dim = 100, 20
        self.labels = np.random.randint(0, 5, (self.n_samples,))
        self.X = np.random.uniform(-1, 1, (self.n_samples, self.x_dim))

    def test_correlation_network(self):
        """
        Test that the correlation network can be fit and predicts the correct shape.
        """
        corr = CorrelationNetwork().fit(self.X)
        assert corr.predict(self.n_samples).shape == (self.n_samples, self.x_dim, self.x_dim)
        assert corr.measure_mses(self.X).mean() < 1.0

    def test_grouped_corr_network(self):
        """
        Test that the grouped correlation network can be fit and predicts the correct shape.
        """
        grouped_corr = GroupedNetworks(CorrelationNetwork).fit(self.X, self.labels)
        assert grouped_corr.predict(self.labels).shape == (
            self.n_samples,
            self.x_dim,
            self.x_dim,
        )
        assert grouped_corr.measure_mses(self.X, self.labels).mean() < 1.0

    def test_markov_network(self):
        """
        Test that the markov network can be fit and predicts the correct shape.
        """
        mark = MarkovNetwork().fit(self.X)
        assert mark.predict(self.n_samples).shape == (self.n_samples, self.x_dim, self.x_dim)
        assert mark.measure_mses(self.X).mean() < 1.0

    def test_grouped_markov_network(self):
        """
        Test that the grouped markov network can be fit and predicts the correct shape.
        """
        grouped_mark = GroupedNetworks(MarkovNetwork).fit(self.X, self.labels)
        grouped_mark.predict(self.labels)
        assert grouped_mark.measure_mses(self.X, self.labels).mean() < 1.0

    def test_bayesian_network(self):
        """
        Test that the bayesian network can be fit and predicts the correct shape.
        """
        dag = BayesianNetwork().fit(self.X)
        assert dag.predict(self.n_samples).shape == (self.n_samples, self.x_dim, self.x_dim)
        assert dag.measure_mses(self.X).mean() < 1.0

    def test_grouped_bayesian_network(self):
        """
        Test that the grouped bayesian network can be fit and predicts the correct shape.
        """
        grouped_dag = GroupedNetworks(BayesianNetwork).fit(self.X, self.labels)
        assert grouped_dag.predict(self.labels).shape == (
            self.n_samples,
            self.x_dim,
            self.x_dim,
        )
        assert grouped_dag.measure_mses(self.X, self.labels).mean() < 1.0


if __name__ == "__main__":
    unittest.main()
