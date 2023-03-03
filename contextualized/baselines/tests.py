"""
Unit tests for baseline models.
"""

import unittest
import numpy as np

from contextualized.baselines import CorrelationNetwork, MarkovNetwork, BayesianNetwork, GroupedNetworks


class TestBaselineNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBaselineNetworks, self).__init__(*args, **kwargs)

    def setUp(self):
        self.n, self.x_dim = 100, 20
        self.labels = np.random.randint(0, 5, (self.n,))
        self.X = np.random.uniform(-1, 1, (self.n, self.x_dim))

    def test_correlation_network(self):
        corr = CorrelationNetwork().fit(self.X)
        assert corr.predict(self.n).shape == (self.n, self.x_dim, self.x_dim)
        assert corr.measure_mses(self.X).mean() < 1.0

    def test_grouped_corr_network(self):
        grouped_corr = GroupedNetworks(CorrelationNetwork).fit(self.X, self.labels)
        assert grouped_corr.predict(self.labels).shape == (self.n, self.x_dim, self.x_dim)
        assert grouped_corr.measure_mses(self.X, self.labels).mean() < 1.0

    def test_markov_network(self):
        mark = MarkovNetwork().fit(self.X)
        assert mark.predict(self.n).shape == (self.n, self.x_dim, self.x_dim)
        assert mark.measure_mses(self.X).mean() < 1.0

    def test_grouped_markov_network(self):
        grouped_mark = GroupedNetworks(MarkovNetwork).fit(self.X, self.labels)
        grouped_mark.predict(self.labels)
        assert grouped_mark.measure_mses(self.X, self.labels).mean() < 1.0

    def test_bayesian_network(self):
        dag = BayesianNetwork().fit(self.X)
        assert dag.predict(self.n).shape == (self.n, self.x_dim, self.x_dim)
        assert dag.measure_mses(self.X).mean() < 1.0

    def test_grouped_bayesian_network(self):
        grouped_dag = GroupedNetworks(BayesianNetwork).fit(self.X, self.labels)
        assert grouped_dag.predict(self.labels).shape == (self.n, self.x_dim, self.x_dim)
        assert grouped_dag.measure_mses(self.X, self.labels).mean() < 1.0


if __name__ == "__main__":
    unittest.main()
