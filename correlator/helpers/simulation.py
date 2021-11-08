import numpy as np
from sklearn.decomposition import PCA


class GaussianSimulator:
    """
    Generate samples with known correlation
    """
    def __init__(self, p, k, c, seed=None):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        # Distribution generation parameters
        self.p = p
        self.k = k
        self.c = c
        # Distribution parameters
        self.sigmas = None
        self.mus = None
        self.contexts = None
        self._build()

    def _build(self):
        """
        Generate parameters for k p-variate gaussians with context
        """
        self.mus = np.zeros((self.k, self.p))
        self.sigmas = np.zeros((self.k, self.p, self.p))
        self.contexts = np.zeros((self.k, self.c))
        for i in range(self.k):
            self.mus[i] = np.zeros(self.p)
            self.contexts[i] = np.random.random((self.c,))
            # TODO: generate sigma using eigen decomposition
            sigma = np.random.random((self.p, self.p)) * 2 - 1
            sigma = sigma @ sigma.T
            self.sigmas[i] = sigma

    def gen_samples(self, k_n):
        """
        Generate full datasets of samples
        """
        # Sample each distribution
        n = self.k * k_n
        C = np.zeros((n, self.c))
        X = np.zeros((n, self.p))
        for i in range(self.k):
            mu, sigma, context = self.mus[i], self.sigmas[i], self.contexts[i]
            sample = np.random.multivariate_normal(mu, sigma, k_n)
            C[i * k_n:(i + 1) * k_n] = context
            X[i * k_n:(i + 1) * k_n] = sample
        return C, X
