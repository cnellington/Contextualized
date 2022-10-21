"""
Helper functions for simulations and experiments.
"""
import numpy as np
from sklearn.decomposition import PCA


class GaussianSimulator:
    """
    Generate samples from Gaussian distributions with known parameters
    """

    def __init__(self, p, k, c, ctype="self", seed=None, sigmas=None, mus=None):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        self.ctype = ctype
        # Distribution generation parameters
        self.p = p
        self.k = k
        self.c = c
        # Distribution parameters
        self.sigmas = sigmas
        self.mus = mus
        self.vars = None
        self.betas = None
        self.rhos = None  # rho^2, Pearson's correlation coefficient squared
        self.contexts = None
        self._build()

    def _build(self):
        """
        Generate parameters for k p-variate gaussians with context
        """
        new_mus = self.mus is None
        new_sigmas = self.sigmas is None
        self.mus = np.zeros((self.k, self.p)) if new_mus else self.mus
        self.sigmas = np.zeros((self.k, self.p, self.p)) if new_sigmas else self.sigmas
        self.contexts = np.zeros((self.k, self.c))
        self.vars = np.zeros((self.k, self.p))
        self.betas = np.zeros((self.k, self.p, self.p))
        self.rhos = np.zeros((self.k, self.p, self.p))
        # Parameterize Gaussian models
        for i in range(self.k):
            if new_mus:
                self.mus[i] = np.random.uniform(-self.p, self.p, self.p)
            # TODO: generate sigma using eigen decomposition
            if new_sigmas:
                sigma = np.random.random((self.p, self.p)) * 2 - 1
                sigma = sigma @ sigma.T
                self.sigmas[i] = sigma
            self.vars[i] = self.sigmas[i].diagonal()
            vars_tiled = np.tile(self.vars[i], (self.p, 1)).T
            self.betas[i] = self.sigmas[i] / vars_tiled  # beta[i,j] = beta_{i-->j}
            self.rhos[i] = np.power(self.sigmas[i], 2) / (vars_tiled * vars_tiled.T)
        # Build contexts
        if self.ctype == "uniform":
            for i in range(self.k):
                self.contexts[i] = np.random.random((self.c,))
        elif self.ctype == "pca":
            gaussian_reps = np.concatenate(
                (self.sigmas, self.mus[:, :, np.newaxis]), axis=-1
            )
            gaussian_reps = gaussian_reps.reshape((self.k, self.p * (self.p + 1)))
            self.contexts = PCA(n_components=self.c).fit_transform(gaussian_reps)
        elif self.ctype == "self":
            gaussian_reps = np.concatenate(
                (self.sigmas, self.mus[:, :, np.newaxis]), axis=-1
            )
            self.contexts = gaussian_reps.reshape((self.k, self.p * (self.p + 1)))
            self.c = self.contexts.shape[-1]

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
            C[i * k_n : (i + 1) * k_n] = context
            X[i * k_n : (i + 1) * k_n] = sample
        return C, X


f_context = lambda z: z + np.random.normal(0, 0.01, z.shape)


def f_sigma(z):
    z_tiled = np.tile(z, (z.shape[-1], 1))
    A = z_tiled + z_tiled.T
    # TODO: maybe make this more controlled
    sigma = A.T @ A  # from spectral decomposition
    return sigma


f_mu = lambda z: np.zeros(z.shape)


class GraphicalSimulator:
    """
    Generate samples from a Graphical model (C <- Z -> Sigma/Mu -> X) with known correlation
    """

    def __init__(
        self, k, p, f_context=f_context, f_sigma=f_sigma, f_mu=f_mu, seed=None
    ):
        self.seed = seed if seed is not None else np.random.randint(1e9)
        np.random.seed(self.seed)
        # Distribution generation parameters
        self.p = p
        self.k = k
        self.c = p
        self.f_context = f_context
        self.f_sigma = f_sigma
        self.f_mu = f_mu
        # Distribution parameters
        self.Z = np.random.uniform(-1, 1, (k, p))
        self.contexts = None
        self.sigmas = None
        self.mus = None
        self.vars = None
        self.betas = None
        self.rhos = None  # rho^2, Pearson's correlation coefficient squared
        self._build()

    def _build(self):
        """
        Generate parameters for n p-variate gaussians with context
        """
        self.contexts = np.zeros((self.k, self.p))
        self.sigmas = np.zeros((self.k, self.p, self.p))
        self.mus = np.zeros((self.k, self.p))
        self.vars = np.zeros((self.k, self.p))
        self.betas = np.zeros((self.k, self.p, self.p))
        self.rhos = np.zeros((self.k, self.p, self.p))
        # Parameterize Gaussian models
        for i in range(self.k):
            Z_i = self.Z[i]
            self.contexts[i] = self.f_context(Z_i)
            self.sigmas[i] = self.f_sigma(Z_i)
            self.mus[i] = self.f_mu(Z_i)
            self.vars[i] = self.sigmas[i].diagonal()
            vars_tiled = np.tile(self.vars[i], (self.p, 1)).T
            self.betas[i] = self.sigmas[i] / vars_tiled  # beta[i,j] = beta_{i-->j}
            self.rhos[i] = np.power(self.sigmas[i], 2) / (vars_tiled * vars_tiled.T)

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
            C[i * k_n : (i + 1) * k_n] = context
            X[i * k_n : (i + 1) * k_n] = sample
        return C, X
