"""
Caleb Ellington
8/3/21

Generates data (C, X, Sigma)
C: Context profile, loosely related to X
X: Multivariate gaussian with true correlation values in Sigma
"""

import numpy as np
from sklearn.decomposition import PCA


class Dataloader:

    def __init__(self, k, p, c, seed=1):
        assert c <= min(p*p, k)
        self.k = k
        self.p = p
        self.c = c
        self.seed = seed
        np.random.seed(self.seed)

    def simulate(self, n):
        # Generate k covariance matrices with n samples each
        N = self.k * n
        means = []
        covs = []
        for _ in range(self.k):
            mean = np.zeros(self.p)
            cov = np.random.rand(self.p, self.p)
            cov = cov @ cov.T
            covs.append(cov)
            means.append(mean)
        context_full = np.copy(covs).reshape(self.k, self.p ** 2)
        pca = PCA(n_components=self.c)
        pca.fit(context_full)

        samples = np.zeros((N, self.p))
        contexts = np.zeros((N, self.c))
        cov_labels = np.zeros((N, self.p, self.p))
        task_ids = np.zeros(N)
        for task_id, (mean, cov) in enumerate(zip(means, covs)):
            samples[task_id*n:(task_id+1)*n] = np.random.multivariate_normal(mean, cov, n)
            contexts[task_id*n:(task_id+1)*n]= np.repeat(pca.transform([cov.flatten()]), n, axis=0)
            cov_labels[task_id*n:(task_id+1)*n] = np.repeat([cov], n, axis=0)
            task_ids[task_id*n:(task_id+1)*n] = np.ones(n) * task_id
        return contexts, samples, cov_labels, task_ids


if __name__ == "__main__":
    dl = Dataloader(2, 3, 2)
    dl.simulate(10)
