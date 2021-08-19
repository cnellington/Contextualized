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
        distribution_ids = np.zeros(N)
        for distribution_id, (mean, cov) in enumerate(zip(means, covs)):
            samples[distribution_id*n:(distribution_id+1)*n] = np.random.multivariate_normal(mean, cov, n)
            contexts[distribution_id*n:(distribution_id+1)*n]= np.repeat(pca.transform([cov.flatten()]), n, axis=0)
            cov_labels[distribution_id*n:(distribution_id+1)*n] = np.repeat([cov], n, axis=0)
            distribution_ids[distribution_id*n:(distribution_id+1)*n] = np.ones(n) * distribution_id
        return contexts, samples, cov_labels, distribution_ids


    def split_tasks(self, C, X):
        n, p = X.shape
        N = n * p ** 2
        C_all = np.repeat(C, p ** 2, axis=0)
        sample_ids = np.repeat(np.arange(C.shape[0]).astype(int), p ** 2)
        Xi = np.zeros((N, 1))
        Xj = np.zeros((N, 1))
        Ti = np.zeros((N, 1))
        Tj = np.zeros((N, 1))
        m = 0
        for k in range(n):
            for i in range(p):
                for j in range(p):
                    Xi[m, 0] = X[k, i]
                    Xj[m, 0] = X[k, j]
                    Ti[m, 0] = i
                    Tj[m, 0] = j
                    m += 1
        return C_all, Ti, Tj, Xi, Xj, sample_ids

if __name__ == "__main__":
    dl = Dataloader(2, 3, 2)
    dl.simulate(10)
