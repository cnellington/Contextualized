import torch
import igraph as ig
import numpy as np


def dag_pred_with_factors(X, W, P):
    """
    Pass observation X through a linear SEM low-dim network W with factors P.
    """
    # For linear SEM with P factors, X is [n, P], W is [P, P], P is [d, P]:
    # X = XP^TWP/||P||_1
    # where the final normalization is performed over the rows of P.
    return torch.matmul(torch.matmul(torch.matmul(X, P.T), W), (P.T / P.sum(axis=1)).T)


dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
dag_pred_np = lambda x, w: np.matmul(x[:, np.newaxis, :], w).squeeze()


def simulate_linear_sem(W, n_samples, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n_samples)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=scale, size=n_samples)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=scale, size=n_samples)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "uniform":
            z = np.random.uniform(low=-scale, high=scale, size=n_samples)
            # x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == "logistic":
            # x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            x = np.random.binomial(1, 1 / (1 + np.exp(-(np.matmul(X, w))))) * 1.0
        elif sem_type == "poisson":
            # x = np.random.poisson(np.exp(X @ w)) * 1.0
            x = np.random.poisson(np.exp(np.matmul(X, w))) * 1.0
        else:
            raise ValueError("unknown sem type")
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or has length d")
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError("W must be a DAG")
    if np.isinf(n_samples):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.matmul(np.diag(scale_vec), np.linalg.inv(np.eye(d) - W))
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n_samples, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def break_symmetry(w):
    for i in range(w.shape[0]):
        w[i][i] = 0.0
        for j in range(i):
            if np.abs(w[i][j]) > np.abs(w[j][i]):
                w[j][i] = 0.0
            else:
                w[i][j] = 0.0
    return w


# w is the weighted adjacency matrix
def project_to_dag_torch(w):
    if is_dag(w):
        return w, 0.0

    w_dag = w.copy()
    w_dag = break_symmetry(w_dag)

    vals = sorted(list(set(np.abs(w_dag).flatten())))
    low = 0
    high = len(vals) - 1

    def binary_search(arr, low, high, w):  # low and high are indices
        # Check base case
        if high == low:
            return high
        if high > low:
            mid = (high + low) // 2
            if mid == 0:
                return mid
            result = trim_params(w, arr[mid])
            if is_dag(result):
                result2 = trim_params(w, arr[mid - 1])
                if is_dag(result2):  # middle value is too high.  go lower.
                    return binary_search(arr, low, mid - 1, w)
                else:
                    return mid  # found it
            else:  # middle value is too low.  go higher.
                return binary_search(arr, mid + 1, high, w)
        else:
            # Element is not present in the array
            print("this should be impossible")
            return -1

    idx = binary_search(vals, low, high, w_dag) + 1
    thresh = vals[idx]
    w_dag = trim_params(w_dag, thresh)

    # Now add back in edges with weights smaller than the thresh that don't violate DAG-ness.
    # want a list of edges (i, j) with weight in decreasing order.
    all_vals = np.abs(w_dag).flatten()
    idxs_sorted = reversed(np.argsort(all_vals))
    for idx in idxs_sorted:
        i = idx // w_dag.shape[1]
        j = idx % w_dag.shape[1]
        if np.abs(w[i][j]) > thresh:  # already retained
            continue
        w_dag[i][j] = w[i][j]
        if not is_dag(w_dag):
            w_dag[i][j] = 0.0

    assert is_dag(w_dag)
    return w_dag, thresh


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def trim_params(w, thresh=0.2):
    return w * (np.abs(w) > thresh)
