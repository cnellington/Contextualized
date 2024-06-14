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

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def trim_params(w, thresh=0.2):
    return w * (np.abs(w) > thresh)


def project_to_dag_torch(w, thresh=0.0):
    """
    Project a weight matrix to the closest DAG in Frobenius norm.
    """

    if is_dag(w):
        return w

    w_dag = w.copy()
    # Easy case first: remove diagnoal entries.
    w_dag *= 1 - np.eye(w.shape[0])

    # First, remove edges with weights smaller than the thresh.
    w_dag = trim_params(w_dag, thresh)

    # Sort nodes by magnitude of edges pointing out.
    order = np.argsort(np.abs(w_dag).sum(axis=1))[::-1]

    # Re-order
    w_dag = w_dag[order, :][:, order]

    # Keep only forward edges (i.e. upper triangular part).
    w_dag = np.triu(w_dag)

    # Return to original order
    w_dag = w_dag[np.argsort(order), :][:, np.argsort(order)]

    assert is_dag(w_dag)
    return w_dag


def break_symmetry(w):
    for i in range(w.shape[0]):
        w[i][i] = 0.0
        for j in range(i):
            if np.abs(w[i][j]) > np.abs(w[j][i]):
                w[j][i] = 0.0
            else:
                w[i][j] = 0.0
    return w


def project_to_dag_search(W):
    W = W.copy()
    if ig.Graph.Weighted_Adjacency(W).is_dag():
        return W
    W_mag = np.abs(W)
    W_flat = W_mag.flatten()
    
    # Binary search for the minimum threshold where W is a DAG, O(|E|log|E|)
    weights = np.sort(W_flat)
    low = 0
    mid = 0
    high = len(weights) - 1
    while low < high - 1:
        new_mid = (low + high) // 2
        mid = new_mid
#         print(low, mid, high)
        if ig.Graph.Weighted_Adjacency(W * (W_mag > weights[mid])).is_dag():
            high = mid
        else:
            low = mid
    W_dag = W * (W_mag > weights[high])
    
    # Re-add edges we removed that don't violate the topological order, O(|E|)
    p = len(W_dag)
    weights_i = np.argsort(W_flat)
    toposort = ig.Graph.Weighted_Adjacency(W_dag).topological_sorting()
    toposort_lookup = np.zeros(p)
    for topo_i, topo_node in enumerate(toposort):
        toposort_lookup[topo_node] = topo_i
    for sorted_i in range(high, -1, -1):
        i = weights_i[sorted_i]
        parent_i = i // p
        child_i = i % p
        if toposort_lookup[parent_i] < toposort_lookup[child_i]:
            W_dag[parent_i, child_i] = weights[sorted_i]
    return W_dag
