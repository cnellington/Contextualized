import numpy as np
import os
import igraph as ig

from sklearn.decomposition import PCA
from contextualized.dags.notmad_helpers import utils
from contextualized.dags.notmad_helpers import graph_utils


def l2_dist(dag1, dag2):
    dist_vec = (dag1 - dag2).flatten()
    return dist_vec.T @ dist_vec


def gen_data(data_params):
    if data_params["simulation_type"] == "archetypes":
        W_dict, C_dict = gen_archetypes(data_params["d"], data_params["n_edges"],
            data_params["n_c"], data_params["k_true"],
            graph_type=data_params["graph_type"], ensure_convex=data_params["ensure_convex"])
        sample_loadings, W, C, X = gen_samples(
            W_dict, C_dict, n=data_params["n"], n_i=data_params["n_i"], n_mix=data_params["n_mix"],
            sem_type=data_params["sem_type"], noise_scale=0.01)
        if data_params['context_snr'] > 0 and data_params['context_snr'] < 1.0:
            for i in range(data_params["n_c"]):
                C[:, i] += np.random.normal(0.,
                                            (1./data_params["context_snr"] - 1)*np.var(C[:, i]),
                                            size=C[:, i].shape)
    elif data_params["simulation_type"] == "clusters":
        W_dict, C_dict = gen_archetypes(data_params["d"], data_params["n_edges"],
            data_params["n_c"], data_params["k_true"],
#             data_params["n_c"], data_params["k_true"] + data_params["n_mix"],  # Use separate mixing archetypes?
            graph_type=data_params["graph_type"], min_radius=data_params["arch_min_radius"],
            ensure_convex=data_params["ensure_convex"])
        W_arch = W_dict[:data_params["k_true"]]
#         W_mix = W_dict[-data_params["n_mix"]:]  # Use separate mixing archetypes?
        W_mix = np.copy(W_arch)
        W, C, X = gen_cluster_samples(W_arch, W_mix, n=data_params["n"], n_i=data_params["n_i"], 
            radius=data_params["cluster_max_radius"], sem_type=data_params["sem_type"], 
            noise_scale=0.01)
        if data_params['context_snr'] > 0 and data_params['context_snr'] < 1.0:
            for i in range(data_params["k_true"] * 2):
                C[:, i] += np.random.normal(0.,
                                            (1./data_params["context_snr"] - 1)*np.var(C[:, i]),
                                            size=C[:, i].shape)
    else:
        W, C, X = gen_samples_no_archs(data_params["n"], data_params["d"],
            data_params["n_edges"], data_params["n_i"], data_params["n_c"],
            c_signal_noise=data_params["context_snr"], graph_type=data_params["graph_type"],
            sem_type=data_params["sem_type"])
        W_dict, C_dict = None, None
    return W, C, X, W_dict, C_dict

def gen_cluster_samples(W_k, W_mix, n, n_i, radius=1, sem_type='gauss', noise_scale=0.1):
    # Generate n samples from k archetypes
    (k, d, _) = W_k.shape
    (k_mix, d, _) = W_mix.shape
    n_c = W_k.shape[0] + W_mix.shape[0]

#     subtypes = np.zeros((n, n_c))
    W_n = np.zeros((n, d, d))
    c_n = np.zeros((n, n_c))
    X_n = np.zeros((n, n_i, d))
    for i in range(n):
        print(f'Generating sample {i}', end='\r')
        # TODO: Principled way to make sparse mixtures
        finished = False
        while not finished:
            weights = np.random.uniform(-radius, radius, k_mix)
            k_i = np.random.choice(k)
            if np.sum(weights) < 1e-3:
                continue
            
            W_i = W_k[k_i] + np.tensordot(weights, W_mix, axes=1)
            c_i = np.zeros(n_c)
            c_i[k_i] = 1
            c_i[-k_mix:] = weights
#             try:
#                 X_i = simulate_linear_sem(W_i, n_i, sem_type, noise_scale=noise_scale)
#             except ValueError: # mixture may not be a DAG
            W_i = graph_utils.project_to_dag(W_i)[0]
            X_i = simulate_linear_sem(W_i, n_i, sem_type, noise_scale=noise_scale)
            X_n[i] = X_i
#             subtypes[i] = weights
            W_n[i] = W_i
            c_n[i] = c_i
            finished = True
    print()
    return W_n, c_n, X_n


def gen_archetypes(d=8, s0=8, n_c=20, k=4, graph_type='ER', min_radius=0, ensure_convex=False):
    # Create network and epigenetic archetypes
    W_k = np.ones((k, d, d))
    c_k = np.ones((k, n_c))
    if ensure_convex:
        # Ensure archetypes define a convex set of DAGs
        while not graph_utils.is_dag(np.sum(W_k, axis=0)):
            for j in range(k):
                B_true = simulate_dag(d, s0, graph_type)
                while not np.sum(B_true) == s0:
                    B_true = simulate_dag(d, s0, graph_type)
                W_k[j] = simulate_parameter(B_true)
                c_k[j] = simulate_context(n_c)
    else:
        for j in range(k):
            B_true = simulate_dag(d, s0, graph_type)
            while not graph_utils.is_dag(B_true):
                B_true = simulate_dag(d, s0, graph_type)
                dists = np.array([l2_dist(B_true, W_i) for W_i in W_k])
            W_param = simulate_parameter(B_true)
            dists = np.array([l2_dist(W_param, W_i) for W_i in W_k])
            while not (dists > min_radius).all():  # min_radius should be <600
                W_param = simulate_parameter(B_true)
                dists = np.array([l2_dist(W_param, W_i) for W_i in W_k])
            W_k[j] = W_param
            c_k[j] = simulate_context(n_c)
    return W_k, c_k


def simulate_context(n_c):
    return np.random.uniform(0, 1, n_c)


def gen_samples(W_k, c_k, n, n_i, n_mix=2, sem_type='gauss', noise_scale=0.1):
    # Generate n samples from k archetypes
    assert (c_k.shape[0] == W_k.shape[0])
    (k, d, _) = W_k.shape
    (k, n_c) = c_k.shape

    subtypes = np.zeros((n, k))
    W_n = np.zeros((n, d, d))
    c_n = np.zeros((n, n_c))
    X_n = np.zeros((n, n_i, d))
    for i in range(n):
        print(i, end='\r')
        # TODO: Principled way to make sparse mixtures
        finished = False
        while not finished:
            weights = np.zeros((k, ))
            idxs = np.random.choice(k, n_mix)
            for idx in idxs:
                weights[idx] = np.random.uniform(0, 1)
            #eights = np.random.uniform(0, 1, k)*np.random.binomial(1, float(n_mix)/k, size=(k))
            if np.sum(weights) < 1e-3:
                continue
            weights /= np.sum(weights)
            W_i = np.tensordot(weights, W_k, axes=1)
            c_i = np.tensordot(weights, c_k, axes=1)
            try:
                X_i = simulate_linear_sem(W_i, n_i, sem_type, noise_scale=noise_scale)
            except ValueError: # mixture may not be a DAG
                continue
            X_n[i] = X_i
            subtypes[i] = weights
            W_n[i] = W_i
            c_n[i] = c_i
            finished = True
    return subtypes, W_n, c_n, X_n


def gen_samples_no_archs(n, d, s0, n_i, n_c, c_signal_noise, 
                         graph_type='ER', sem_type='gauss', noise_scale=0.1):
    W_n = np.zeros((n, d, d))
    c_n = np.zeros((n, n_c))
    X_n = np.zeros((n, n_i, d))
    for i in range(n):
        print(i, end='\r')
        W_n[i] = simulate_dag(d, s0, graph_type)
        W_n[i] = simulate_parameter(W_n[i])
        X_n[i] = simulate_linear_sem(W_n[i], n_i, sem_type, noise_scale=noise_scale)
    pca = PCA(n_components=n_c)
    c_n = pca.fit_transform(np.array([w.flatten() for w in W_n]))
    if c_signal_noise > 0 and c_signal_noise < 1:
        for j in range(n_c):
            c_n[:, j] += np.random.normal(0., np.var(c_n[:, j])*(1./c_signal_noise - 1), size=(n, ))

    return W_n, c_n, X_n


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        #return P.T @ M @ P
        return np.matmul(np.matmul(P.T, M), P)

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-10.0, -1), (1, 10.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
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
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            #x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            #x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            #x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            #x = X @ w + z
            x = np.matmul(X, w) + z
        elif sem_type == 'logistic':
            #x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            x = np.random.binomial(1, sigmoid(np.matmul(X, w))) * 1.0
        elif sem_type == 'poisson':
            #x = np.random.poisson(np.exp(X @ w)) * 1.0
            x = np.random.poisson(np.exp(np.matmul(X, w))) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not graph_utils.is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            #X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            X = np.sqrt(d) * np.matmul(np.diag(scale_vec), np.linalg.inv(np.eye(d) - W))
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            #x = sigmoid(X @ W1) @ W2 + z
            x = np.matmul(sigmoid(np.matmul(X, W1)), W2) + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            #x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            x = np.tanh(np.matmul(X, w1)) + np.cos(np.matmul(X, w2)) + np.sin(np.matmul(X, w3)) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if not (noise_scale is None) else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X
