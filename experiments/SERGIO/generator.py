"""
Caleb Ellington
8/3/21

Randomly perturbs a biologically derived network while enforcing acyclicity and preserving topological ordering.
Simulates these networks as single-cell transcriptomic profiles using SERGIO.
"""


import argparse
import os
import networkx as nx
import numpy as np

from sergio import sergio

def _make_consistent(G):
    return nx.convert_matrix.from_numpy_matrix(nx.convert_matrix.to_numpy_matrix(G), create_using=nx.DiGraph)

def _read_dot(dotpath, w_range: tuple = (0.5, 2.0)):
    """
    Build directed graph greedily from .dot file
    ignore edges that violate DAGness
    """
    T = nx.DiGraph()
    lines = open(dotpath, 'r').readlines()
    gene_ids = {}
    id_ticker = 0
    for line in lines[2:-1]:
        if '->' in line:
            vals = line.strip('\t\n;').replace('-> ', '').split(' ')
            parent = gene_ids[vals[0].strip('\"')]
            child = gene_ids[vals[1].strip('\"')]
            weight = np.random.uniform(w_range[0], w_range[1])
            sign = np.random.randint(0, 2) * 2 - 1
            weight *= sign
            T.add_edge(parent, child, weight=weight)
            if not nx.is_directed_acyclic_graph(T):
                T.remove_edge(parent, child)
        else:
            gene = line.strip('\t\n;\"')
            gene_ids[gene] = id_ticker
            T.add_node(id_ticker)
            id_ticker += 1
    return _make_consistent(T)


def _perturb(T, n_perturbs, w_range: tuple = (0.5, 2.0)):
    """
    Sample K dags with the same topological ordering as G
    with <n_perturbs> edges added and <n_perturbs> edges
    removed per sample
    """
    toposort = np.array(list(nx.lexicographical_topological_sort(T)))
    indegree = np.array(list(dict(T.in_degree(toposort)).values()))
    MR_idx = indegree == 0
    g = nx.DiGraph(T)
    for _ in range(n_perturbs):  # add n_perturbs edges
        # Get parent-child pair that doesn't violate topological order or MRs
        parent_i = np.random.choice(len(toposort) - 1)
        potential_children = (toposort[parent_i + 1:])[~MR_idx[parent_i + 1:]]
        parent = toposort[parent_i]
        child = np.random.choice(potential_children)
        weight = np.random.uniform(w_range[0], w_range[1])
        sign = np.random.randint(0, 2) * 2 - 1
        weight *= sign
        g.add_edge(parent, child, weight=weight)
        assert list(nx.lexicographical_topological_sort(T)) == list(nx.lexicographical_topological_sort(g))
    for _ in range(n_perturbs):  # remove n_pertrubs edges, preserve expected sparsity
        remove_i = np.random.choice(len(g.edges))
        g.remove_edge(*list(g.edges)[remove_i])
    return g


def _simulate_sergio(G, n_samples, hill=2, mr_range: tuple = (0.5, 2.0)):
    def write_rows(path, rows):
        file = open(path, 'w')
        for row in rows:
            line = ''
            for val in row:
                line += ', ' + str(val)
            line = line[2:] + '\n'
            file.write(line)

    swap_dir = 'sergio_temp/'
    if not os.path.isdir(swap_dir):
        os.makedirs(swap_dir)

    # To rows
    nodes = np.array(list(G.nodes))
    indegree = np.array(list(dict(G.in_degree(nodes)).values()))
    MRs = nodes[indegree == 0]
    targets = nodes[indegree != 0]
    mr_rows = [[mr] + list(np.random.uniform(mr_range[0], mr_range[1], n_samples)) for mr in MRs]
    # mr_rows = [[mr] + list(np.random.uniform(mr_range[0], mr_range[1], 1)) for mr in MRs]
    grn_rows = []
    for target in targets:
        parents = list(G.predecessors(target))
        weights = [G[parent][target]['weight'] for parent in parents]
        n_hill = [hill] * len(weights)
        row = [target, len(parents)] + parents + weights + n_hill
        grn_rows.append(row)

    mr_path = swap_dir + 'MR.txt'
    grn_path = swap_dir + 'GRN.txt'
    write_rows(mr_path, mr_rows)
    write_rows(grn_path, grn_rows)

    sim = sergio(number_genes=len(nodes), number_bins=n_samples, number_sc=1, noise_params=0.0, decays=1,
                 sampling_state=15, noise_type='sp')
    sim.build_graph(input_file_taregts=grn_path, input_file_regs=mr_path)
    sim.simulate()
    expr = sim.getExpressions()
    expr = np.concatenate(expr, axis=1)
    return expr


if __name__ == '__main__':
    cmd_opt = argparse.ArgumentParser(description='')
    cmd_opt.add_argument('-dot_path', type=str, default='./regulatory_networks/Ecoli_100_net1.dot', help='path to GRN .dot file')
    cmd_opt.add_argument('-save_dir', type=str, default='../data/', help='directory to save simulation')
    cmd_opt.add_argument('-K', type=int, default=10, help='number of tasks')
    cmd_opt.add_argument('-e', type=int, default=10, help='number of edge perturbations to make per task')
    cmd_opt.add_argument('-n', type=int, default=300, help='number of samples per task')
    cmd_opt.add_argument('-nh', type=float, default=2, help='hill coefficient for SERGIO')
    args = cmd_opt.parse_args()

    w_range = (0.5, 10.0)
    T = _read_dot(args.dot_path, w_range=w_range)
    p = len(T.nodes)
    G = [_perturb(T, args.e, w_range=w_range) for _ in range(args.K)]
    exprs = [_simulate_sergio(g, args.n, hill=args.nh, mr_range=w_range) for g in G]
    expr_mat = np.concatenate([expr for expr in exprs], axis=-1).T
    task_labels = [i // args.n for i in range(len(G) * args.n)]
    T_adj = nx.convert_matrix.to_numpy_matrix(T).T
    G_adj = [nx.convert_matrix.to_numpy_matrix(g).T for g in G]

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    savepath = os.path.join(args.save_dir, f"sergio_K-{args.K}_p-{p}_e-{args.e}_n-{args.n * args.K}_nh-{args.nh}.npz")
    np.savez(savepath, expression=expr_mat, task_labels=task_labels, task_adjacencies=G_adj, true_adjacency=T_adj)
