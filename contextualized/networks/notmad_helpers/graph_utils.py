import igraph as ig
import numpy as np
import random
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf

import sys
from contextualized.networks.notmad_helpers.tf_utils import DAG_loss

def break_symmetry(w):
    for i in range(w.shape[0]):
        w[i][i] = 0.
        for j in range(i):
            if np.abs(w[i][j]) > np.abs(w[j][i]):
                w[j][i] = 0.
            else:
                w[i][j] = 0.
    return w

def project_to_dag_tf(w, thresh=0.1):
    w = break_symmetry(w)
    w_tf = tf.Variable(w, trainable=True)
    i = 0
    while not is_dag(w_tf.numpy()):
        i += 1
        thresh *= 1.1
        w_tf.assign(trim_params(w_tf.numpy(), thresh))
        if i % 10 == 0:
            w_tf.assign(break_symmetry(w_tf.numpy()))
        with tf.GradientTape(persistent=False) as tape:
            losses = [DAG_loss(w_tf, alpha=1, rho=0.)]
            grads = tape.gradient(losses, [w_tf])
            opt = tf.keras.optimizers.Adam(learning_rate=1e-1)
            opt.apply_gradients(zip(grads, [w_tf]))
    return w_tf.numpy()


#w is the weighted adjacency matrix
def project_to_dag(w):
    if is_dag(w):
        return w, 0.0
    
    w_dag = w.copy()
    w_dag = break_symmetry(w_dag)
    
    vals = sorted(list(set(np.abs(w_dag).flatten())))
    low = 0
    high = len(vals) -1

    def binary_search(arr, low, high, w):    #low and high are indices
        # Check base case
        if high == low:
            return high
        if high > low:
            mid = (high + low) // 2
            if (mid == 0):
                return mid
            result = trim_params(w, arr[mid])
            if is_dag(result):
                result2 = trim_params(w, arr[mid-1])
                if is_dag(result2): #middle value is too high.  go lower.
                    return binary_search(arr, low, mid-1, w)
                else:
                    return mid #found it
            else: # middle value is too low.  go higher.
                return binary_search(arr, mid+1, high, w)
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
        if np.abs(w[i][j]) > thresh: # already retained
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
