import numpy as np
from simulation import simulate_linear_sem

def create_test_cxw_network(n=1000): #n data points
    C = np.linspace(1, 2, n).reshape((n, 1))
    blank = np.zeros_like(C)
    W_00 = blank
    W_01 = C-2
    W_02 = blank
    W_03 = blank
    W_10 = blank
    W_11 = blank
    W_12 = blank
    W_13 = blank
    W_20 = blank
    W_21 = C**2
    W_22 = blank
    W_23 = blank
    W_30 = blank
    W_31 = C**3
    W_32 = C
    W_33 = blank

    W = np.array([
        [W_00, W_01, W_02, W_03],
        [W_10, W_11, W_12, W_13],
        [W_20, W_21, W_22, W_23],
        [W_30, W_31, W_32, W_33],
    ]).squeeze()
    
    def _dag_pred(self,x,w):
        return np.matmul(x, w).squeeze()

    W = np.transpose(W, (2, 0, 1))
    X = np.zeros((n, 4))
    X_pre = np.random.uniform(-1, 1, (n, 4))
    for i, w in enumerate(W):
        eps = np.random.normal(0, .01, 4)
        eps = 0
        X_new = simulate_linear_sem(w, 1, 'uniform',noise_scale=0.1)[0]
        #X_new = _dag_pred(X_p[np.newaxis, :], w)
        X[i] = X_new + eps

    return C, X, W
