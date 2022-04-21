"""
Adapted from https://github.com/xunzheng/notears/blob/master/notears/utils.py
"""

#from scipy.special import expit as sigmoid
import igraph as ig
import numpy as np
import random
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf

# from NOTEARS import NOTEARS

"""
from LowRankContextualNOTEARS import LRContextualNOTEARS
def fit_crossval(C, X, crossval_epochs, full_epochs, batch_size,
                 ks=[2, 4], ranks=[8], lrs=[1e-2, 1e-1], widths=[16], n_encoder_layers=[1],
                 l1s=[0, 1e-3], alphas=[1e0], rhos=[1e-1],val_size=0.25, es_patience=10, verbose=1, init_mat=None):
    C_train, C_val, X_train, X_val = train_test_split(C, X, test_size=val_size)

    encoder_input_shape = (C.shape[1], 1)
    archetype_loss_params = {'l1': 1e-3}
    histories = {}

    best_val_loss = np.inf
    best_args = None

    for k in ks:
        encoder_output_shape = (k, )
        for rank in ranks:
            dict_shape =(k, X.shape[-1], rank)

            # Calc init archetypes by random subset of data.
            loss_params = {'l1': 1e-2, 'init_alpha': 0., 'init_rho':1.}
            W_shape = (X_train.shape[1], X_train.shape[1])
            notears = NOTEARS(loss_params, encoder_input_shape, W_shape)
            init_archs = {"A": [], "B": []}
            arch_pca = PCA(n_components=rank)
            for i in range(k):
                C_sel, _, X_sel, _ = train_test_split(C_train, X_train, test_size=0.9)
                notears.fit(C_sel, X_sel, epochs=10, batch_size=1, es_patience=10)
                sel_network = notears.model.predict(np.expand_dims(C_train[0], 0)).squeeze()
                arch_pca.fit(np.array(sel_network))
                init_archs["A"].append(arch_pca.transform(sel_network))
                init_archs["B"].append(arch_pca.components_)

            for lr in lrs:
                for width in widths:
                    for n_layers in n_encoder_layers:
                        for l1 in l1s:
                            for alpha in alphas:
                                for rho in rhos:
                                    sample_specific_loss_params = {'l1':l1, 'init_alpha': alpha, 'init_rho': rho}

                                    model = LRContextualNOTEARS(encoder_input_shape, encoder_output_shape, dict_shape,
                                        sample_specific_loss_params, archetype_loss_params,
                                        n_encoder_layers=n_layers, encoder_width=width,
                                        context_activity_regularizer=tf.keras.regularizers.l1(1e-3),
                                        learning_rate=lr, tf_dtype=tf.dtypes.float32,
                                                                init_mat=init_mat, init_archs=init_archs)
                                    histories[(k, lr, width, n_layers, l1, alpha, rho)] = model.fit(C_train, X_train,
                                        batch_size=batch_size, epochs=crossval_epochs, verbose=verbose)

                                    # TODO: Should measure the full loss, not just MSE?
                                    val_loss = np.mean(mses_xw(X_val, model.predict_w(C_val)))
                                    if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        best_args = {'k': k, 'rank': rank, 'lr': lr, 'width': width,
                                                     'n_layers': n_layers, 'l1': l1, 'init_alpha': alpha, 'init_rho': rho}
    model = LRContextualNOTEARS(encoder_input_shape, encoder_output_shape=(best_args['k'],),
                                dict_shape=(best_args['k'], X.shape[-1], best_args['rank']),
                                sample_specific_loss_params=best_args,
                                archetype_loss_params=archetype_loss_params,
                                n_encoder_layers=best_args['n_layers'], encoder_width=best_args['width'],
                                context_activity_regularizer=tf.keras.regularizers.l1(0),
                                learning_rate=best_args['lr'], tf_dtype=tf.dtypes.float32,
                                init_mat=init_mat, init_archs=init_archs)
    model.fit(C, X, epochs=full_epochs, batch_size=batch_size, es_patience=es_patience, val_split=0.25, verbose=verbose)
    return model, best_args, histories
"""

def mses_xw(X, W):
    return np.array([mse_xw(X[i], W[i]) for i in range(len(X))])

def mse_xw(x_true, w_pred):
    x_prime = np.matmul(x_true, w_pred)
    return (0.5 / float(np.shape(x_true)[0])) * np.square(np.linalg.norm(x_true - x_prime))


def get_f1s(W_test, W_test_hat, threshs):
    return [
        np.mean([f1_mat(W_test[i], W_test_hat[i], thresh, thresh) for i in range(len(W_test))])
            for thresh in threshs
    ]

def f1_mat(y_true, y_pred, true_thresh, pred_thresh):
    return f1_score(np.abs(y_true.flatten()) > true_thresh, np.abs(y_pred.flatten()) > pred_thresh, average='macro')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



def fpr(w_true, w_hat, thresh_est=0.2, thresh_true=0.1):
    pos_preds = w_hat > thresh_est
    pos_true  = w_true > thresh_true
    return 1.0 - np.mean((pos_preds * pos_true)[pos_preds])


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

