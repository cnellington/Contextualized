import itertools
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from correlator.helpers.simulation import GaussianSimulator
from correlator.correlator import ContextualCorrelator


# Data Params
def linear_interpolate(mat1, mat2, steps):
    assert mat1.shape == mat2.shape
    mat_step = (mat2 - mat1) / steps
    ret = np.zeros([steps] + list(mat1.shape))
    ret[0] = mat1
    for i in range(1, steps - 1):
        ret[i] = mat1 + i * mat_step
    ret[-1] = mat2
    return np.copy(ret)

def make_sigma(p):
    A = np.random.random((p, p)) * 2 - 1
    sigma = A.T @ A
    return sigma

k_list = [10, 50, 100, 500]
k_n = 1
p_list = [5, 10, 15, 20, 50, 100] 

# Model Params
target_params = ['num_archetypes', 'encoder_width', 'encoder_layers', 'final_dense_size', 'l1']
num_archetypes_list = [100, ]
encoder_width_list = [50, ]
encoder_layers_list = [3, ]
final_dense_size_list = [100, ]
l1_list = [0.001, ]
bootstraps = None

paramlists = [
    num_archetypes_list,
    encoder_width_list,
    encoder_layers_list,
    final_dense_size_list,
    l1_list,
]
paramsets = list(itertools.product(*paramlists))

# Simulation Params
basedir = "/home/caleb.ellington/ContextualizedCorrelator/experiments/simulation/results/"
filename = sys.argv[0].split('/')[-1].split('.')[0]
jobid = os.environ["SLURM_JOB_ID"]
script_path = basedir + f'{filename}_{jobid}.out'
if not os.path.exists(script_path):
    header = ['p', 'k'] + target_params + ['mse', 'mse_var', 'norm', 'norm_var', 'edge_var']
    open(script_path, 'w').write(', '.join(header) + '\n')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runs = 10

# Run Simulation
print('Started Successfully, see you later <:)')
for _ in range(runs):
    for p in p_list:
        sigma1 = make_sigma(p)
        sigma2 = make_sigma(p)
        mu1 = (np.random.random(p) * 2 - 1) * 10
        mu2 = (np.random.random(p) * 2 - 1) * 10
        c = (p + 1) * p
        for k in k_list:
            sigmas = linear_interpolate(sigma1, sigma2, k)
            mus = linear_interpolate(mu1, mu2, k)
            sim = GaussianSimulator(p, k, c, ctype='self', sigmas=sigmas, mus=mus)
            C_train, X_train = sim.gen_samples(k_n)
            C_test, X_test = sim.gen_samples(k_n)
            C_val, X_val = sim.gen_samples(k_n)
            for paramset in paramsets:
                model_params = {key: val for key, val in zip(target_params, paramset)}
                data_params = {'context_dim': c, 'x_dim': p, 'y_dim': p}
                model_params.update(data_params)
                model = ContextualCorrelator(**model_params)
                model = model.to(DEVICE)
                model.fit(C_train, X_train, X_train, 100, 1, validation_set=(C_val, X_val, X_val), es_epoch=10, es_patience=100, silent=True) 
                mses = model.get_mse(C_test, X_test, X_test, all_bootstraps=True)
                mse = mses.mean()
                mse_var = mses.var(axis=-1).mean()
                corrs = model.predict_correlation(C_test, all_bootstraps=True).numpy()
                true_corrs = sim.rhos[:,:,:,np.newaxis]
                if bootstraps is not None:
                    true_corrs = np.repeat(true_corrs, bootstraps, axis=-1)
                norm_sqdiff = (corrs - true_corrs)**2
                edge_var = norm_sqdiff.var(axis=-1).mean()
                norms = norm_sqdiff.mean(axis=(1, 2))
                norm = norms.mean()
                norm_var = norms.var(axis=-1).mean()
                vals = [p, k] + list(paramset) + [mse, mse_var, norm, norm_var, edge_var]
                row = ', '.join([str(val) for val in vals])
                open(script_path, 'a').write(row + '\n')
print('Finished successfully')
