# system/LHS.py
# Latin Hypercube Sampling for initial parameter samples.

import config
import sim_handler
import pandas as pd
import numpy as np
from scipy.stats import qmc
from smt.sampling_methods import LHS as SMT_LHS


def generate_lhs_samples(start_sim_id=None):
    """Generate LHS samples and write them to config.PARAMS_CSV_FILE."""
    n_samples = config.LHS_N_SAMPLES
    bounds = config.SWEEP_PARAMETERS
    output_file = config.PARAMS_CSV_FILE
    method = config.LHS_SAMPLING_METHOD

    seed = getattr(config, 'LHS_RANDOM_SEED', None)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))

    if start_sim_id is None:
        start_sim_id = 1

    param_names = list(bounds.keys())
    n_dim = len(param_names)
    limits = np.array([[bounds[k]['min'], bounds[k]['max']] for k in param_names])

    if method == 'random':
        sampler = qmc.LatinHypercube(d=n_dim, optimization=None, seed=seed)
        samples = qmc.scale(sampler.random(n=n_samples), limits[:, 0], limits[:, 1])
    elif method in ('maximin', 'optimum'):
        # SMT criteria names; SMT does not accept random_state in newer versions.
        criterion = 'm' if method == 'maximin' else 'ese'
        np.random.seed(seed)
        samples = SMT_LHS(xlimits=limits, criterion=criterion)(n_samples)
    else:
        raise ValueError(f"Method must be 'random', 'maximin', or 'optimum'. Got: '{method}'")

    df = pd.DataFrame(samples, columns=param_names)
    for col in param_names:
        df[col] = df[col].apply(lambda x: float(f'{x:.4g}'))
    for param in param_names:
        df[param] = df[param].apply(lambda x: sim_handler.snap_to_discrete(param, x))

    df.insert(0, 'sim_id', range(start_sim_id, start_sim_id + len(df)))

    units = {p: bounds[p].get('unit', '-') for p in param_names}
    units_row = pd.DataFrame([{**{'sim_id': '-'}, **units}])
    pd.concat([units_row, df], ignore_index=True).to_csv(output_file, index=False)
    print(f"  LHS: {n_samples} samples ({method}, seed={seed}) → {output_file}")
    return df
