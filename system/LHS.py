# system/LHS.py
# Latin Hypercube Sampling (LHS) module for generating initial parameter samples

import config
import results_archive
import pandas as pd
import numpy as np
from scipy.stats import qmc
from smt.sampling_methods import LHS as SMT_LHS


def generate_lhs_samples(start_sim_id=None):
    """
    Generates Latin Hypercube Sampling (LHS) samples based on parameter bounds from config.

    Args:
        start_sim_id (int, optional): Starting sim_id. If None, auto-detects from archive.

    Returns:
        pd.DataFrame: DataFrame containing the generated samples with columns
                      for each parameter and a 'sim_id' column.

    Output:
        Saves the samples to a CSV file (params.csv) with parameter values.
    """
    # Get values from config
    n_samples = config.LHS_N_SAMPLES
    bounds = config.SWEEP_PARAMETERS
    output_file = config.PARAMS_CSV_FILE
    method = config.LHS_SAMPLING_METHOD

    # Get random seed from config (None = use system time)
    seed = getattr(config, 'LHS_RANDOM_SEED', None)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"  Using random seed: {seed}")

    # Determine starting sim_id
    if start_sim_id is None:
        start_sim_id = results_archive.get_next_sim_id()
        print(f"  Starting sim_id: {start_sim_id} (auto-detected from archive)")

    # Extract parameter names and convert bounds dictionary to numpy array format
    param_names = list(bounds.keys())
    n_dim = len(param_names)

    # Convert bounds dictionary to numpy array: [[min1, max1], [min2, max2], ...]
    limits = np.array([[bounds[key]['min'], bounds[key]['max']] for key in param_names])

    print(f"--- Running LHS: Method='{method}', Dims={n_dim}, Samples={n_samples}, Seed={seed} ---")

    # Generate samples based on method
    if method == 'random':
        # Use Scipy for the basic version
        sampler = qmc.LatinHypercube(d=n_dim, optimization=None, seed=seed)
        sample_01 = sampler.random(n=n_samples)
        # Manual scaling (required for scipy)
        samples = qmc.scale(sample_01, limits[:, 0], limits[:, 1])

    elif method in ['maximin', 'optimum']:
        # Convert to SMT criteria names
        criterion = 'm' if method == 'maximin' else 'ese'
        # SMT performs scaling automatically based on xlimits
        sampling = SMT_LHS(xlimits=limits, criterion=criterion, random_state=seed)
        samples = sampling(n_samples)

    else:
        raise ValueError(f"Method must be 'random', 'maximin', or 'optimum'. Got: '{method}'")
    
    # Create DataFrame with parameter columns
    df = pd.DataFrame(samples, columns=param_names)

    # Round to 4 significant figures
    for col in param_names:
        df[col] = df[col].apply(lambda x: float(f'{x:.4g}'))

    # Add simulation ID as first column (continue from start_sim_id)
    df.insert(0, 'sim_id', range(start_sim_id, start_sim_id + len(df)))
    
    # Create units row
    units = {param: bounds[param].get('unit', '-') for param in param_names}
    units_row = pd.DataFrame([{**{'sim_id': '-'}, **units}])
    
    # Combine units row with data
    df_with_units = pd.concat([units_row, df], ignore_index=True)
    
    # Save to CSV file
    df_with_units.to_csv(output_file, index=False)
    print(f"LHS samples saved to {output_file}")
    print(f"Generated {n_samples} samples with {n_dim} parameters")
    
    return df
