# system/BO.py
# Bayesian Optimization module for parameter optimization
# Uses Gaussian Process model and UCB acquisition function
#
# Parameters are normalized to [0,1] and costs are log-transformed
# before registering with the GP. This is necessary because:
#   - bayes_opt uses an isotropic Matern kernel (single length_scale)
#   - Parameters span 26 orders of magnitude (w_r ~4e-7 vs doping ~5e17)
#   - Costs span 8000x (valid: 2-18 vs failed: 1468-18003)

import os
import config
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound


# ============================================================================
# Parameter normalization (raw ↔ [0,1])
# ============================================================================

def _normalize_params(params):
    """Normalize raw params to [0,1] using SWEEP_PARAMETERS bounds."""
    return {name: (params[name] - cfg['min']) / (cfg['max'] - cfg['min'])
            for name, cfg in config.SWEEP_PARAMETERS.items()}


def _denormalize_params(norm_params):
    """Convert [0,1] params back to raw scale."""
    return {name: norm_params[name] * (cfg['max'] - cfg['min']) + cfg['min']
            for name, cfg in config.SWEEP_PARAMETERS.items()}


# ============================================================================
# Optimizer lifecycle
# ============================================================================

def train_optimizer(result_csv_path=None):
    """
    Creates and trains the Bayesian Optimizer with prior data from result.csv.

    Parameters are normalized to [0,1] and costs are log-transformed
    before registration. The optimizer should be created ONCE and reused
    across iterations (not recreated each iteration).

    Args:
        result_csv_path: Path to result.csv. Defaults to config.RESULTS_CSV_FILE.

    Returns:
        BayesianOptimization: Trained optimizer object
    """
    # load the results csv file
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(f"Results file not found: {result_csv_path}")

    df = pd.read_csv(result_csv_path)
    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")

    print(f"  -> Loaded {len(df)} prior data points from {result_csv_path}")

    # All parameters normalized to [0,1]
    param_names = list(config.SWEEP_PARAMETERS.keys())
    pbounds = {name: (0.0, 1.0) for name in param_names}

    # create the optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=42,
        acquisition_function=UpperConfidenceBound(kappa=config.BO_KAPPA),
        verbose=2,
        allow_duplicate_points=True,
    )

    # Register each prior data point
    registered = 0
    skipped = 0

    for index, row in df.iterrows():
        try:
            if not all(name in row for name in param_names):
                skipped += 1
                continue

            raw_params = {name: row[name] for name in param_names}

            if 'cost' not in row or np.isnan(row['cost']) or row['cost'] <= 0:
                skipped += 1
                continue

            # Normalize params to [0,1], log-transform cost
            norm_params = _normalize_params(raw_params)
            optimizer.register(params=norm_params, target=-np.log(row['cost']))
            registered += 1

        except Exception as e:
            print(f"  [WARNING] Error processing row {index}: {e}")
            skipped += 1

    print(f"  -> Registered {registered} data points with optimizer")
    if skipped > 0:
        print(f"  -> Skipped {skipped} rows")

    return optimizer


def get_next_sample(optimizer):
    """
    Suggests next parameter set to sample using the trained optimizer.

    Returns parameters in raw scale (denormalized from [0,1]).

    Args:
        optimizer: Trained BayesianOptimization object

    Returns:
        dict: Next parameter values in raw scale, or None on failure
    """
    if optimizer is None:
        raise ValueError("Optimizer is None. Train the optimizer first.")

    try:
        norm_point = optimizer.suggest()
        if norm_point is None:
            print("  [WARNING] Optimizer suggest() returned None")
            return None

        # Denormalize from [0,1] back to raw scale
        raw_point = _denormalize_params(norm_point)

        # Clip to raw bounds (safety)
        for name, cfg in config.SWEEP_PARAMETERS.items():
            raw_point[name] = np.clip(raw_point[name], cfg['min'], cfg['max'])

        print(f"  -> Next suggested point: {raw_point}")
        return raw_point

    except Exception as e:
        print(f"  [ERROR] Failed to get next sample: {e}")
        return None


def register_result(optimizer, params, cost_value):
    """
    Registers a new simulation result with the optimizer.

    Args:
        optimizer: BayesianOptimization object
        params: dict of parameter values in raw scale (w_r, h_si, etc.)
        cost_value: Positive cost value (will be log-transformed and negated)
    """
    norm_params = _normalize_params(params)
    optimizer.register(params=norm_params, target=-np.log(cost_value))


def get_best_result(result_csv_path=None):
    """
    Returns the best result (lowest cost) from results CSV.

    Args:
        result_csv_path: Path to result.csv. Defaults to config.RESULTS_CSV_FILE.

    Returns:
        dict: Best result row, or None if no valid results
    """
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    if not os.path.exists(result_csv_path):
        return None

    df = pd.read_csv(result_csv_path)
    if len(df) == 0 or 'cost' not in df.columns:
        return None

    # Drop rows with NaN cost, find minimum (lowest positive cost = best)
    valid = df.dropna(subset=['cost'])
    if len(valid) == 0:
        return None

    best_idx = valid['cost'].idxmin()
    return valid.loc[best_idx].to_dict()
