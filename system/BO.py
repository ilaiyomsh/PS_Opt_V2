# system/BO.py
# Bayesian Optimization module for parameter optimization
# Uses Gaussian Process model and UCB acquisition function

import os
import config
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound


def train_optimizer(result_csv_path=None):
    """
    Trains the Bayesian Optimizer with prior data from result.csv.

    Args:
        result_csv_path: Path to result.csv. Defaults to config.RESULTS_CSV_FILE.

    Returns:
        BayesianOptimization: Trained optimizer object
    """
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(f"Results file not found: {result_csv_path}")

    df = pd.read_csv(result_csv_path)
    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")

    print(f"  -> Loaded {len(df)} prior data points from {result_csv_path}")

    # Build pbounds: {'w_r': (350e-9, 450e-9), ...}
    param_names = list(config.SWEEP_PARAMETERS.keys())
    pbounds = {name: (cfg['min'], cfg['max'])
               for name, cfg in config.SWEEP_PARAMETERS.items()}

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=42,
        acquisition_function=UpperConfidenceBound(kappa=config.BO_KAPPA),
        verbose=2,
    )

    # Register each prior data point
    registered = 0
    skipped = 0

    for index, row in df.iterrows():
        try:
            # Check all params exist
            if not all(name in row for name in param_names):
                skipped += 1
                continue

            params = {name: row[name] for name in param_names}

            # Cost column is always present (written by run_simulation)
            if 'cost' not in row or np.isnan(row['cost']):
                skipped += 1
                continue

            # CSV stores positive cost; BO maximizes, so negate
            optimizer.register(params=params, target=-row['cost'])
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

    Args:
        optimizer: Trained BayesianOptimization object

    Returns:
        dict: Next parameter values, or None on failure
    """
    if optimizer is None:
        raise ValueError("Optimizer is None. Train the optimizer first.")

    try:
        next_point = optimizer.suggest()
        if next_point is None:
            print("  [WARNING] Optimizer suggest() returned None")
            return None

        # Clip to bounds
        for name, cfg in config.SWEEP_PARAMETERS.items():
            if name not in next_point:
                print(f"  [WARNING] Parameter '{name}' missing from suggested point")
                return None
            next_point[name] = np.clip(next_point[name], cfg['min'], cfg['max'])

        print(f"  -> Next suggested point: {next_point}")
        return next_point

    except Exception as e:
        print(f"  [ERROR] Failed to get next sample: {e}")
        return None


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
