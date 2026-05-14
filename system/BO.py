# system/BO.py
# Bayesian Optimization wrapper. Params normalized to [0,1] and costs log-transformed
# before registration with the GP, because:
#   - bayes_opt uses an isotropic Matern kernel (single length_scale)
#   - Raw params span ~26 orders of magnitude (w_r ~4e-7 vs doping ~5e17)
#   - Costs span ~8000x between valid and failed sims

import os
import config
import sim_handler
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound


def _normalize_params(params):
    return {name: (params[name] - cfg['min']) / (cfg['max'] - cfg['min'])
            for name, cfg in config.SWEEP_PARAMETERS.items()}


def _denormalize_params(norm_params):
    return {name: norm_params[name] * (cfg['max'] - cfg['min']) + cfg['min']
            for name, cfg in config.SWEEP_PARAMETERS.items()}


def train_optimizer(result_csv_path=None):
    """Build the optimizer and register all prior rows. Call once per run."""
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE
    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(f"Results file not found: {result_csv_path}")

    df = pd.read_csv(result_csv_path)
    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")

    param_names = list(config.SWEEP_PARAMETERS.keys())
    pbounds = {name: (0.0, 1.0) for name in param_names}

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=42,
        acquisition_function=UpperConfidenceBound(
            kappa=config.BO_KAPPA,
            exploration_decay=config.BO_KAPPA_DECAY,
            exploration_decay_delay=0,
        ),
        verbose=0,
        allow_duplicate_points=True,
    )

    registered = 0
    for _, row in df.iterrows():
        if not all(name in row for name in param_names):
            continue
        if 'cost' not in row or np.isnan(row['cost']) or row['cost'] <= 0:
            continue
        raw_params = {name: row[name] for name in param_names}
        optimizer.register(params=_normalize_params(raw_params), target=-np.log(row['cost']))
        registered += 1

    print(f"  BO: registered {registered}/{len(df)} prior points")
    return optimizer


def get_next_sample(optimizer):
    if optimizer is None:
        raise ValueError("Optimizer is None. Train the optimizer first.")

    norm_point = optimizer.suggest()
    if norm_point is None:
        return None

    raw_point = _denormalize_params(norm_point)
    for name, cfg in config.SWEEP_PARAMETERS.items():
        raw_point[name] = np.clip(raw_point[name], cfg['min'], cfg['max'])
    return raw_point


def register_result(optimizer, params, cost_value):
    """Register a result with the optimizer. params are raw, cost is positive."""
    snapped = sim_handler.snap_params_dict(params)
    for k in config.SWEEP_PARAMETERS.keys():
        if k in params and k in snapped and abs(params[k] - snapped[k]) > 1e-12:
            print(f"[WARNING] Non-snapped {k}={params[k]:.4e} (expected {snapped[k]:.4e})")
    optimizer.register(params=_normalize_params(params), target=-np.log(cost_value))


def get_current_kappa(optimizer):
    try:
        return optimizer.acquisition_function.kappa
    except AttributeError:
        return None


def get_best_result(result_csv_path=None):
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE
    if not os.path.exists(result_csv_path):
        return None

    df = pd.read_csv(result_csv_path)
    if len(df) == 0 or 'cost' not in df.columns:
        return None
    valid = df.dropna(subset=['cost'])
    if len(valid) == 0:
        return None
    return valid.loc[valid['cost'].idxmin()].to_dict()
