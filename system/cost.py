# system/cost.py
# Cost function for PIN diode phase shifter optimization (Eq. 27)
#   Success: cost = w_loss * (alpha/target_loss)^2 + w_vpil * (vpil/target_vpil)^2
#   Penalty: cost = C_BASE + beta * (pi - max_dphi)^2

import os
import config
import pandas as pd
import numpy as np

# Module-level state for dynamic C_BASE (Eq. 27)
_C_BASE = config.C_BASE_DEFAULT
_PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)


def get_c_base():
    return _C_BASE


def get_penalty_beta():
    return _PENALTY_BETA


def update_c_base(value):
    """Set C_BASE and recompute PENALTY_BETA."""
    global _C_BASE, _PENALTY_BETA
    _C_BASE = value
    _PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)


def reset_c_base():
    """Reset C_BASE to the default value from config."""
    update_c_base(config.C_BASE_DEFAULT)


def update_c_base_if_needed(new_valid_cost):
    """Update C_BASE only if new_valid_cost exceeds current value.

    Returns True if C_BASE changed (caller should re-score failed rows).
    """
    if new_valid_cost > _C_BASE:
        update_c_base(new_valid_cost)
        return True
    return False


def calculate_c_base_from_results(result_csv_path_or_df):
    """
    Sets C_BASE = max(cost) among valid simulations (those that reached pi).

    Args:
        result_csv_path_or_df: Path to result.csv or a DataFrame

    Returns:
        float: Updated C_BASE value
    """
    if isinstance(result_csv_path_or_df, pd.DataFrame):
        df = result_csv_path_or_df
    else:
        if not os.path.exists(result_csv_path_or_df):
            return config.C_BASE_DEFAULT
        df = pd.read_csv(result_csv_path_or_df)

    if len(df) == 0:
        return config.C_BASE_DEFAULT

    # Filter to valid simulations (have v_pi_l and alpha)
    valid = df.dropna(subset=['v_pi_l_Vmm', 'loss_at_v_pi_dB_per_cm'])
    if len(valid) == 0:
        print(f"  -> No valid simulations found, using default C_BASE = {config.C_BASE_DEFAULT}")
        return config.C_BASE_DEFAULT

    # Calculate cost for each valid row
    costs = []
    for _, row in valid.iterrows():
        norm_loss = (row['loss_at_v_pi_dB_per_cm'] / config.TARGETS['loss']) ** 2
        norm_vpil = (row['v_pi_l_Vmm'] / config.TARGETS['vpil']) ** 2
        cost = config.FOM_WEIGHTS['loss'] * norm_loss + config.FOM_WEIGHTS['vpil'] * norm_vpil
        costs.append(cost)

    max_valid_cost = max(costs)
    update_c_base(max_valid_cost)

    print(f"  -> C_BASE = {_C_BASE:.4f} from {len(valid)} valid simulations")
    print(f"  -> PENALTY_BETA = 9*C_BASE/pi^2 = {_PENALTY_BETA:.4f}")

    return _C_BASE


def calculate_cost(alpha, v_pi_l, max_dphi=None, weights=None, targets=None):
    """
    Calculates cost (Eq. 27). Returns negative value for BayesOpt maximization.

    Success (v_pi_l is valid): cost = w_loss*(alpha/target)^2 + w_vpil*(vpil/target)^2
    Penalty (v_pi_l is None/NaN): cost = C_BASE + beta*(pi - max_dphi)^2
    """
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    is_success = v_pi_l is not None and not np.isnan(v_pi_l)

    if is_success:
        norm_loss = (alpha / targets['loss']) ** 2
        norm_vpil = (v_pi_l / targets['vpil']) ** 2
        cost = weights['loss'] * norm_loss + weights['vpil'] * norm_vpil
    else:
        current_phi = 0.0 if (max_dphi is None or np.isnan(max_dphi)) else max_dphi
        cost = _C_BASE + _PENALTY_BETA * (np.pi - current_phi) ** 2

    return -cost
