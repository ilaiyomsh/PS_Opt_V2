# system/cost.py
# Cost function module for PIN diode phase shifter optimization
# Implements Eq. 27 from the methodology report:
#   Success case: cost = w_loss * (alpha/target_loss)^2 + w_vpil * (vpil/target_vpil)^2
#   Penalty case: cost = C_BASE + beta * (pi - max_dphi)^2

import os
import config
import pandas as pd
import numpy as np

# --- Module-level state (dynamic C_BASE per methodology Eq. 27) ---
_C_BASE = config.C_BASE_DEFAULT
_PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)


def get_c_base():
    """Return the current C_BASE value."""
    return _C_BASE


def get_penalty_beta():
    """Return the current PENALTY_BETA value."""
    return _PENALTY_BETA


def update_c_base(value):
    """
    Set C_BASE and atomically recompute PENALTY_BETA.

    Args:
        value (float): New C_BASE value
    """
    global _C_BASE, _PENALTY_BETA
    _C_BASE = value
    _PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)


def reset_c_base():
    """Reset C_BASE to the default value from config."""
    update_c_base(config.C_BASE_DEFAULT)


def calculate_c_base_from_results(result_csv_path_or_df):
    """
    Calculates C_BASE as max(Cost_valid) per methodology Eq. 27.
    C_BASE is the worst (maximum) cost among valid simulations (those that reached pi phase shift).

    Args:
        result_csv_path_or_df: Either path to result.csv file (str) or DataFrame

    Returns:
        float: C_BASE value (max valid cost, or default if no valid simulations)
    """
    # Handle both DataFrame and file path inputs
    if isinstance(result_csv_path_or_df, pd.DataFrame):
        df = result_csv_path_or_df
    else:
        if not os.path.exists(result_csv_path_or_df):
            return config.C_BASE_DEFAULT
        df = pd.read_csv(result_csv_path_or_df)

    if len(df) == 0:
        return config.C_BASE_DEFAULT

    max_valid_cost = 0.0
    valid_count = 0

    for _, row in df.iterrows():
        try:
            # Extract v_pi_l to check if simulation was valid (reached pi)
            v_pi_l = None
            if 'v_pi_l_Vmm' in row:
                v_pi_l = row['v_pi_l_Vmm']
            elif 'vpil' in row:
                v_pi_l = row['vpil']
            elif 'v_pi_l' in row:
                v_pi_l = row['v_pi_l']

            # Skip invalid simulations (didn't reach pi)
            if v_pi_l is None or np.isnan(v_pi_l):
                continue

            # Extract alpha
            alpha = None
            if 'loss_at_v_pi_dB_per_cm' in row:
                alpha = row['loss_at_v_pi_dB_per_cm']
            elif 'loss_db' in row:
                alpha = row['loss_db']
            elif 'alpha' in row:
                alpha = row['alpha']

            if alpha is None or np.isnan(alpha):
                continue

            # Calculate cost for this valid simulation (Eq. 27 top case)
            norm_loss = (alpha / config.TARGETS['loss']) ** 2
            norm_vpil = (v_pi_l / config.TARGETS['vpil']) ** 2
            cost = config.FOM_WEIGHTS['loss'] * norm_loss + config.FOM_WEIGHTS['vpil'] * norm_vpil

            if cost > max_valid_cost:
                max_valid_cost = cost
            valid_count += 1

        except Exception:
            continue

    if valid_count == 0:
        print(f"  -> No valid simulations found, using default C_BASE = {config.C_BASE_DEFAULT}")
        return config.C_BASE_DEFAULT

    # C_BASE = max(Cost_valid) per methodology
    update_c_base(max_valid_cost)

    print(f"  -> Calculated C_BASE = {_C_BASE:.4f} from {valid_count} valid simulations")
    print(f"  -> PENALTY_BETA = 9*C_BASE/pi^2 = {_PENALTY_BETA:.4f}")

    return _C_BASE


def calculate_cost(alpha, v_pi_l, max_dphi=None, weights=None, targets=None):
    """
    Calculates cost based on Report Eq. 27:
    1. If success (phi >= pi): Quadratic weighted cost.
    2. If fail (phi < pi): Penalty cost based on distance from pi.

    Args:
        alpha (float): Optical loss in dB/cm
        v_pi_l (float): V_pi*L product in V*mm
        max_dphi (float, optional): Maximum phase shift in radians (for penalty calculation)
        weights (dict, optional): Weights for different metrics.
                                 If None, uses config.FOM_WEIGHTS
        targets (dict, optional): Target values for optimization.
                                 If None, uses config.TARGETS

    Returns:
        float: Negative cost value (for maximization)

    Formula (from Report Eq. 27):
        If success: cost = w_loss * (alpha / target_loss)^2 + w_vpil * (vpil / target_vpil)^2
        If fail: cost = C_Base + beta * (pi - max_dphi)^2
        return -cost (negative for maximization - lower cost = better)
    """
    # Use default weights and targets from config if not provided
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    # Check if simulation succeeded (reached pi)
    # Success if v_pi_l is not None and not NaN
    is_success = (v_pi_l is not None) and (not np.isnan(v_pi_l))

    if is_success:
        # Case 1: Success (Eq. 27 Top) - Quadratic weighted cost
        norm_loss = (alpha / targets['loss']) ** 2  # Square term
        norm_vpil = (v_pi_l / targets['vpil']) ** 2  # Square term
        cost = weights['loss'] * norm_loss + weights['vpil'] * norm_vpil
    else:
        # Case 2: Penalty (Eq. 27 Bottom)
        # Cost = C_Base + beta * (pi - max_dphi)^2
        # Uses dynamic _C_BASE and _PENALTY_BETA calculated from valid results
        if max_dphi is None or np.isnan(max_dphi):
            current_phi = 0.0
        else:
            current_phi = max_dphi

        penalty_term = (np.pi - current_phi) ** 2
        cost = _C_BASE + _PENALTY_BETA * penalty_term

    # Return negative because BayesOpt maximizes, and we want to minimize Cost
    return -cost
