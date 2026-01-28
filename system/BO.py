# system/BO.py
# Bayesian Optimization module for parameter optimization
# Uses Gaussian Process model, loss function, and Acquisition Function (UCB)

import os
import config
import results_archive
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

# Dynamic C_BASE value (calculated from max valid cost per methodology Eq. 27)
# This is set by train_optimizer() and used by calculate_loss_function()
_C_BASE = config.C_BASE_DEFAULT
_PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)


def _calculate_c_base_from_results(result_csv_path_or_df):
    """
    Calculates C_BASE as max(Cost_valid) per methodology Eq. 27.
    C_BASE is the worst (maximum) cost among valid simulations (those that reached π phase shift).

    Args:
        result_csv_path_or_df: Either path to result.csv file (str) or DataFrame

    Returns:
        float: C_BASE value (max valid cost, or default if no valid simulations)
    """
    global _C_BASE, _PENALTY_BETA

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
            # Extract v_pi_l to check if simulation was valid (reached π)
            v_pi_l = None
            if 'v_pi_l_Vmm' in row:
                v_pi_l = row['v_pi_l_Vmm']
            elif 'vpil' in row:
                v_pi_l = row['vpil']
            elif 'v_pi_l' in row:
                v_pi_l = row['v_pi_l']

            # Skip invalid simulations (didn't reach π)
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
    _C_BASE = max_valid_cost
    _PENALTY_BETA = (9 * _C_BASE) / (np.pi ** 2)

    print(f"  -> Calculated C_BASE = {_C_BASE:.4f} from {valid_count} valid simulations")
    print(f"  -> PENALTY_BETA = 9*C_BASE/π² = {_PENALTY_BETA:.4f}")

    return _C_BASE


def train_optimizer(result_csv_path=None, use_archive=True):
    """
    Trains the Bayesian Optimizer with prior data from result.csv and archive.

    Args:
        result_csv_path (str, optional): Path to result.csv file.
                                        If None, uses config.RESULTS_CSV_FILE
        use_archive (bool): If True, loads all archived results as well.
                           This enables cumulative learning across runs.

    Returns:
        BayesianOptimization: Trained optimizer object

    Process:
        1. Loads result.csv with current simulation results
        2. Optionally loads all archived results
        3. Merges and deduplicates
        4. Registers all data points with the optimizer
    """
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    # Load results - either from archive or just current file
    if use_archive:
        df = results_archive.load_all_results_for_bo()
        if df.empty:
            # Fallback to current file only
            if not os.path.exists(result_csv_path):
                raise FileNotFoundError(f"Results file not found: {result_csv_path}. Run initial simulations first.")
            df = pd.read_csv(result_csv_path)
    else:
        # Load only from specified file
        if not os.path.exists(result_csv_path):
            raise FileNotFoundError(f"Results file not found: {result_csv_path}. Run initial simulations first.")
        df = pd.read_csv(result_csv_path)
        print(f"  -> Loaded {len(df)} prior data points from {result_csv_path}")

    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")

    # Calculate dynamic C_BASE from valid results (per methodology Eq. 27)
    _calculate_c_base_from_results(df)
    
    # Convert SWEEP_PARAMETERS to pbounds format for BayesianOptimization
    # config has: {'w_r': {'min': 350e-9, 'max': 450e-9, 'unit': 'm'}}
    # BayesianOptimization needs: {'w_r': (350e-9, 450e-9)}
    pbounds = {}
    param_names = list(config.SWEEP_PARAMETERS.keys())
    for param_name, param_config in config.SWEEP_PARAMETERS.items():
        pbounds[param_name] = (param_config['min'], param_config['max'])
    
    # Create acquisition function with kappa from config
    utility = UpperConfidenceBound(kappa=config.BO_KAPPA)
    
    # Create optimizer
    # Note: f=None because we use suggest() instead of maximize()
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=42,
        acquisition_function=utility,
        verbose=2,
    )
    
    # Register each prior data point
    registered_count = 0
    skipped_count = 0
    
    for index, row in df.iterrows():
        # Extract parameter values
        try:
            params = {}
            for param_name in param_names:
                if param_name not in row:
                    print(f"  [WARNING] Parameter '{param_name}' not found in row {index}. Skipping.")
                    break
                params[param_name] = row[param_name]
            else:
                # Extract loss and vpil from results
                # Check for column names (may vary based on result.csv structure)
                alpha = None
                v_pi_l = None
                max_dphi = None
                
                if 'loss_at_v_pi_dB_per_cm' in row:
                    alpha = row['loss_at_v_pi_dB_per_cm']
                elif 'loss_db' in row:
                    alpha = row['loss_db']
                elif 'alpha' in row:
                    alpha = row['alpha']
                
                if 'v_pi_l_Vmm' in row:
                    v_pi_l = row['v_pi_l_Vmm']
                elif 'vpil' in row:
                    v_pi_l = row['vpil']
                elif 'v_pi_l' in row:
                    v_pi_l = row['v_pi_l']
                
                # Extract max_dphi (with default 0.0 if not present - for backwards compatibility)
                if 'max_dphi_rad' in row:
                    max_dphi = row['max_dphi_rad']
                else:
                    max_dphi = 0.0
                
                # Check if we have valid alpha and v_pi_l (NaN is allowed - it's treated as failure)
                if alpha is None or v_pi_l is None:
                    print(f"  [WARNING] Missing alpha or v_pi_l in row {index}. Skipping.")
                    skipped_count += 1
                    continue
                
                # Handle NaN values by converting to None (for calculate_loss_function)
                if np.isnan(alpha):
                    alpha = None
                if np.isnan(v_pi_l):
                    v_pi_l = None
                if max_dphi is not None and np.isnan(max_dphi):
                    max_dphi = 0.0
                
                # Calculate cost (function handles both success and failure cases)
                cost = calculate_loss_function(alpha, v_pi_l, max_dphi)
                
                # Register with optimizer
                optimizer.register(params=params, target=cost)
                registered_count += 1
                
        except Exception as e:
            print(f"  [WARNING] Error processing row {index}: {e}. Skipping.")
            skipped_count += 1
            continue
    
    print(f"  -> Registered {registered_count} data points with optimizer")
    if skipped_count > 0:
        print(f"  -> Skipped {skipped_count} rows due to errors or missing data")

    return optimizer


def _is_duplicate_params(new_params, existing_df, tolerance=None):
    """
    Check if suggested parameters are too similar to existing results.

    Args:
        new_params (dict): Suggested parameter values
        existing_df (pd.DataFrame): DataFrame with existing results
        tolerance (float): Relative tolerance for considering params as duplicate (default 1%)

    Returns:
        bool: True if duplicate found, False otherwise
    """
    if tolerance is None:
        tolerance = config.DUPLICATE_TOLERANCE

    if existing_df is None or len(existing_df) == 0:
        return False

    param_names = list(config.SWEEP_PARAMETERS.keys())

    for _, row in existing_df.iterrows():
        is_match = True
        for param in param_names:
            if param not in new_params or param not in row:
                is_match = False
                break

            old_val = row[param]
            new_val = new_params[param]

            # Calculate relative difference
            if old_val == 0:
                rel_diff = abs(new_val)
            else:
                rel_diff = abs((new_val - old_val) / old_val)

            if rel_diff > tolerance:
                is_match = False
                break

        if is_match:
            return True

    return False


def get_next_sample(optimizer, result_csv_path=None, max_retries=5):
    """
    Predicts next parameter set to sample using Bayesian Optimization.
    Includes duplicate detection to avoid re-sampling similar points.

    Args:
        optimizer (BayesianOptimization): Trained BayesianOptimization object
        result_csv_path (str, optional): Path to result.csv file (for reference).
                                        If None, uses config.RESULTS_CSV_FILE
        max_retries (int): Maximum attempts to find non-duplicate point

    Returns:
        dict: Dictionary containing next parameter values to sample
              Keys match parameter names in config.SWEEP_PARAMETERS

    Process:
        1. Uses Gaussian Process model to predict objective function
        2. Uses Acquisition Function (UCB) to select next point
        3. Checks for duplicates and retries if needed
        4. Returns parameter dictionary for next simulation
    """
    if optimizer is None:
        raise ValueError("Optimizer is None. Train the optimizer first.")

    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    # Load ALL existing results (current + archived) for duplicate checking
    existing_df = results_archive.load_all_results_for_bo()

    try:
        for retry in range(max_retries):
            # Use suggest() to get the next point to sample
            next_point = optimizer.suggest()

            if next_point is None:
                print("  [WARNING] Optimizer suggest() returned None")
                return None

            # Verify all parameters are present and within bounds
            param_names = list(config.SWEEP_PARAMETERS.keys())
            valid = True

            for param_name in param_names:
                if param_name not in next_point:
                    print(f"  [WARNING] Parameter '{param_name}' missing from suggested point")
                    valid = False
                    break

                # Clip to bounds if needed
                min_val = config.SWEEP_PARAMETERS[param_name]['min']
                max_val = config.SWEEP_PARAMETERS[param_name]['max']

                if next_point[param_name] < min_val:
                    print(f"  [WARNING] Clipping {param_name} from {next_point[param_name]} to {min_val}")
                    next_point[param_name] = min_val
                elif next_point[param_name] > max_val:
                    print(f"  [WARNING] Clipping {param_name} from {next_point[param_name]} to {max_val}")
                    next_point[param_name] = max_val

            if not valid:
                continue

            # Check for duplicates
            if _is_duplicate_params(next_point, existing_df):
                print(f"  [INFO] Suggested point is duplicate of existing result (retry {retry + 1}/{max_retries})")
                # Register this point with a dummy value to force optimizer to explore elsewhere
                optimizer.register(params=next_point, target=-1e10)
                continue

            print(f"  -> Next suggested point: {next_point}")
            return next_point

        # If all retries exhausted, return the last suggestion anyway
        print(f"  [WARNING] Could not find non-duplicate point after {max_retries} retries. Using last suggestion.")
        return next_point

    except Exception as e:
        print(f"  [ERROR] Failed to get next sample: {e}")
        return None


def calculate_loss_function(alpha, v_pi_l, max_dphi=None, weights=None, targets=None):
    """
    Calculates cost based on Report Eq. 27:
    1. If success (phi >= pi): Quadratic weighted cost.
    2. If fail (phi < pi): Penalty cost based on distance from pi.
    
    Args:
        alpha (float): Optical loss in dB/cm
        v_pi_l (float): V_π*L product in V*mm
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


def get_best_result(result_csv_path=None):
    """
    Returns the best result found so far from results CSV.

    Args:
        result_csv_path (str, optional): Path to result.csv file.
                                        If None, uses config.RESULTS_CSV_FILE

    Returns:
        dict: Best result row as dictionary, or None if no valid results
    """
    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE

    if not os.path.exists(result_csv_path):
        return None

    df = pd.read_csv(result_csv_path)

    if len(df) == 0:
        return None

    # Ensure C_BASE is calculated from current results
    _calculate_c_base_from_results(result_csv_path)
    
    # Calculate cost for each row and find the best (lowest cost = highest negative value)
    best_cost = float('inf')
    best_row = None
    
    for index, row in df.iterrows():
        try:
            alpha = None
            v_pi_l = None
            max_dphi = None
            
            if 'loss_at_v_pi_dB_per_cm' in row:
                alpha = row['loss_at_v_pi_dB_per_cm']
            if 'v_pi_l_Vmm' in row:
                v_pi_l = row['v_pi_l_Vmm']
            
            # Extract max_dphi (with default 0.0 if not present - for backwards compatibility)
            if 'max_dphi_rad' in row:
                max_dphi = row['max_dphi_rad']
            else:
                max_dphi = 0.0
            
            # Check if we have valid alpha and v_pi_l (NaN is allowed - it's treated as failure)
            if alpha is None or v_pi_l is None:
                continue
            
            # Handle NaN values by converting to None (for calculate_loss_function)
            if np.isnan(alpha):
                alpha = None
            if np.isnan(v_pi_l):
                v_pi_l = None
            if max_dphi is not None and np.isnan(max_dphi):
                max_dphi = 0.0
            
            # Calculate cost (positive value, lower is better)
            cost = -calculate_loss_function(alpha, v_pi_l, max_dphi)
            
            if cost < best_cost:
                best_cost = cost
                best_row = row.to_dict()
                
        except Exception:
            continue
    
    return best_row

