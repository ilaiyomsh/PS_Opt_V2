# system/run_simulation.py
# Simulation runner — orchestration and data I/O
# Coordinates sim_handler and data_processor, handles CSV read/write

import os
import time
import traceback
import json
from datetime import datetime
import pandas as pd
import numpy as np

import config
import cost as cost_module
import sim_handler
import data_processor


def debug_prompt(message):
    """Prompts user for confirmation in DEBUG mode. Returns True to continue."""
    if not config.DEBUG:
        return True
    print(f"\n[DEBUG] {message}")
    response = input("[DEBUG] Press Enter to continue, 's' to skip: ").strip().lower()
    if response == 's':
        print("[DEBUG] Step skipped by user.")
        return False
    return True


def cooling_delay(skip=False):
    """Pauses between simulation runs. Skipped if delay is 0 or skip=True."""
    if skip or config.DELAY_BETWEEN_RUNS <= 0:
        return

    delay = config.DELAY_BETWEEN_RUNS
    print(f"  [Cooling delay: {delay // 60}:{delay % 60:02d} minutes...]")

    remaining = delay
    while remaining > 0:
        sleep_time = min(60, remaining)
        time.sleep(sleep_time)
        remaining -= sleep_time
        if remaining > 0:
            print(f"    Cooling: {remaining // 60}:{remaining % 60:02d} remaining...")

    print("  [Cooling delay complete]")


# ============================================================================
# CSV I/O
# ============================================================================

def _save_to_csv(filename, result_df, columns=None):
    """Appends a single-row DataFrame to CSV, creating the file if needed."""
    # If columns is specified, only save those columns (if they exist in result_df).
    if columns is not None:
        cols_to_use = [c for c in columns if c in result_df.columns]
        df_to_save = result_df[cols_to_use].copy()
    else:
        df_to_save = result_df.copy()

    if not os.path.exists(filename):
        df_to_save.to_csv(filename, index=False, mode='w', float_format='%.6e')
    else:
        # Match existing column order
        try:
            existing_header = pd.read_csv(filename, nrows=0).columns.tolist()
            for col in existing_header:
                if col not in df_to_save.columns:
                    df_to_save[col] = np.nan
            df_to_save = df_to_save[existing_header]
        except Exception:
            pass
        df_to_save.to_csv(filename, index=False, mode='a', header=False, float_format='%.6e')


def save_single_result_to_csv(filename, current_result):
    """Saves a result to both minimal (result.csv) and full (result_full.csv) CSV files."""
    sim_id = current_result.get('sim_id', 'N/A')
    result_df = pd.DataFrame([current_result])

    # Reorder: sim_id, input params, output metrics
    input_params = list(config.SWEEP_PARAMETERS.keys())
    output_metrics = [c for c in result_df.columns if c not in input_params and c != 'sim_id']
    column_order = ['sim_id'] + input_params + output_metrics
    if all(c in result_df.columns for c in column_order):
        result_df = result_df[column_order]

    # Save minimal file
    _save_to_csv(filename, result_df, columns=config.MINIMAL_RESULT_COLUMNS)

    # Save full file
    _save_to_csv(config.RESULTS_FULL_CSV_FILE, result_df)
    print(f"  -> Result for sim_id {sim_id} saved to {os.path.basename(filename)} and {os.path.basename(config.RESULTS_FULL_CSV_FILE)}")


def save_error_to_csv(sim_id, stage, error, params=None):
    """Saves error details to errors.csv."""
    error_record = {
        'sim_id': sim_id,
        'stage': stage,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'params': json.dumps(params) if params else ''
    }

    error_df = pd.DataFrame([error_record])
    file_exists = os.path.exists(config.ERRORS_CSV_FILE)

    error_df.to_csv(config.ERRORS_CSV_FILE, index=False,
                    mode='a' if file_exists else 'w',
                    header=not file_exists)

    print(f"  -> Error for sim_id {sim_id} (stage: {stage}) saved to {config.ERRORS_CSV_FILE}")


# ============================================================================
# Simulation Orchestration
# ============================================================================

def run_init_file(params_csv_path=None, logger_func=None):
    """
    Runs simulations for all parameter sets in params.csv.

    Args:
        params_csv_path: Path to params.csv. Defaults to config.PARAMS_CSV_FILE.
        logger_func: Optional function for logging messages.

    Returns:
        str: Path to the results CSV file
    """
    if params_csv_path is None:
        params_csv_path = config.PARAMS_CSV_FILE

    print(f"\n{'='*60}")
    print("Running Initial Simulations")
    print(f"{'='*60}")
    print(f"Parameters file: {params_csv_path}")

    params_df = pd.read_csv(params_csv_path, skiprows=[1])
    print(f"Loaded {len(params_df)} parameter sets")

    successful = 0
    failed = 0
    total_sims = len(params_df)

    for idx, row in params_df.iterrows():
        sim_id = int(row['sim_id'])
        is_last = (idx == len(params_df) - 1)
        print(f"\n--- Simulation {sim_id}/{total_sims} ---")

        if logger_func:
            logger_func(f"\n  [{sim_id}/{total_sims}] sim_id={sim_id}")
            logger_func(f"        Input:  w_r={row.get('w_r', 0)*1e9:.1f}nm, "
                       f"h_si={row.get('h_si', 0)*1e9:.1f}nm, "
                       f"S={row.get('S', 0)*1e9:.1f}nm, "
                       f"doping={row.get('doping', 0):.2e}, "
                       f"lambda={row.get('lambda', 0)*1e9:.0f}nm, "
                       f"L={row.get('length', 0)*1e3:.2f}mm")

        if not debug_prompt(f"Ready to run simulation {sim_id}"):
            print(f"[INFO] Simulation {sim_id} skipped by user.")
            continue

        try:
            result = run_row(row, sim_id=sim_id, is_last=is_last)

            if result is not None:
                successful += 1
                print(f"  [OK] Simulation {sim_id} completed successfully.")
                if logger_func:
                    _log_result(logger_func, result)
            else:
                failed += 1
                print(f"  [FAIL] Simulation {sim_id} returned None.")
                if logger_func:
                    logger_func(f"        Output: FAILED (simulation returned None)")

        except Exception as e:
            failed += 1
            print(f"  [ERROR] Simulation {sim_id} failed: {e}")
            if logger_func:
                logger_func(f"        Output: ERROR - {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Initial Simulations Summary: {successful}/{total_sims} successful, {failed} failed")
    print(f"Results saved to: {config.RESULTS_CSV_FILE}")
    print(f"{'='*60}")

    # Error summary
    if os.path.exists(config.ERRORS_CSV_FILE):
        errors_df = pd.read_csv(config.ERRORS_CSV_FILE)
        if len(errors_df) > 0:
            print(f"\nErrors by stage:")
            for stage, count in errors_df['stage'].value_counts().items():
                print(f"  {stage}: {count}")
            print(f"Failed simulation IDs: {sorted(errors_df['sim_id'].unique())}")

    return config.RESULTS_CSV_FILE


def _log_result(logger_func, result):
    """Log simulation result details."""
    v_pi = result.get('v_pi_V', float('nan'))
    charge_t = result.get('charge_time_s', 0)
    fde_t = result.get('fde_time_s', 0)
    total_t = result.get('total_time_s', 0)
    logger_func(f"        Timing: CHARGE={charge_t:.1f}s, FDE={fde_t:.1f}s, Total={total_t:.1f}s")

    intrinsic_w = result.get('intrinsic_width_m', 0)
    logger_func(f"        Geometry: intrinsic_width={intrinsic_w*1e9:.1f}nm (w_r + 2*S)")

    if not np.isnan(v_pi):
        logger_func(f"        Output: V_pi={v_pi:.2f}V, "
                   f"V_pi*L={result.get('v_pi_l_Vmm', 0):.3f} V*mm, "
                   f"Loss={result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm, "
                   f"C={result.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm")
        logger_func(f"        Cost:   {result.get('cost', 0):.4f} "
                   f"(norm_loss={result.get('norm_loss', 0):.2f}, "
                   f"norm_vpil={result.get('norm_vpil', 0):.2f})")
        logger_func(f"        Ranges: Loss=[{result.get('loss_min_dB_per_cm', 0):.1f}, {result.get('loss_max_dB_per_cm', 0):.1f}] dB/cm, "
                   f"C=[{result.get('C_min_pF_per_cm', 0):.1f}, {result.get('C_max_pF_per_cm', 0):.1f}] pF/cm")
    else:
        logger_func(f"        Output: V_pi=NaN (did not reach pi), max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
        logger_func(f"        Cost:   {result.get('cost', 0):.4f} "
                   f"(norm_loss={result.get('norm_loss', 0):.2f}, "
                   f"norm_vpil={result.get('norm_vpil', 0):.2f})")


def run_row(row, sim_id=None, is_last=False):
    """
    Runs a single simulation and saves the result to CSV.

    Args:
        row: Dict or pd.Series with parameter values
        sim_id: Simulation ID. If None, extracted from row['sim_id'].
        is_last: If True, skip cooling delay after this simulation.

    Returns:
        dict with simulation results, or None on failure
    """
    params = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    if sim_id is None:
        sim_id = int(params.get('sim_id', 0))

    print(f"\n{'='*50}")
    print(f"Running simulation {sim_id}")
    print(f"{'='*50}")

    sim_start_time = time.time()

    # 1. Run simulation
    try:
        raw_df, raw_csv_path, timing = sim_handler.run_full_simulation(params, sim_id=sim_id)
    except sim_handler.SimulationError as e:
        save_error_to_csv(sim_id, e.stage, e.original_error or e, params)
        return None

    # 2. Process data
    try:
        V_cap, C_total_pF_cm = data_processor.process_charge_data(
            raw_df['V'].values, raw_df['n'].values, raw_df['p'].values)

        neff = raw_df['neff_re'].values + 1j * raw_df['neff_im'].values
        d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi = data_processor.process_optical_data(
            neff, float(params['length']), float(params['lambda']))
    except Exception as e:
        save_error_to_csv(sim_id, "DATA_PROCESSING", e, params)
        return None

    # 3. Update raw CSV with processed columns
    raw_df['d_neff'] = d_neff
    raw_df['d_phi'] = d_phi
    raw_df['C_total_pF_cm'] = C_total_pF_cm
    raw_df.to_csv(raw_csv_path, index=False)

    # 4. Calculate derived metrics
    try:
        result = _build_result(
            sim_id, params, timing, V_cap, C_total_pF_cm,
            d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi,
            time.time() - sim_start_time)
    except Exception as e:
        save_error_to_csv(sim_id, "RESULTS_CALC", e, params)
        return None

    # 5. Save result
    if debug_prompt("Ready to save result to CSV"):
        save_single_result_to_csv(config.RESULTS_CSV_FILE, result)

    # 6. Cooling delay
    cooling_delay(skip=is_last)

    return result


def _build_result(sim_id, params, timing, V_cap, C_total_pF_cm,
                  d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi, total_time):
    """Builds the result dict from processed data."""
    V_fde = np.linspace(0, config.V_MAX, len(d_neff))

    w_r = float(params['w_r'])
    S = float(params['S'])
    is_valid = not np.isnan(v_pi)

    if is_valid:
        loss_at_v_pi = np.interp(v_pi, V_fde, alpha_dB_per_cm)
        C_at_v_pi = np.interp(v_pi, V_cap, C_total_pF_cm)
        v_pi_l = v_pi * float(params['length']) * 1e3  # V*mm
    else:
        loss_at_v_pi = np.max(alpha_dB_per_cm)                    # worst-case from actual sim
        C_at_v_pi = np.nan
        v_pi_l = config.V_MAX * float(params['length']) * 1e3     # V_MAX * L in V·mm

    # Cost — piecewise quadratic penalty based on max_dphi
    neg_cost = cost_module.calculate_cost(alpha=loss_at_v_pi, v_pi_l=v_pi_l, max_dphi=max_dphi)
    cost_value = -neg_cost  # Positive for CSV storage

    norm_loss = loss_at_v_pi / config.TARGETS['loss']
    norm_vpil = v_pi_l / config.TARGETS['vpil']

    # Print summary
    if is_valid:
        print(f"\n  Result: V_pi*L = {v_pi_l:.4f} V*mm, Loss = {loss_at_v_pi:.2f} dB/cm, C = {C_at_v_pi:.2f} pF/cm")
        print(f"  Cost: {cost_value:.4f} (norm_loss={norm_loss:.2f}, norm_vpil={norm_vpil:.2f})")
    else:
        print(f"\n  Result: V_pi not reached (max_dphi={max_dphi:.2f} rad, worst_loss={loss_at_v_pi:.2f} dB/cm)")
        print(f"  Cost: {cost_value:.4f} (norm_loss={norm_loss:.2f}, norm_vpil={norm_vpil:.2f})")
    print(f"  Timing: CHARGE={timing.get('charge_time', 0):.1f}s, FDE={timing.get('fde_time', 0):.1f}s, Total={total_time:.1f}s")

    return {
        'sim_id': sim_id,
        **params,
        'v_pi_V': v_pi,
        'v_pi_l_Vmm': v_pi_l,
        'loss_at_v_pi_dB_per_cm': loss_at_v_pi,
        'C_at_v_pi_pF_per_cm': C_at_v_pi,
        'max_dphi_rad': max_dphi,
        'is_valid': is_valid,
        'norm_loss': norm_loss,
        'norm_vpil': norm_vpil,
        'cost': cost_value,
        'charge_time_s': timing.get('charge_time', 0),
        'fde_time_s': timing.get('fde_time', 0),
        'total_time_s': total_time,
        'intrinsic_width_m': w_r + 2 * S,
        'loss_min_dB_per_cm': np.min(alpha_dB_per_cm),
        'loss_max_dB_per_cm': np.max(alpha_dB_per_cm),
        'C_min_pF_per_cm': np.min(C_total_pF_cm),
        'C_max_pF_per_cm': np.max(C_total_pF_cm),
    }
