# system/main.py
# Main entry point for PS_Opt_V2 optimization system
# Coordinates LHS sampling, simulation runs, and Bayesian Optimization

import sys
import os
import logging
from datetime import datetime
import config
import LHS
import run_simulation
import BO
import pandas as pd
import numpy as np


# ============================================================================
# Logging
# ============================================================================

def setup_logging():
    """Sets up dual logging to both console and file. Returns log file path."""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"test_run_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return log_path


def log(message):
    """Log with timestamp."""
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def log_raw(message):
    """Log without timestamp."""
    logging.info(message)


def debug_prompt(message):
    """Prompt for confirmation in DEBUG mode. Returns True to continue."""
    if not config.DEBUG:
        return True
    print(f"\n[DEBUG] {message}")
    response = input("[DEBUG] Press Enter to continue, 's' to skip, 'q' to quit: ").strip().lower()
    if response == 'q':
        sys.exit(0)
    return response != 's'


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Orchestrates the optimization workflow:
        1. Generate LHS samples (or skip if SKIP_INITIAL_SIMS)
        2. Run initial simulations (or skip if SKIP_INITIAL_SIMS)
        3. Train BO once, then loop: suggest → simulate → register
        4. Output final results
    """
    log_path = setup_logging()
    config.RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    log_raw("=" * 80)
    log_raw(f"PS_Opt_V2 TEST RUN - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_raw("=" * 80)
    log_raw("Configuration:")
    log_raw(f"  - LHS Samples: {config.LHS_N_SAMPLES}")
    log_raw(f"  - Max BO Iterations: {config.MAX_ITERATIONS}")
    log_raw(f"  - Lumerical API: {config.LUMERICAL_API_PATH}")
    log_raw(f"  - SKIP_LHS: {config.SKIP_LHS}")
    log_raw(f"  - SKIP_INITIAL_SIMS: {config.SKIP_INITIAL_SIMS}")
    log_raw(f"  - HIDE_GUI: {config.HIDE_GUI}")
    log_raw(f"  - DEBUG: {config.DEBUG}")
    log_raw(f"  - Delay Between Runs: {config.DELAY_BETWEEN_RUNS} seconds")
    log_raw(f"  - TARGETS: {config.TARGETS}")
    log_raw(f"  - FOM_WEIGHTS: {config.FOM_WEIGHTS}")
    log_raw(f"  - BO_KAPPA: {config.BO_KAPPA}")
    log_raw(f"  - Log file: {log_path}")
    
    # Log discrete parameter configuration
    enabled_discrete = [k for k, v in config.DISCRETE_PARAMETERS.items() if v.get('enabled', False)]
    if enabled_discrete:
        log_raw(f"  - Discrete Parameters Enabled: {enabled_discrete}")
        for param in enabled_discrete:
            grid = config.DISCRETE_PARAMETERS[param]['values']
            log_raw(f"    • {param}: {len(grid)} values from {grid[0]:.4e} to {grid[-1]:.4e}")
    else:
        log_raw(f"  - Discrete Parameters: None enabled (continuous optimization)")
    
    log_raw("")

    # --- Steps 1 & 2: Get initial results ---
    if config.SKIP_INITIAL_SIMS:
        _skip_initial(config.RESULTS_CSV_FILE)
    else:
        _run_initial(config.PARAMS_CSV_FILE, config.RESULTS_CSV_FILE)

    # Check we have results before proceeding
    if not os.path.exists(config.RESULTS_CSV_FILE):
        log_raw("[ERROR] No results file. Cannot start BO.")
        return

    # --- Step 3: Bayesian Optimization Loop ---
    _run_bo_loop(config.RESULTS_CSV_FILE)

    # --- Step 4: Final Results ---
    _print_final_results(config.RESULTS_CSV_FILE)

    # Duration
    duration = datetime.now() - start_time
    log_raw("")
    log_raw("=" * 80)
    log_raw("TEST COMPLETED")
    log_raw(f"  Total duration: {duration}")
    log_raw(f"  Log saved to: {log_path}")
    log_raw("=" * 80)


# ============================================================================
# Stage functions
# ============================================================================

def _skip_initial(results_path):
    """Load existing result.csv instead of running LHS + simulations."""
    log_raw("-" * 80)
    log("SKIPPING STAGES 1 & 2 (SKIP_INITIAL_SIMS=True)")
    log_raw("-" * 80)

    if not os.path.exists(results_path):
        log_raw(f"  [ERROR] result.csv not found at {results_path}")
        log_raw("  [ERROR] Run prepare_initial_data.py first, or set SKIP_INITIAL_SIMS=False.")
        return

    results_df = pd.read_csv(results_path)
    valid_count = results_df['v_pi_V'].notna().sum()
    log_raw(f"  Using existing result.csv: {len(results_df)} rows ({valid_count} valid)")
    log_raw("")


def _run_initial(params_path, results_path):
    """Run LHS sampling + initial simulations."""
    # --- Step 1: LHS ---
    log_raw("-" * 80)
    log("STAGE 1: LHS SAMPLING")
    log_raw("-" * 80)

    if config.SKIP_LHS:
        log_raw(f"  SKIP_LHS=True. Using existing: {params_path}")
        if not os.path.exists(params_path):
            log_raw("  [ERROR] params.csv not found. Set SKIP_LHS=False to generate new samples.")
            return
    else:
        if not debug_prompt("Ready to generate LHS samples"):
            return
        log_raw(f"  Generating {config.LHS_N_SAMPLES} samples using '{config.LHS_SAMPLING_METHOD}' method...")
        LHS.generate_lhs_samples()
        log_raw(f"  Saved to: {params_path}")
    log_raw("")

    # --- Step 2: Initial Simulations ---
    log_raw("-" * 80)
    log("STAGE 2: INITIAL SIMULATIONS (LHS)")
    log_raw("-" * 80)

    if not debug_prompt("Ready to run initial simulations"):
        return

    run_simulation.run_init_file(params_csv_path=params_path, logger_func=log_raw)
    log_raw(f"  Initial simulations completed. Results saved to {results_path}")

    # Log summary
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        valid_count = results_df['v_pi_V'].notna().sum()
        log_raw(f"  Summary: {valid_count}/{len(results_df)} valid (reached pi)")
        if valid_count > 0 and 'cost' in results_df.columns:
            valid_results = results_df[results_df['v_pi_V'].notna()]
            best_idx = valid_results['cost'].idxmin()
            log_raw(f"  Best initial: sim_id={int(valid_results.loc[best_idx, 'sim_id'])}, "
                    f"Cost={valid_results.loc[best_idx, 'cost']:.3f}")
    log_raw("")


def _run_bo_loop(results_path):
    """Train BO once, then run suggest → simulate → register loop."""
    log_raw("-" * 80)
    log("STAGE 3: BAYESIAN OPTIMIZATION")
    log_raw("-" * 80)

    if not debug_prompt("Ready to start Bayesian Optimization loop"):
        return

    bo_sim_id = len(pd.read_csv(results_path)) + 1

    # Train optimizer ONCE with all existing results
    log_raw("  Training optimizer...")
    optimizer = BO.train_optimizer(result_csv_path=results_path)

    for iteration in range(1, config.MAX_ITERATIONS + 1):
        log_raw(f"\n  --- Iteration {iteration}/{config.MAX_ITERATIONS} ---")

        if not debug_prompt(f"Ready for BO iteration {iteration}"):
            continue

        try:
            # Suggest next point
            next_params = BO.get_next_sample(optimizer)
            if next_params is None:
                log_raw("  [WARNING] Optimizer returned None. Stopping.")
                break

            log_raw(f"  [{bo_sim_id}] sim_id={bo_sim_id}")
            log_raw(f"      Suggested: w_r={next_params.get('w_r', 0)*1e9:.1f}nm, "
                    f"h_si={next_params.get('h_si', 0)*1e9:.1f}nm, "
                    f"S={next_params.get('S', 0)*1e9:.1f}nm, "
                    f"doping={next_params.get('doping', 0):.2e}, "
                    f"lambda={next_params.get('lambda', 0)*1e9:.0f}nm, "
                    f"L={next_params.get('length', 0)*1e3:.2f}mm")

            # Run simulation
            next_params['sim_id'] = bo_sim_id
            is_last = (iteration == config.MAX_ITERATIONS)
            result = run_simulation.run_row(next_params, sim_id=bo_sim_id, is_last=is_last)

            if result is None:
                log_raw(f"      Output: FAILED (simulation returned None)")
                bo_sim_id += 1
                continue

            # Register with optimizer
            BO.register_result(optimizer, result, result['cost'])

            # Log result
            _log_bo_result(result)
            bo_sim_id += 1

            # Show best so far
            best = BO.get_best_result(results_path)
            if best:
                log_raw(f"      Best so far: sim_id={int(best['sim_id'])}, "
                        f"V_pi*L={best['v_pi_l_Vmm']:.3f} V*mm")

        except KeyboardInterrupt:
            log_raw("\n  [INFO] Optimization interrupted by user.")
            break
        except Exception as e:
            log_raw(f"  [ERROR] BO iteration {iteration}: {e}")
            bo_sim_id += 1
            continue

    log_raw("")


def _log_bo_result(result):
    """Log a single BO iteration result."""
    v_pi = result.get('v_pi_V', float('nan'))
    if not np.isnan(v_pi):
        log_raw(f"      Output: V_pi={v_pi:.2f}V, "
                f"V_pi*L={result.get('v_pi_l_Vmm', 0):.3f} V*mm, "
                f"Loss={result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm, "
                f"C={result.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm, "
                f"max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
    else:
        log_raw(f"      Output: V_pi=NaN (did not reach pi), "
                f"max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
        log_raw(f"      Cost: {result.get('cost', 0):.4f}")


def _print_final_results(results_path):
    """Print final optimization results summary."""
    log_raw("-" * 80)
    log("STAGE 4: FINAL RESULTS")
    log_raw("-" * 80)

    if not os.path.exists(results_path):
        log_raw("  [WARNING] Results file not found.")
        return

    results_df = pd.read_csv(results_path)
    valid_count = results_df['v_pi_V'].notna().sum()
    invalid_count = len(results_df) - valid_count

    log_raw(f"  Total simulations run: {len(results_df)}")
    log_raw(f"  Valid simulations:     {valid_count} (reached pi phase shift)")
    log_raw(f"  Invalid simulations:   {invalid_count}")
    log_raw("")

    if len(results_df) == 0:
        return

    best = BO.get_best_result(results_path)
    if not best:
        log_raw("  [WARNING] Could not determine best result")
        return

    # Table
    log_raw("  BEST RESULT:")
    log_raw("  +-----------+------------------+")
    log_raw(f"  | sim_id    | {int(best.get('sim_id', 0)):<16} |")
    log_raw("  +-----------+------------------+")
    log_raw("  | INPUTS                       |")
    log_raw("  +-----------+------------------+")
    log_raw(f"  | w_r       | {best.get('w_r', 0)*1e9:.1f} nm".ljust(30) + "|")
    log_raw(f"  | h_si      | {best.get('h_si', 0)*1e9:.1f} nm".ljust(30) + "|")
    log_raw(f"  | S         | {best.get('S', 0)*1e9:.1f} nm".ljust(30) + "|")
    log_raw(f"  | doping    | {best.get('doping', 0):.2e} cm^-3".ljust(30) + "|")
    log_raw(f"  | lambda    | {best.get('lambda', 0)*1e9:.0f} nm".ljust(30) + "|")
    log_raw(f"  | length    | {best.get('length', 0)*1e3:.2f} mm".ljust(30) + "|")
    log_raw("  +-----------+------------------+")
    log_raw("  | OUTPUTS                      |")
    log_raw("  +-----------+------------------+")
    log_raw(f"  | V_pi      | {best.get('v_pi_V', 0):.2f} V".ljust(30) + "|")
    log_raw(f"  | V_pi*L    | {best.get('v_pi_l_Vmm', 0):.3f} V*mm".ljust(30) + "|")
    log_raw(f"  | Loss      | {best.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm".ljust(30) + "|")
    log_raw(f"  | C         | {best.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm".ljust(30) + "|")
    log_raw(f"  | max_dphi  | {best.get('max_dphi_rad', 0):.2f} rad".ljust(30) + "|")
    log_raw("  +-----------+------------------+")


if __name__ == "__main__":
    main()
