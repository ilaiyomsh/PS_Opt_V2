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
import results_archive
import pandas as pd
import numpy as np


def setup_logging():
    """
    Sets up dual logging to both console and file.

    Creates a timestamped log file in the logs/ directory.

    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_run_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)
    logger.addHandler(console_handler)

    return log_path


def log(message):
    """
    Logs a message with timestamp to both console and file.

    Args:
        message (str): Message to log
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    logging.info(f"[{timestamp}] {message}")


def log_raw(message):
    """
    Logs a message without timestamp to both console and file.

    Args:
        message (str): Message to log
    """
    logging.info(message)


def debug_prompt(message):
    """
    Prompts user for confirmation in DEBUG mode.
    
    Args:
        message (str): Message to display before prompt
    
    Returns:
        bool: True to continue, False to skip/abort
    """
    if not config.DEBUG:
        return True
    
    print(f"\n[DEBUG] {message}")
    response = input("[DEBUG] Press Enter to continue, 's' to skip, 'q' to quit: ").strip().lower()
    
    if response == 'q':
        print("[DEBUG] User requested quit.")
        sys.exit(0)
    elif response == 's':
        print("[DEBUG] User requested skip.")
        return False
    
    return True


def main():
    """
    Main function that orchestrates the entire optimization workflow.

    Workflow:
        1. Generate LHS samples (initial parameter sets)
        2. Run initial simulations for all LHS samples
        3. Loop:
           a. Train Bayesian Optimizer with current results
           b. Get next parameter set from optimizer
           c. Run simulation for new parameter set
           d. Check stop condition
        4. Output final results

    Stop Conditions:
        - Maximum number of iterations reached
        - Convergence criteria met
        - User interruption

    Returns:
        None

    Side Effects:
        - Creates params.csv with initial LHS samples
        - Creates/updates result.csv with simulation results
        - Prints optimization progress and final results
    """

    # Setup logging
    log_path = setup_logging()
    start_time = datetime.now()

    log_raw("=" * 80)
    log_raw(f"PS_Opt_V2 TEST RUN - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_raw("=" * 80)
    log_raw("Configuration:")
    log_raw(f"  - LHS Samples: {config.LHS_N_SAMPLES}")
    log_raw(f"  - Max BO Iterations: {config.MAX_ITERATIONS}")
    log_raw(f"  - Lumerical API: {config.LUMERICAL_API_PATH}")
    log_raw(f"  - SKIP_LHS: {config.SKIP_LHS}")
    log_raw(f"  - HIDE_GUI: {config.HIDE_GUI}")
    log_raw(f"  - DEBUG: {config.DEBUG}")
    log_raw(f"  - Delay Between Runs: {config.DELAY_BETWEEN_RUNS} seconds")
    log_raw(f"  - Log file: {log_path}")
    log_raw("")

    print("=" * 60)
    print("PS_Opt_V2 - PIN Diode Phase Shifter Optimization")
    print(f"DEBUG Mode: {'ON' if config.DEBUG else 'OFF'}")
    print("=" * 60)
    
    # --- Step 1: Generate LHS Sampling Plan ---
    log_raw("-" * 80)
    log("STAGE 1: LHS SAMPLING")
    log_raw("-" * 80)
    print("\n=== Step 1: Generating LHS Sampling Plan ===")

    if config.SKIP_LHS:
        log_raw(f"  SKIP_LHS=True. Using existing params file: {config.PARAMS_CSV_FILE}")
        print(f"[INFO] SKIP_LHS=True. Using existing params file: {config.PARAMS_CSV_FILE}")
        if not os.path.exists(config.PARAMS_CSV_FILE):
            log_raw(f"  [ERROR] params.csv not found at {config.PARAMS_CSV_FILE}")
            print(f"[ERROR] params.csv not found at {config.PARAMS_CSV_FILE}")
            print("[ERROR] Set SKIP_LHS=False to generate new samples.")
            return
    else:
        if not debug_prompt("Ready to generate LHS samples"):
            print("[INFO] LHS generation skipped by user.")
            return

        try:
            log_raw(f"  Generating {config.LHS_N_SAMPLES} samples using '{config.LHS_SAMPLING_METHOD}' method...")
            LHS.generate_lhs_samples()
            log_raw(f"  Saved to: {config.PARAMS_CSV_FILE}")
            print(f"LHS samples generated and saved to {config.PARAMS_CSV_FILE}")
        except Exception as e:
            log_raw(f"  [ERROR] Failed to generate LHS samples: {e}")
            print(f"[ERROR] Failed to generate LHS samples: {e}")
            return
    log_raw("")
    
    # --- Step 2: Run Initial Simulations ---
    log_raw("-" * 80)
    log("STAGE 2: INITIAL SIMULATIONS (LHS)")
    log_raw("-" * 80)
    print("\n=== Step 2: Running Initial Simulations ===")

    if not debug_prompt("Ready to run initial simulations"):
        print("[INFO] Initial simulations skipped by user.")
        return
    try:
        run_simulation.run_init_file(params_csv_path=config.PARAMS_CSV_FILE, logger_func=log_raw)
        log_raw(f"  Initial simulations completed. Results saved to {config.RESULTS_CSV_FILE}")
        print(f"Initial simulations completed. Results saved to {config.RESULTS_CSV_FILE}")

        # Log summary of initial simulations
        if os.path.exists(config.RESULTS_CSV_FILE):
            results_df = pd.read_csv(config.RESULTS_CSV_FILE)
            valid_count = results_df['v_pi_V'].notna().sum()
            invalid_count = len(results_df) - valid_count
            log_raw(f"\n  Summary: {valid_count}/{len(results_df)} valid (reached pi), {invalid_count}/{len(results_df)} penalized")
            if valid_count > 0:
                # Find best initial result
                valid_results = results_df[results_df['v_pi_V'].notna()]
                if 'cost' in valid_results.columns:
                    best_idx = valid_results['cost'].idxmin()
                    best_sim_id = valid_results.loc[best_idx, 'sim_id']
                    best_cost = valid_results.loc[best_idx, 'cost']
                    log_raw(f"  Best initial: sim_id={int(best_sim_id)}, Cost={best_cost:.3f}")
    except Exception as e:
        log_raw(f"  [ERROR] Failed to run initial simulations: {e}")
        print(f"[ERROR] Failed to run initial simulations: {e}")
        # Check if partial results exist
        if os.path.exists(config.RESULTS_CSV_FILE):
            try:
                results_df = pd.read_csv(config.RESULTS_CSV_FILE)
                if len(results_df) > 0:
                    log_raw(f"  [WARNING] Found {len(results_df)} partial results. Continuing with existing data...")
                    print(f"[WARNING] Found {len(results_df)} partial results. Continuing with existing data...")
                    # Continue to BO loop with partial results
                else:
                    log_raw("  [ERROR] Results file exists but is empty. Cannot continue.")
                    print("[ERROR] Results file exists but is empty. Cannot continue.")
                    return
            except Exception as read_error:
                log_raw(f"  [ERROR] Failed to read existing results file: {read_error}")
                print(f"[ERROR] Failed to read existing results file: {read_error}")
                return
        else:
            log_raw("  [ERROR] No results file found. Cannot continue without initial simulations.")
            print("[ERROR] No results file found. Cannot continue without initial simulations.")
            return
    log_raw("")
    
    # --- Step 3: Bayesian Optimization Loop ---
    log_raw("-" * 80)
    log("STAGE 3: BAYESIAN OPTIMIZATION")
    log_raw("-" * 80)
    print("\n=== Step 3: Starting Bayesian Optimization Loop ===")

    if not debug_prompt("Ready to start Bayesian Optimization loop"):
        print("[INFO] BO loop skipped by user.")
        return

    # Determine starting sim_id for BO iterations
    # Continue from the last sim_id in results (safe even if rows were deleted)
    bo_sim_id = 1  # Default if no results exist
    if os.path.exists(config.RESULTS_CSV_FILE):
        try:
            existing_results = pd.read_csv(config.RESULTS_CSV_FILE)
            if 'sim_id' in existing_results.columns and len(existing_results) > 0:
                bo_sim_id = int(existing_results['sim_id'].max()) + 1
                log_raw(f"  Training GP on {len(existing_results)} data points...")
        except Exception:
            pass

    iteration = 0

    while iteration < config.MAX_ITERATIONS:
        iteration += 1
        log_raw(f"\n  --- Iteration {iteration}/{config.MAX_ITERATIONS} ---")
        print(f"\n--- BO Iteration {iteration}/{config.MAX_ITERATIONS} (sim_id={bo_sim_id}) ---")

        if not debug_prompt(f"Ready for BO iteration {iteration}"):
            print(f"[INFO] BO iteration {iteration} skipped by user.")
            continue

        # main loop for BO
        try:
            # 3a. Train Bayesian Optimizer with current results
            print("Training optimizer with existing results...")
            optimizer = BO.train_optimizer(result_csv_path=config.RESULTS_CSV_FILE)

            # 3b. Get next parameter set from optimizer
            print("Predicting next parameter set...")
            next_params = BO.get_next_sample(optimizer, result_csv_path=config.RESULTS_CSV_FILE)

            if next_params is None:
                log_raw("  [WARNING] Optimizer returned None. Stopping optimization.")
                print("[WARNING] Optimizer returned None. Stopping optimization.")
                break

            print(f"Next parameters to sample: {next_params}")

            # Log suggested parameters
            log_raw(f"  [{bo_sim_id}] sim_id={bo_sim_id}")
            log_raw(f"      Suggested: w_r={next_params.get('w_r', 0)*1e9:.1f}nm, "
                    f"h_si={next_params.get('h_si', 0)*1e9:.1f}nm, "
                    f"S={next_params.get('S', 0)*1e9:.1f}nm, "
                    f"doping={next_params.get('doping', 0):.2e}, "
                    f"lambda={next_params.get('lambda', 0)*1e9:.0f}nm, "
                    f"L={next_params.get('length', 0)*1e3:.2f}mm")

            # 3c. Run simulation for new parameter set
            # Add sim_id to the parameters dictionary
            next_params_with_id = next_params.copy()
            next_params_with_id['sim_id'] = bo_sim_id

            print(f"Running simulation for new parameter set (sim_id={bo_sim_id})...")
            result = run_simulation.run_row(next_params_with_id, sim_id=bo_sim_id, is_last=(iteration == config.MAX_ITERATIONS))

            if result is None:
                log_raw(f"      Output: FAILED (simulation returned None)")
                print("[WARNING] Simulation returned None. Skipping this iteration.")
                bo_sim_id += 1  # Still increment sim_id to track failed attempts
                continue

            # Log result
            v_pi = result.get('v_pi_V', float('nan'))
            if not np.isnan(v_pi):
                log_raw(f"      Output: V_pi={v_pi:.2f}V, "
                        f"V_pi*L={result.get('v_pi_l_Vmm', 0):.3f} V*mm, "
                        f"Loss={result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm, "
                        f"C={result.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm, "
                        f"max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
            else:
                log_raw(f"      Output: V_pi=NaN (did not reach pi), max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
                log_raw(f"      Cost: PENALTY")

            print(f"Simulation completed. Result saved with sim_id={bo_sim_id}")
            bo_sim_id += 1

            # 3d. Check convergence (optional)
            # Display current best result
            best_result = BO.get_best_result(config.RESULTS_CSV_FILE)
            if best_result:
                log_raw(f"      Best so far: sim_id={int(best_result.get('sim_id', 0))}, "
                        f"V_pi*L={best_result.get('v_pi_l_Vmm', 0):.3f} V*mm")
                print(f"  Current best: loss={best_result.get('loss_at_v_pi_dB_per_cm', 'N/A'):.4f} dB/cm, "
                      f"VpiL={best_result.get('v_pi_l_Vmm', 'N/A'):.4f} V*mm")

        except KeyboardInterrupt:
            log_raw("\n  [INFO] Optimization interrupted by user.")
            print("\n[INFO] Optimization interrupted by user.")
            break
        except Exception as e:
            log_raw(f"  [ERROR] Error in BO iteration {iteration}: {e}")
            print(f"[ERROR] Error in BO iteration {iteration}: {e}")
            print("Continuing to next iteration...")
            bo_sim_id += 1  # Increment sim_id even on error
            continue
    log_raw("")
    
    # --- Step 4: Output Final Results ---
    log_raw("-" * 80)
    log("STAGE 4: FINAL RESULTS")
    log_raw("-" * 80)
    print("\n=== Step 4: Final Results ===")

    try:
        if os.path.exists(config.RESULTS_CSV_FILE):
            results_df = pd.read_csv(config.RESULTS_CSV_FILE)
            valid_count = results_df['v_pi_V'].notna().sum()
            invalid_count = len(results_df) - valid_count

            log_raw(f"  Total simulations run: {len(results_df)}")
            log_raw(f"  Valid simulations:     {valid_count} (reached pi phase shift)")
            log_raw(f"  Invalid simulations:   {invalid_count} (penalized)")
            log_raw("")
            print(f"\nTotal simulations completed: {len(results_df)}")

            if len(results_df) > 0:
                # Get the best result using BO module
                best_result = BO.get_best_result(config.RESULTS_CSV_FILE)

                if best_result:
                    # Log best result in table format
                    log_raw("  BEST RESULT:")
                    log_raw("  +-----------+------------------+")
                    log_raw(f"  | sim_id    | {int(best_result.get('sim_id', 0)):<16} |")
                    log_raw("  +-----------+------------------+")
                    log_raw("  | INPUTS                       |")
                    log_raw("  +-----------+------------------+")
                    log_raw(f"  | w_r       | {best_result.get('w_r', 0)*1e9:.1f} nm".ljust(30) + "|")
                    log_raw(f"  | h_si      | {best_result.get('h_si', 0)*1e9:.1f} nm".ljust(30) + "|")
                    log_raw(f"  | S         | {best_result.get('S', 0)*1e9:.1f} nm".ljust(30) + "|")
                    log_raw(f"  | doping    | {best_result.get('doping', 0):.2e} cm^-3".ljust(30) + "|")
                    log_raw(f"  | lambda    | {best_result.get('lambda', 0)*1e9:.0f} nm".ljust(30) + "|")
                    log_raw(f"  | length    | {best_result.get('length', 0)*1e3:.2f} mm".ljust(30) + "|")
                    log_raw("  +-----------+------------------+")
                    log_raw("  | OUTPUTS                      |")
                    log_raw("  +-----------+------------------+")
                    log_raw(f"  | V_pi      | {best_result.get('v_pi_V', 0):.2f} V".ljust(30) + "|")
                    log_raw(f"  | V_pi*L    | {best_result.get('v_pi_l_Vmm', 0):.3f} V*mm".ljust(30) + "|")
                    log_raw(f"  | Loss      | {best_result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm".ljust(30) + "|")
                    log_raw(f"  | C         | {best_result.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm".ljust(30) + "|")
                    log_raw(f"  | max_dphi  | {best_result.get('max_dphi_rad', 0):.2f} rad".ljust(30) + "|")
                    log_raw("  +-----------+------------------+")

                    print("\n" + "=" * 50)
                    print("BEST RESULT FOUND:")
                    print("=" * 50)
                    print(f"  sim_id: {best_result.get('sim_id', 'N/A')}")
                    print(f"  V_pi: {best_result.get('v_pi_V', 'N/A'):.4f} V")
                    print(f"  V_pi*L: {best_result.get('v_pi_l_Vmm', 'N/A'):.4f} V*mm")
                    print(f"  Loss at V_pi: {best_result.get('loss_at_v_pi_dB_per_cm', 'N/A'):.4f} dB/cm")
                    print(f"  Capacitance at V_pi: {best_result.get('C_at_v_pi_pF_per_cm', 'N/A'):.4f} pF/cm")
                    print("\nBest Parameters:")
                    for param_name in config.SWEEP_PARAMETERS.keys():
                        if param_name in best_result:
                            unit = config.SWEEP_PARAMETERS[param_name].get('unit', '')
                            print(f"  {param_name}: {best_result[param_name]:.4e} {unit}")
                    print("=" * 50)
                else:
                    log_raw("\n  [WARNING] Could not determine best result (missing loss or VpiL values)")
                    print("\n[WARNING] Could not determine best result (missing loss or VpiL values)")
                    print("\nResults summary:")
                    print(results_df.describe())
        else:
            log_raw("  [WARNING] Results file not found.")
            print("[WARNING] Results file not found.")

    except Exception as e:
        log_raw(f"  [ERROR] Failed to read final results: {e}")
        print(f"[ERROR] Failed to read final results: {e}")

    # Auto-archive results if enabled
    if config.AUTO_ARCHIVE_RESULTS:
        print("\n--- Archiving Results ---")
        log_raw("\n--- Archiving Results ---")
        try:
            archive_path = results_archive.archive_current_results()
            if archive_path:
                log_raw(f"  Results archived to: {archive_path}")
        except Exception as e:
            log_raw(f"  [WARNING] Failed to archive results: {e}")
            print(f"  [WARNING] Failed to archive results: {e}")

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    log_raw("")
    log_raw("=" * 80)
    log_raw("TEST COMPLETED")
    log_raw(f"  Total duration: {duration}")
    log_raw(f"  Log saved to: {log_path}")
    log_raw("=" * 80)

    print("\n" + "=" * 60)
    print("Optimization completed!")
    print(f"Log saved to: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
