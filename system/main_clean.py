# system/main.py
# Main entry point for PS_Opt_V2 optimization system
# Coordinates LHS sampling, simulation runs, and Bayesian Optimization

import sys
import os
from datetime import datetime
import config
import LHS
import run_simulation
import BO
import pandas as pd
import numpy as np


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
    start_time = datetime.now()
    config.RUN_TIMESTAMP = start_time.strftime("%Y%m%d_%H%M%S")

    # --- Steps 1 & 2: Get initial results ---
    if config.SKIP_INITIAL_SIMS:
        _skip_initial(config.RESULTS_CSV_FILE)
    else:
        _run_initial(config.PARAMS_CSV_FILE, config.RESULTS_CSV_FILE)

    if not os.path.exists(config.RESULTS_CSV_FILE):
        print("[ERROR] No results file. Cannot start BO.")
        return

    # --- Step 3: Bayesian Optimization Loop ---
    _run_bo_loop(config.RESULTS_CSV_FILE)

    # --- Step 4: Final Results ---
    _print_final_results(config.RESULTS_CSV_FILE)

    print(f"\nCompleted in {datetime.now() - start_time}")


# ============================================================================
# Stage functions
# ============================================================================

def _skip_initial(results_path):
    """Load existing result.csv instead of running LHS + simulations."""
    if not os.path.exists(results_path):
        print(f"[ERROR] result.csv not found at {results_path}")
        print("Run prepare_initial_data.py first, or set SKIP_INITIAL_SIMS=False.")
        return

    results_df = pd.read_csv(results_path) # Should we skip header?
    valid_count = results_df['v_pi_V'].notna().sum() # Count how many rows have a valid v_pi_V value (not NaN)
    print(f"[INFO] Using existing result.csv: {len(results_df)} rows ({valid_count} valid)")


def _run_initial(params_path, results_path):
    """Run LHS sampling + initial simulations."""
    # --- Step 1: LHS ---
    if config.SKIP_LHS:
        print(f"SKIP_LHS=True. Using existing: {params_path}")
        if not os.path.exists(params_path):
            print("[ERROR] params.csv not found. Set SKIP_LHS=False to generate new samples.")
            return
    else:
        print(f"Generating {config.LHS_N_SAMPLES} LHS samples...")
        LHS.generate_lhs_samples()
        print(f"Saved to: {params_path}")

    # --- Step 2: Initial Simulations ---
    print(f"\nRunning initial simulations from {params_path}...")
    run_simulation.run_init_file(params_csv_path=params_path)

    # Summary
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        valid_count = results_df['v_pi_V'].notna().sum()
        print(f"Summary: {valid_count}/{len(results_df)} valid (reached pi)")


def _run_bo_loop(results_path):
    """Train BO once, then run suggest -> simulate -> register loop."""
    bo_sim_id = len(pd.read_csv(results_path)) + 1

    # Train optimizer ONCE with all existing results
    print("\nTraining optimizer...")
    optimizer = BO.train_optimizer(result_csv_path=results_path)

    for iteration in range(1, config.MAX_ITERATIONS + 1):
        print(f"\n--- BO Iteration {iteration}/{config.MAX_ITERATIONS} (sim_id={bo_sim_id}) ---")

        try:
            # Suggest next point
            next_params = BO.get_next_sample(optimizer)
            if next_params is None:
                print("[WARNING] Optimizer returned None. Stopping.")
                break

            _print_params(next_params)

            # Run simulation
            next_params['sim_id'] = bo_sim_id
            is_last = (iteration == config.MAX_ITERATIONS)
            result = run_simulation.run_row(next_params, sim_id=bo_sim_id, is_last=is_last)

            if result is None:
                print("  -> FAILED (simulation returned None)")
                bo_sim_id += 1
                continue

            # Register with optimizer
            BO.register_result(optimizer, result, result['cost'])
            _print_result(result)
            bo_sim_id += 1

            # Show best so far
            best = BO.get_best_result(results_path)
            if best:
                print(f"  Best so far: sim_id={int(best['sim_id'])}, "
                      f"VpiL={best['v_pi_l_Vmm']:.3f} V*mm, "
                      f"Loss={best['loss_at_v_pi_dB_per_cm']:.2f} dB/cm")

        except KeyboardInterrupt:
            print("\n[INFO] Optimization interrupted by user.")
            break
        except Exception as e:
            print(f"  [ERROR] BO iteration {iteration}: {e}")
            bo_sim_id += 1
            continue


# ============================================================================
# Print helpers
# ============================================================================

def _print_params(params):
    """Print suggested parameters in human-readable format."""
    print(f"  Suggested: w_r={params['w_r']*1e9:.1f}nm, "
          f"h_si={params['h_si']*1e9:.1f}nm, "
          f"S={params['S']*1e9:.1f}nm, "
          f"doping={params['doping']:.2e}, "
          f"lambda={params['lambda']*1e9:.0f}nm, "
          f"L={params['length']*1e3:.2f}mm")


def _print_result(result):
    """Print a single simulation result."""
    v_pi = result.get('v_pi_V', float('nan'))
    if not np.isnan(v_pi):
        print(f"  Result: V_pi={v_pi:.2f}V, "
              f"VpiL={result['v_pi_l_Vmm']:.3f} V*mm, "
              f"Loss={result['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
              f"Cost={result['cost']:.2f}")
    else:
        print(f"  Result: V_pi not reached (max_dphi={result['max_dphi_rad']:.2f} rad), "
              f"Cost={result['cost']:.2f}")


def _print_final_results(results_path):
    """Print final optimization results summary."""
    if not os.path.exists(results_path):
        return

    results_df = pd.read_csv(results_path)
    valid_count = results_df['v_pi_V'].notna().sum()

    print(f"\n{'='*50}")
    print(f"FINAL: {len(results_df)} sims, {valid_count} valid")

    best = BO.get_best_result(results_path)
    if not best:
        print("No valid results found.")
        return

    print(f"\nBest (sim_id={int(best['sim_id'])}):")
    print(f"  V_pi    = {best['v_pi_V']:.2f} V")
    print(f"  V_pi*L  = {best['v_pi_l_Vmm']:.3f} V*mm")
    print(f"  Loss    = {best['loss_at_v_pi_dB_per_cm']:.2f} dB/cm")
    print(f"  C       = {best['C_at_v_pi_pF_per_cm']:.1f} pF/cm")
    print(f"\n  w_r={best['w_r']*1e9:.1f}nm, h_si={best['h_si']*1e9:.1f}nm, "
          f"S={best['S']*1e9:.1f}nm, doping={best['doping']:.2e}, "
          f"lambda={best['lambda']*1e9:.0f}nm, L={best['length']*1e3:.2f}mm")
    print("=" * 50)


if __name__ == "__main__":
    main()
