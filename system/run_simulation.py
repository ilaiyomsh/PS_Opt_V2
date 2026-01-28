# system/run_simulation.py
# Simulation runner module that coordinates sim_handler and data_processor
# Handles running initial simulations and individual simulation rows

import sys
import os
import time
import pandas as pd
import numpy as np

import config
import sim_handler
import data_processor
import results_archive

# Add Lumerical API path to system path
sys.path.append(config.LUMERICAL_API_PATH)

# Import Lumerical API (will fail if not installed, but allows code structure)
try:
    import lumapi
except ImportError:
    print("[WARNING] lumapi not found. Lumerical simulations will not work.")
    lumapi = None


def debug_prompt(message):
    """
    Prompts user for confirmation in DEBUG mode.

    Args:
        message (str): Message to display before prompt

    Returns:
        bool: True to continue, False to skip
    """
    if not config.DEBUG:
        return True

    print(f"\n[DEBUG] {message}")
    response = input("[DEBUG] Press Enter to continue, 's' to skip: ").strip().lower()

    if response == 's':
        print("[DEBUG] Step skipped by user.")
        return False

    return True


def cooling_delay(logger_func=None, skip=False):
    """
    Performs a cooling delay between simulation runs.

    Args:
        logger_func (callable, optional): Function to log messages
        skip (bool): If True, skip the delay (for last simulation)
    """
    if skip or config.DELAY_BETWEEN_RUNS <= 0:
        return

    delay_seconds = config.DELAY_BETWEEN_RUNS
    delay_minutes = delay_seconds // 60
    delay_remaining_seconds = delay_seconds % 60

    msg = f"  [Cooling delay: {delay_minutes}:{delay_remaining_seconds:02d} minutes...]"
    print(msg)
    if logger_func:
        logger_func(msg)

    # Display countdown every minute
    remaining = delay_seconds
    while remaining > 0:
        # Update every 60 seconds or remaining time
        sleep_time = min(60, remaining)
        time.sleep(sleep_time)
        remaining -= sleep_time
        if remaining > 0:
            mins = remaining // 60
            secs = remaining % 60
            print(f"    Cooling: {mins}:{secs:02d} remaining...")

    print("  [Cooling delay complete]")


def run_init_file(params_csv_path=None, logger_func=None):
    """
    Runs initial simulations for all parameter sets from params.csv.

    Args:
        params_csv_path (str, optional): Path to params.csv file.
                                        If None, uses config.PARAMS_CSV_FILE
        logger_func (callable, optional): Function to log messages to file

    Returns:
        str: Path to the generated results CSV file

    Output:
        Creates result.csv file with simulation results for all initial samples.
        Each row contains input parameters and output metrics (C, alpha, V_π*L).

    Process:
        1. Loads params.csv (skipping units row)
        2. For each row: calls run_row()
        3. Returns path to results file
    """
    # Use default path if not provided
    if params_csv_path is None:
        params_csv_path = config.PARAMS_CSV_FILE

    print(f"\n{'='*60}")
    print("Running Initial Simulations")
    print(f"{'='*60}")
    print(f"Parameters file: {params_csv_path}")

    # Load parameters CSV (skip units row at index 1)
    try:
        params_df = pd.read_csv(params_csv_path, skiprows=[1])
        print(f"Loaded {len(params_df)} parameter sets")
    except Exception as e:
        print(f"[ERROR] Failed to load params CSV: {e}")
        raise

    # Track results
    successful = 0
    failed = 0
    total_sims = len(params_df)

    # Run simulation for each row
    for idx, row in params_df.iterrows():
        sim_id = int(row['sim_id'])
        is_last = (idx == len(params_df) - 1)
        print(f"\n--- Simulation {sim_id}/{total_sims} ---")

        # Log simulation start
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

                # Log result
                if logger_func:
                    v_pi = result.get('v_pi_V', float('nan'))
                    # Log timing
                    charge_t = result.get('charge_time_s', 0)
                    fde_t = result.get('fde_time_s', 0)
                    total_t = result.get('total_time_s', 0)
                    logger_func(f"        Timing: CHARGE={charge_t:.1f}s, FDE={fde_t:.1f}s, Total={total_t:.1f}s")

                    # Log geometry analysis
                    intrinsic_w = result.get('intrinsic_width_m', 0)
                    logger_func(f"        Geometry: intrinsic_width={intrinsic_w*1e9:.1f}nm (w_r + 2*S)")

                    if not np.isnan(v_pi):
                        logger_func(f"        Output: V_pi={v_pi:.2f}V, "
                                   f"V_pi*L={result.get('v_pi_l_Vmm', 0):.3f} V*mm, "
                                   f"Loss={result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm, "
                                   f"C={result.get('C_at_v_pi_pF_per_cm', 0):.1f} pF/cm")
                        # Log cost metrics
                        logger_func(f"        Cost:   {result.get('cost', 0):.4f} "
                                   f"(norm_loss={result.get('norm_loss', 0):.2f}, "
                                   f"norm_vpil={result.get('norm_vpil', 0):.2f})")
                        # Log ranges
                        logger_func(f"        Ranges: Loss=[{result.get('loss_min_dB_per_cm', 0):.1f}, {result.get('loss_max_dB_per_cm', 0):.1f}] dB/cm, "
                                   f"C=[{result.get('C_min_pF_per_cm', 0):.1f}, {result.get('C_max_pF_per_cm', 0):.1f}] pF/cm")
                    else:
                        logger_func(f"        Output: V_pi=NaN (did not reach pi), max_dphi={result.get('max_dphi_rad', 0):.2f} rad")
                        logger_func(f"        Cost:   PENALTY (is_valid=False)")
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
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("Initial Simulations Summary")
    print(f"{'='*60}")
    print(f"Total: {len(params_df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {config.RESULTS_CSV_FILE}")
    
    # Error Summary
    if os.path.exists(config.ERRORS_CSV_FILE):
        try:
            errors_df = pd.read_csv(config.ERRORS_CSV_FILE)
            if len(errors_df) > 0:
                print(f"\n{'='*60}")
                print("Error Summary")
                print(f"{'='*60}")
                print(f"Total errors recorded: {len(errors_df)}")
                
                # Count errors by stage
                stage_counts = errors_df['stage'].value_counts()
                print("\nErrors by stage:")
                for stage, count in stage_counts.items():
                    print(f"  {stage}: {count}")
                
                # List failed sim_ids
                failed_sim_ids = sorted(errors_df['sim_id'].unique())
                print(f"\nFailed simulation IDs: {failed_sim_ids}")
                print(f"Errors saved to: {config.ERRORS_CSV_FILE}")
        except Exception as e:
            print(f"\n[WARNING] Could not read error summary: {e}")
    else:
        if failed > 0:
            print(f"\n[NOTE] No errors.csv file found, but {failed} simulations failed.")
    
    return config.RESULTS_CSV_FILE


def _is_duplicate_params(new_params, existing_df, tolerance=None):
    """
    Check if suggested parameters are too similar to existing results.

    Args:
        new_params (dict): Parameter values to check
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


def run_row(row, sim_id=None, is_last=False):
    """
    Runs a single simulation for a given parameter row and appends result to CSV.

    Args:
        row (dict or pd.Series): Dictionary or pandas Series containing parameter values
                                Keys should match parameter names in config.SWEEP_PARAMETERS
        sim_id (int, optional): Simulation ID. If None, extracted from row['sim_id']
        is_last (bool): If True, skip the cooling delay after this simulation

    Returns:
        dict: Dictionary containing the simulation results:
              - All input parameters from row
              - v_pi_V: V_pi voltage
              - v_pi_l_Vmm: V_pi*L product
              - loss_at_v_pi_dB_per_cm: Optical loss at V_pi
              - Cn_at_v_pi_pF_per_cm: N-type capacitance at V_pi
              - Cp_at_v_pi_pF_per_cm: P-type capacitance at V_pi

    Output:
        Appends a new row to result.csv with the simulation results.

    Process:
        1. Opens CHARGE session and runs electrical simulation
        2. Opens FDE session and runs optical simulation
        3. Extracts and calculates results
        4. Saves results to CSV
        5. Wait for cooling delay (if configured)
    """
    if lumapi is None:
        print("[ERROR] lumapi not available. Cannot run simulation.")
        return None
    
    # Convert row to dict if pandas Series
    if isinstance(row, pd.Series):
        params = row.to_dict()
    else:
        params = dict(row)
    
    # Get simulation ID
    if sim_id is None:
        sim_id = int(params.get('sim_id', 0))

    print(f"\n{'='*50}")
    print(f"Running simulation {sim_id}")
    print(f"{'='*50}")
    print(f"Parameters: {params}")

    # ========== Duplicate Detection ==========
    # Check if these parameters are too similar to already-tested samples
    existing_df = results_archive.load_all_results_for_bo()
    if _is_duplicate_params(params, existing_df):
        print(f"  [SKIP] Parameters are too similar to an existing result. Skipping simulation.")
        return None

    # Track timing
    sim_start_time = time.time()
    charge_time = 0.0
    fde_time = 0.0

    # Show GUI if: DEBUG=True OR HIDE_GUI=False
    hide_gui = config.HIDE_GUI and not config.DEBUG
   
   
    # Path to save charge data
    charge_data_path = config.CHARGE_DATA_FILE
    
    # Initialize result variables
    V_cap = None
    C_total_pF_cm = None
    d_neff = None
    alpha_dB_per_cm = None
    d_phi = None
    v_pi = None
    max_dphi = None
    
    # ========== A. CHARGE Simulation (Electrical) ==========
    print("\n--- A. CHARGE Simulation ---")
    
    if not debug_prompt("Ready to open CHARGE session"):
        print("[INFO] CHARGE simulation skipped.")
        return None
    
    # Stage 1: CHARGE Setup
    try:
        charge = lumapi.DEVICE(hide=hide_gui)
        charge.load(config.CHARGE_SIM_FILE)
        print("  CHARGE simulation file loaded.")
    except Exception as e:
        error_msg = f"Failed to open/load CHARGE session: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "CHARGE_SETUP", e, params)
        return None
    
    # Stage 2: Set CHARGE parameters
    try:
        if not debug_prompt("Ready to set CHARGE parameters"):
            charge.close()
            return None
        
        sim_handler.set_charge_parameters(charge, params, charge_data_path)
    except Exception as e:
        error_msg = f"Failed to set CHARGE parameters: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "CHARGE_SETUP", e, params)
        try:
            charge.close()
        except:
            pass
        return None
    
    # Stage 3: Run CHARGE simulation
    if config.RUN_SIMULATION:
        try:
            if not debug_prompt("Ready to run CHARGE simulation"):
                charge.close()
                return None

            charge_start = time.time()
            sim_handler.run_charge_simulation(charge)
            charge_time = time.time() - charge_start
            print("  CHARGE simulation completed.")
        except Exception as e:
            error_msg = f"CHARGE simulation run failed: {e}"
            print(f"  [ERROR] {error_msg}")
            data_processor.save_error_to_csv(sim_id, "CHARGE_RUN", e, params)
            try:
                charge.close()
            except:
                pass
            return None
    
    # Stage 4: Extract capacitance data
    try:
        if not debug_prompt("Ready to extract capacitance data"):
            charge.close()
            return None
        
        V_cap, C_total_pF_cm = data_processor.extract_capacitance(charge, sim_id=sim_id)
        print(f"  Capacitance extracted: {len(V_cap)} voltage points")
    except Exception as e:
        error_msg = f"Failed to extract capacitance: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "CHARGE_EXTRACT", e, params)
        try:
            charge.close()
        except:
            pass
        return None
    finally:
        # Always close CHARGE session
        try:
            charge.close()
        except:
            pass
    
    # ========== B. FDE Simulation (Optical) ==========
    print("\n--- B. FDE Simulation ---")
    
    if not debug_prompt("Ready to open FDE session"):
        print("[INFO] FDE simulation skipped.")
        return None
    
    # Stage 1: FDE Setup
    try:
        fde = lumapi.MODE(hide=hide_gui)
        fde.load(config.FDE_SIM_FILE)
        print("  FDE simulation file loaded.")
    except Exception as e:
        error_msg = f"Failed to open/load FDE session: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "FDE_SETUP", e, params)
        return None
    
    # Stage 2: Set FDE parameters
    try:
        if not debug_prompt("Ready to set FDE parameters"):
            fde.close()
            return None
        
        sim_handler.set_fde_parameters(fde, params)
    except Exception as e:
        error_msg = f"Failed to set FDE parameters: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "FDE_SETUP", e, params)
        try:
            fde.close()
        except:
            pass
        return None
    
    # Stage 3: Import CHARGE data
    try:
        if not debug_prompt("Ready to import CHARGE data into FDE"):
            fde.close()
            return None
        
        sim_handler.import_charge_data(fde, charge_data_path)
        print("  CHARGE data imported into FDE.")
    except Exception as e:
        error_msg = f"Failed to import CHARGE data: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "FDE_SETUP", e, params)
        try:
            fde.close()
        except:
            pass
        return None
    
    # Stage 4: Run FDE sweep
    if config.RUN_SIMULATION:
        try:
            if not debug_prompt("Ready to run FDE voltage sweep"):
                fde.close()
                return None

            fde_start = time.time()
            sim_handler.run_fde_sweep(fde)
            fde_time = time.time() - fde_start
            print("  FDE sweep completed.")
        except Exception as e:
            error_msg = f"FDE sweep run failed: {e}"
            print(f"  [ERROR] {error_msg}")
            data_processor.save_error_to_csv(sim_id, "FDE_RUN", e, params)
            try:
                fde.close()
            except:
                pass
            return None
    
    # Stage 5: Extract optical data
    try:
        if not debug_prompt("Ready to extract optical data"):
            fde.close()
            return None
        
        # Extract optical parameters using actual device length and wavelength
        device_length = float(params['length'])
        device_wavelength = float(params['lambda'])
        d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi = data_processor.extract_optical_parameters(
            fde, length=device_length, wavelength=device_wavelength, sim_id=sim_id
        )
        print(f"  Optical data extracted: V_pi = {v_pi:.4f} V" if not np.isnan(v_pi) else "  V_pi: NaN")
    except Exception as e:
        error_msg = f"Failed to extract optical parameters: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "FDE_EXTRACT", e, params)
        try:
            fde.close()
        except:
            pass
        return None
    finally:
        # Always close FDE session
        try:
            fde.close()
        except:
            pass
    
    # ========== C. Calculate Results ==========
    print("\n--- C. Calculating Results ---")

    # Calculate total simulation time
    total_sim_time = time.time() - sim_start_time

    try:
        # Reconstruct voltage vector for interpolation
        if d_neff is None or len(d_neff) == 0:
            raise ValueError("No optical data available for calculation")
        if V_cap is None or len(V_cap) == 0:
            raise ValueError("No capacitance data available for calculation")

        V_fde = np.linspace(0, config.V_MAX, len(d_neff))

        # Calculate derived geometry parameters
        w_r = float(params['w_r'])
        S = float(params['S'])
        intrinsic_width = w_r + 2 * S  # Total intrinsic region width

        # Calculate performance metrics at operating point (V_pi)
        if not np.isnan(v_pi):
            loss_at_v_pi = np.interp(v_pi, V_fde, alpha_dB_per_cm)
            C_at_v_pi = np.interp(v_pi, V_cap, C_total_pF_cm)
            v_pi_l = v_pi * float(params['length']) * 1e3  # V*mm
        else:
            loss_at_v_pi = np.nan
            C_at_v_pi = np.nan
            v_pi_l = np.nan

        # Additional analysis metrics
        loss_min = np.min(alpha_dB_per_cm) if alpha_dB_per_cm is not None else np.nan
        loss_max = np.max(alpha_dB_per_cm) if alpha_dB_per_cm is not None else np.nan
        C_min = np.min(C_total_pF_cm) if C_total_pF_cm is not None else np.nan
        C_max = np.max(C_total_pF_cm) if C_total_pF_cm is not None else np.nan

        # Cost function metrics (for optimization analysis)
        is_valid = not np.isnan(v_pi_l) if v_pi_l is not None else False

        if is_valid:
            # Calculate normalized metrics (Eq. 27 top case)
            norm_loss = (loss_at_v_pi / config.TARGETS['loss']) ** 2
            norm_vpil = (v_pi_l / config.TARGETS['vpil']) ** 2
            cost = config.FOM_WEIGHTS['loss'] * norm_loss + config.FOM_WEIGHTS['vpil'] * norm_vpil
        else:
            # Penalty case - set normalized metrics to NaN
            norm_loss = np.nan
            norm_vpil = np.nan
            # Use a large penalty value (actual C_BASE is dynamic, use placeholder)
            cost = 1e9  # Penalty indicator

        # Create result dictionary with timing and analysis data
        result = {
            'sim_id': sim_id,
            **params,  # Include all input parameters
            'v_pi_V': v_pi,
            'v_pi_l_Vmm': v_pi_l,
            'loss_at_v_pi_dB_per_cm': loss_at_v_pi,
            'C_at_v_pi_pF_per_cm': C_at_v_pi,
            'max_dphi_rad': max_dphi,
            # Cost function metrics
            'is_valid': is_valid,
            'norm_loss': norm_loss,
            'norm_vpil': norm_vpil,
            'cost': cost,
            # Timing data
            'charge_time_s': charge_time,
            'fde_time_s': fde_time,
            'total_time_s': total_sim_time,
            # Additional analysis data
            'intrinsic_width_m': intrinsic_width,
            'loss_min_dB_per_cm': loss_min,
            'loss_max_dB_per_cm': loss_max,
            'C_min_pF_per_cm': C_min,
            'C_max_pF_per_cm': C_max,
        }

        print(f"\n  Result: V_pi*L = {v_pi_l:.4f} V*mm, Loss = {loss_at_v_pi:.2f} dB/cm, C = {C_at_v_pi:.2f} pF/cm")
        print(f"  Cost: {cost:.4f} (valid={is_valid}, norm_loss={norm_loss:.2f}, norm_vpil={norm_vpil:.2f})" if is_valid else f"  Cost: PENALTY (valid=False)")
        print(f"  Timing: CHARGE={charge_time:.1f}s, FDE={fde_time:.1f}s, Total={total_sim_time:.1f}s")
        
    except Exception as e:
        error_msg = f"Failed to calculate results: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "RESULTS_CALC", e, params)
        return None
    
    # ========== D. Save Result ==========
    try:
        if not debug_prompt("Ready to save result to CSV"):
            return result  # Return result but don't save

        data_processor.save_single_result_to_csv(config.RESULTS_CSV_FILE, result)
        print(f"  Result saved to {config.RESULTS_CSV_FILE}")
    except Exception as e:
        error_msg = f"Failed to save result to CSV: {e}"
        print(f"  [ERROR] {error_msg}")
        data_processor.save_error_to_csv(sim_id, "CSV_SAVE", e, params)
        # Still return result even if save failed
        # Apply cooling delay before returning
        cooling_delay(skip=is_last)
        return result

    # ========== E. Cooling Delay ==========
    # Wait between simulations to prevent overheating (skip for last simulation)
    cooling_delay(skip=is_last)

    return result

