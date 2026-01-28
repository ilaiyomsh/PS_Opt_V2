# system/data_processor.py
# Data processing module for extracting and processing simulation results
# Processes Lumerical simulation files and extracts C, alpha, V_π*L

import config
import numpy as np
import pandas as pd
import os
import traceback
import json
from datetime import datetime

# Import matplotlib only if needed (lazy import for performance)
_plt = None

def _get_plt():
    """Lazy import of matplotlib to avoid overhead when not plotting."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def should_plot():
    """Determines if plots should be displayed based on config settings."""
    return config.SHOW_PLOTS or config.DEBUG


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


def extract_capacitance(charge_session, sim_id=None):
    """
    Extracts total capacitance data from CHARGE simulation.
    
    Args:
        charge_session: Lumerical CHARGE session object
        sim_id (int, optional): Simulation ID for plot saving
    
    Returns:
        tuple: (V, C_total_pF_cm) - Voltage array, total capacitance (pF/cm)
               C_total = Cn + Cp (electron + hole capacitance)
    """
    if not debug_prompt("Ready to extract capacitance data from CHARGE"):
        return None, None
    
    print("\n--- Extracting Capacitance Data ---")
    
    try:
        # Get charge data from monitor
        charge_data = charge_session.getresult("CHARGE::monitor_charge", "total_charge")
        
        # Extract voltage and charge arrays
        V = charge_data['V_drain'].flatten()
        Qn = config.ELEMENTARY_CHARGE * charge_data['n'].flatten()
        Qp = config.ELEMENTARY_CHARGE * charge_data['p'].flatten()
        
        # Calculate differential capacitance: C = dQ/dV
        Cn = np.gradient(Qn, V)
        Cp = np.gradient(Qp, V)
        
        # Total capacitance = Cn + Cp
        C_total = Cn + Cp
        
        # Convert to pF/cm (multiply by 1e10)
        C_total_pF_cm = C_total * 1e10
        
        print(f"  -> Extracted {len(V)} voltage points")
        print(f"  -> C_total range: [{np.min(C_total_pF_cm):.2f}, {np.max(C_total_pF_cm):.2f}] pF/cm")
        
        # Plot if enabled
        if should_plot():
            plot_capacitance(V, C_total_pF_cm)
        
        return V, C_total_pF_cm
        
    except Exception as e:
        print(f"  [ERROR] Failed to extract capacitance: {e}")
        raise e


def extract_optical_parameters(fde_session, length, wavelength, sim_id=None):
    """
    Extracts optical parameters from FDE simulation sweep results.

    Args:
        fde_session: Lumerical FDE/MODE session object
        length (float): Device length in meters (for phase shift calculation)
        wavelength (float): Operating wavelength in meters
        sim_id (int, optional): Simulation ID for plot saving

    Returns:
        tuple: (d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi)
               - d_neff: Change in effective index (array)
               - alpha_dB_per_cm: Optical loss in dB/cm (array)
               - d_phi: Phase shift in radians (array)
               - v_pi: Voltage for π phase shift (float or np.nan)
               - max_dphi: Maximum absolute phase shift in radians (float)
    """
    if not debug_prompt("Ready to extract optical parameters from FDE"):
        return None, None, None, np.nan, 0.0

    print("\n--- Extracting Optical Parameters ---")
    print(f"  Using device length: {length*1e3:.4f} mm, wavelength: {wavelength*1e9:.1f} nm")

    try:
        # Get sweep results
        sweep_result = fde_session.getsweepresult("voltage", "neff")
        neff = np.squeeze(sweep_result['neff'])

        # Reconstruct voltage array
        num_points = len(neff)
        V = np.linspace(0, config.V_MAX, num_points)

        # Calculate optical parameters using actual device parameters
        d_neff = calc_dneff(neff)
        alpha_dB_per_cm = calc_alpha(neff, wavelength)
        d_phi = calc_dphi(d_neff, length, wavelength)
        
        # Calculate maximum phase shift (for penalty calculation)
        abs_dphi = np.abs(d_phi)
        max_dphi = np.max(abs_dphi) if len(abs_dphi) > 0 else 0.0
        
        v_pi = calculate_v_pi(V, abs_dphi)
        
        print(f"  -> Extracted {num_points} voltage points")
        print(f"  -> d_neff range: [{np.min(d_neff):.2e}, {np.max(d_neff):.2e}]")
        print(f"  -> Loss range: [{np.min(alpha_dB_per_cm):.2f}, {np.max(alpha_dB_per_cm):.2f}] dB/cm")
        if not np.isnan(v_pi):
            print(f"  -> V_pi = {v_pi:.4f} V")
        else:
            print(f"  -> V_pi = NaN (phase shift did not reach π)")
        
        # Plot if enabled
        if should_plot():
            plot_optical_results(V, d_neff, alpha_dB_per_cm, d_phi, v_pi)
        
        return d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi
        
    except Exception as e:
        print(f"  [ERROR] Failed to extract optical parameters: {e}")
        raise e


def calc_alpha(neff, wavelength):
    """
    Calculates the optical loss (alpha) based on the effective index.

    Formula: alpha = 2 * k0 * Im(neff) * (10/ln(10)) * 1e-2
             where k0 = 2π/λ

    Args:
        neff (np.array): Complex effective index array
        wavelength (float): Operating wavelength in meters

    Returns:
        np.array: Optical loss in dB/cm
    """
    # Calculate k0 using actual wavelength
    k0 = 2 * np.pi / wavelength
    # Convert from Np/m to dB/cm
    # Factor: 10/ln(10) converts Np to dB, 1e-2 converts /m to /cm
    alpha = 2 * k0 * np.imag(neff) * (10 / np.log(10)) * 1e-2
    return alpha


def calc_dneff(neff):
    """
    Calculates the change in effective index relative to V=0.
    
    Args:
        neff (np.array): Complex effective index array
    
    Returns:
        np.array: Real part of effective index change
    """
    d_neff = np.real(neff - neff[0])
    return d_neff


def calc_dphi(d_neff, length, wavelength):
    """
    Calculates the phase shift based on effective index change.

    Formula: delta_phi = (2π * d_neff * L) / λ

    Args:
        d_neff (np.array): Change in effective index
        length (float): Device length in meters
        wavelength (float): Operating wavelength in meters

    Returns:
        np.array: Phase shift in radians
    """
    delta_phi = (2 * np.pi * d_neff * length) / wavelength
    return delta_phi


def calculate_v_pi(voltages, abs_dphi):
    """
    Calculates Vπ by interpolating the voltage required to achieve a π-radian phase shift.
    
    Args:
        voltages (np.array): The array of voltage points from the simulation
        abs_dphi (np.array): The array of corresponding absolute phase shifts in radians
    
    Returns:
        float: The calculated Vπ value, or np.nan if π is not reached
    """
    # Check if the phase shift ever reaches or exceeds π
    if np.max(abs_dphi) < np.pi:
        return np.nan
    
    # Use numpy's interpolation to find the voltage at which phi = π
    v_pi = np.interp(np.pi, abs_dphi, voltages)
    return v_pi


def plot_capacitance(V, C_total_pF_cm):
    """
    Plots total capacitance vs voltage.
    
    Args:
        V (np.array): Voltage array
        C_total_pF_cm (np.array): Total capacitance in pF/cm
    """
    plt = _get_plt()
    
    plt.figure("Capacitance vs Voltage")
    plt.plot(V, C_total_pF_cm, 'b-o', markersize=4, label='Total Capacitance')
    plt.title("Diode Capacitance vs. Voltage")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Capacitance (pF/cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    print("  -> Capacitance plot displayed")


def plot_optical_results(V, d_neff, alpha_dB_per_cm, d_phi, v_pi=None):
    """
    Plots optical simulation results: d_neff, loss, and phase shift vs voltage.
    
    Args:
        V (np.array): Voltage array
        d_neff (np.array): Change in effective index
        alpha_dB_per_cm (np.array): Optical loss in dB/cm
        d_phi (np.array): Phase shift in radians
        v_pi (float, optional): V_pi value to mark on plot
    """
    plt = _get_plt()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Delta neff vs. Voltage
    axes[0].plot(V, d_neff, 'b-o', markersize=4)
    axes[0].set_title("Effective Index Change vs. Voltage")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Δn_eff (Real Part)")
    axes[0].grid(True)
    
    # Plot 2: Loss vs. Voltage
    axes[1].plot(V, alpha_dB_per_cm, 'r-s', markersize=4)
    axes[1].set_title("Optical Loss vs. Voltage")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Loss (dB/cm)")
    axes[1].grid(True)
    
    # Plot 3: Phase Shift vs. Voltage
    axes[2].plot(V, d_phi, 'g-^', markersize=4)
    axes[2].axhline(y=np.pi, color='k', linestyle='--', alpha=0.5, label='π')
    axes[2].axhline(y=-np.pi, color='k', linestyle='--', alpha=0.5)
    if v_pi is not None and not np.isnan(v_pi):
        axes[2].axvline(x=v_pi, color='m', linestyle=':', alpha=0.7, label=f'V_π = {v_pi:.3f}V')
        axes[2].legend()
    axes[2].set_title("Phase Shift vs. Voltage")
    axes[2].set_xlabel("Voltage (V)")
    axes[2].set_ylabel("Phase Shift (radians)")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show(block=False)
    print("  -> Optical results plots displayed")


def _save_to_csv(filename, result_df, columns=None):
    """
    Internal helper to save a DataFrame to CSV file.

    Args:
        filename (str): Path to the CSV file
        result_df (pd.DataFrame): DataFrame with result data
        columns (list, optional): List of columns to include. If None, includes all.
    """
    # Filter columns if specified
    if columns is not None:
        # Only keep columns that exist in the result
        cols_to_use = [c for c in columns if c in result_df.columns]
        df_to_save = result_df[cols_to_use].copy()
    else:
        df_to_save = result_df.copy()

    file_exists = os.path.exists(filename)

    if not file_exists:
        # Create new file with headers
        df_to_save.to_csv(filename, index=False, mode='w', float_format='%.6e')
    else:
        # Read existing header to ensure column consistency
        try:
            existing_header = pd.read_csv(filename, nrows=0).columns.tolist()
            for col in existing_header:
                if col not in df_to_save.columns:
                    df_to_save[col] = np.nan
            df_to_save = df_to_save[existing_header]
        except Exception as e:
            print(f"  [WARNING] Could not read existing CSV header for {filename}: {e}")

        df_to_save.to_csv(filename, index=False, mode='a', header=False, float_format='%.6e')


def save_single_result_to_csv(filename, current_result):
    """
    Saves a single result to both minimal and full CSV files.

    Args:
        filename (str): Path to the minimal CSV file (result.csv)
        current_result (dict): Dictionary containing the result data to save

    Saves to:
        - filename (minimal): Only essential columns defined in config.MINIMAL_RESULT_COLUMNS
        - config.RESULTS_FULL_CSV_FILE (full): All columns
    """
    sim_id = current_result.get('sim_id', 'N/A')
    result_df = pd.DataFrame([current_result])

    # Reorder columns: sim_id, input params, output metrics
    try:
        input_params = list(config.SWEEP_PARAMETERS.keys())
        output_metrics = [col for col in result_df.columns
                       if col not in input_params and col != 'sim_id']
        column_order = ['sim_id'] + input_params + output_metrics
        if all(col in result_df.columns for col in column_order):
            result_df = result_df[column_order]
    except (AttributeError, KeyError):
        pass

    # Save to minimal file (essential columns only)
    minimal_cols = getattr(config, 'MINIMAL_RESULT_COLUMNS', None)
    _save_to_csv(filename, result_df, columns=minimal_cols)

    # Save to full file (all columns)
    full_filename = getattr(config, 'RESULTS_FULL_CSV_FILE', None)
    if full_filename:
        _save_to_csv(full_filename, result_df, columns=None)
        print(f"  -> Result for sim_id {sim_id} saved to {os.path.basename(filename)} and {os.path.basename(full_filename)}")
    else:
        print(f"  -> Result for sim_id {sim_id} saved to {filename}")


def save_error_to_csv(sim_id, stage, error, params=None):
    """
    Saves error details to errors.csv file.
    
    Args:
        sim_id (int): Simulation ID
        stage (str): Stage where error occurred (e.g., 'CHARGE_SETUP', 'CHARGE_RUN', etc.)
        error (Exception): Exception object that was raised
        params (dict, optional): Dictionary of input parameters
    
    Returns:
        None
    
    Side effects:
        Creates or appends to errors.csv with error details including:
        - sim_id, stage, error_type, error_message, traceback, timestamp, params
    """
    try:
        # Prepare error data
        error_type = type(error).__name__
        error_message = str(error)
        
        # Get full traceback
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        full_traceback = ''.join(tb_lines)
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert params to JSON string if provided
        params_json = json.dumps(params) if params else ''
        
        # Create error record
        error_record = {
            'sim_id': sim_id,
            'stage': stage,
            'error_type': error_type,
            'error_message': error_message,
            'traceback': full_traceback,
            'timestamp': timestamp,
            'params': params_json
        }
        
        # Convert to DataFrame
        error_df = pd.DataFrame([error_record])
        
        # Check if file exists
        file_exists = os.path.exists(config.ERRORS_CSV_FILE)
        
        if not file_exists:
            # Create new file with headers
            error_df.to_csv(config.ERRORS_CSV_FILE, index=False, mode='w')
            print(f"  -> Created errors CSV file: {config.ERRORS_CSV_FILE}")
        else:
            # Append to existing file without headers
            error_df.to_csv(config.ERRORS_CSV_FILE, index=False, mode='a', header=False)
        
        print(f"  -> Error for sim_id {sim_id} (stage: {stage}) saved to {config.ERRORS_CSV_FILE}")
        
    except Exception as save_error:
        # If saving error fails, at least print it
        print(f"  [CRITICAL] Failed to save error to CSV: {save_error}")
        print(f"  [CRITICAL] Original error: {error_type}: {error_message}")
