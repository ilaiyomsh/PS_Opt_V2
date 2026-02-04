# system/data_processor.py
# Data processing module — pure math and calculations
# No lumapi calls, no file I/O

import config
import numpy as np

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


def process_charge_data(V_drain, n, p, sim_id=None):
    """
    Processes raw charge data into capacitance values.

    Pure math — takes raw arrays from sim_handler, returns processed results.

    Args:
        V_drain (np.array): Voltage array from CHARGE monitor
        n (np.array): Electron count array
        p (np.array): Hole count array
        sim_id (int, optional): Simulation ID for plot saving

    Returns:
        tuple: (V, C_total_pF_cm) - Voltage array, total capacitance (pF/cm)
               C_total = Cn + Cp (electron + hole capacitance)
    """
    print("\n--- Processing Charge Data ---")

    # Calculate charge from carrier counts
    Qn = config.ELEMENTARY_CHARGE * n
    Qp = config.ELEMENTARY_CHARGE * p

    # Calculate differential capacitance: C = dQ/dV
    Cn = np.gradient(Qn, V_drain)
    Cp = np.gradient(Qp, V_drain)

    # Total capacitance = Cn + Cp
    C_total = Cn + Cp

    # Convert to pF/cm (multiply by 1e10)
    C_total_pF_cm = C_total * 1e10

    print(f"  -> Processed {len(V_drain)} voltage points")
    print(f"  -> C_total range: [{np.min(C_total_pF_cm):.2f}, {np.max(C_total_pF_cm):.2f}] pF/cm")

    # Plot if enabled
    if should_plot():
        plot_capacitance(V_drain, C_total_pF_cm)

    return V_drain, C_total_pF_cm


def process_optical_data(neff, length, wavelength, sim_id=None):
    """
    Processes raw neff data into optical parameters.

    Pure math — takes raw neff from sim_handler, returns processed results.

    Args:
        neff (np.array): Complex effective index array from FDE sweep
        length (float): Device length in meters
        wavelength (float): Operating wavelength in meters
        sim_id (int, optional): Simulation ID for plot saving

    Returns:
        tuple: (d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi)
               - d_neff: Change in effective index (array)
               - alpha_dB_per_cm: Optical loss in dB/cm (array)
               - d_phi: Phase shift in radians (array)
               - v_pi: Voltage for pi phase shift (float or np.nan)
               - max_dphi: Maximum absolute phase shift in radians (float)
    """
    print("\n--- Processing Optical Data ---")
    print(f"  Using device length: {length*1e3:.4f} mm, wavelength: {wavelength*1e9:.1f} nm")

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

    print(f"  -> Processed {num_points} voltage points")
    print(f"  -> d_neff range: [{np.min(d_neff):.2e}, {np.max(d_neff):.2e}]")
    print(f"  -> Loss range: [{np.min(alpha_dB_per_cm):.2f}, {np.max(alpha_dB_per_cm):.2f}] dB/cm")
    if not np.isnan(v_pi):
        print(f"  -> V_pi = {v_pi:.4f} V")
    else:
        print(f"  -> V_pi = NaN (phase shift did not reach pi)")

    # Plot if enabled
    if should_plot():
        plot_optical_results(V, d_neff, alpha_dB_per_cm, d_phi, v_pi)

    return d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi


def calc_alpha(neff, wavelength):
    """
    Calculates the optical loss (alpha) based on the effective index.

    Formula: alpha = 2 * k0 * Im(neff) * (10/ln(10)) * 1e-2
             where k0 = 2pi/lambda

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

    Formula: delta_phi = (2pi * d_neff * L) / lambda

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
    Calculates V_pi by interpolating the voltage required to achieve a pi-radian phase shift.

    Args:
        voltages (np.array): The array of voltage points from the simulation
        abs_dphi (np.array): The array of corresponding absolute phase shifts in radians

    Returns:
        float: The calculated V_pi value, or np.nan if pi is not reached
    """
    # Check if the phase shift ever reaches or exceeds pi
    if np.max(abs_dphi) < np.pi:
        return np.nan

    # Use numpy's interpolation to find the voltage at which phi = pi
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
    axes[0].set_ylabel("Dn_eff (Real Part)")
    axes[0].grid(True)

    # Plot 2: Loss vs. Voltage
    axes[1].plot(V, alpha_dB_per_cm, 'r-s', markersize=4)
    axes[1].set_title("Optical Loss vs. Voltage")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Loss (dB/cm)")
    axes[1].grid(True)

    # Plot 3: Phase Shift vs. Voltage
    axes[2].plot(V, d_phi, 'g-^', markersize=4)
    axes[2].axhline(y=np.pi, color='k', linestyle='--', alpha=0.5, label='pi')
    axes[2].axhline(y=-np.pi, color='k', linestyle='--', alpha=0.5)
    if v_pi is not None and not np.isnan(v_pi):
        axes[2].axvline(x=v_pi, color='m', linestyle=':', alpha=0.7, label=f'V_pi = {v_pi:.3f}V')
        axes[2].legend()
    axes[2].set_title("Phase Shift vs. Voltage")
    axes[2].set_xlabel("Voltage (V)")
    axes[2].set_ylabel("Phase Shift (radians)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show(block=False)
    print("  -> Optical results plots displayed")
