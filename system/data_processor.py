# system/data_processor.py
# Pure math for processing CHARGE and FDE outputs. No lumapi, no file I/O.

import config
import numpy as np

_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def should_plot():
    return config.SHOW_PLOTS or config.DEBUG


def process_charge_data(V_drain, n, p, sim_id=None):
    """Convert carrier counts to total capacitance in pF/cm. Returns (V, C_total_pF_cm)."""
    Qn = config.ELEMENTARY_CHARGE * n
    Qp = config.ELEMENTARY_CHARGE * p
    C_total = np.gradient(Qn, V_drain) + np.gradient(Qp, V_drain)
    C_total_pF_cm = C_total * 1e10  # F/m → pF/cm

    if should_plot():
        plot_capacitance(V_drain, C_total_pF_cm)
    return V_drain, C_total_pF_cm


def process_optical_data(neff, length, wavelength, sim_id=None):
    """Compute d_neff, loss, phase shift, V_pi, and max |Δφ| from the FDE neff sweep."""
    V = np.linspace(0, config.V_MAX, len(neff))
    d_neff = calc_dneff(neff)
    alpha_dB_per_cm = calc_alpha(neff, wavelength)
    d_phi = calc_dphi(d_neff, length, wavelength)

    abs_dphi = np.abs(d_phi)
    max_dphi = np.max(abs_dphi) if len(abs_dphi) > 0 else 0.0
    v_pi = calculate_v_pi(V, abs_dphi)

    if should_plot():
        plot_optical_results(V, d_neff, alpha_dB_per_cm, d_phi, v_pi)
    return d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi


def calc_alpha(neff, wavelength):
    """alpha [dB/cm] = 2·k0·Im(neff) · (10/ln 10) · 1e-2."""
    k0 = 2 * np.pi / wavelength
    return 2 * k0 * np.imag(neff) * (10 / np.log(10)) * 1e-2


def calc_dneff(neff):
    return np.real(neff - neff[0])


def calc_dphi(d_neff, length, wavelength):
    return (2 * np.pi * d_neff * length) / wavelength


def calculate_v_pi(voltages, abs_dphi):
    """Linear interpolation of |Δφ|(V) at π. Returns NaN if π is never reached."""
    if np.max(abs_dphi) < np.pi:
        return np.nan
    return np.interp(np.pi, abs_dphi, voltages)


def plot_capacitance(V, C_total_pF_cm):
    plt = _get_plt()
    plt.figure("Capacitance vs Voltage")
    plt.plot(V, C_total_pF_cm, 'b-o', markersize=4, label='Total capacitance')
    plt.title("Diode capacitance vs. voltage")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Capacitance (pF/cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def plot_optical_results(V, d_neff, alpha_dB_per_cm, d_phi, v_pi=None):
    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(V, d_neff, 'b-o', markersize=4)
    axes[0].set_title("Δn_eff vs. voltage")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Δn_eff (real)")
    axes[0].grid(True)

    axes[1].plot(V, alpha_dB_per_cm, 'r-s', markersize=4)
    axes[1].set_title("Optical loss vs. voltage")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Loss (dB/cm)")
    axes[1].grid(True)

    axes[2].plot(V, d_phi, 'g-^', markersize=4)
    axes[2].axhline(y=np.pi, color='k', linestyle='--', alpha=0.5, label='π')
    axes[2].axhline(y=-np.pi, color='k', linestyle='--', alpha=0.5)
    if v_pi is not None and not np.isnan(v_pi):
        axes[2].axvline(x=v_pi, color='m', linestyle=':', alpha=0.7, label=f'V_π = {v_pi:.3f} V')
        axes[2].legend()
    axes[2].set_title("Phase shift vs. voltage")
    axes[2].set_xlabel("Voltage (V)")
    axes[2].set_ylabel("Δφ (rad)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show(block=False)
