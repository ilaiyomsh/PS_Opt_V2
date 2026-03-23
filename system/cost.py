# system/cost.py
# Cost function for PIN diode phase shifter optimization
#   Piecewise quadratic penalty:
#     Valid case (Δφ_max ≥ π): cost = w_loss*(α/T_loss)² + w_vpil*(V_π*L/T_vpil)²
#     Failed case (Δφ_max < π): cost = C_BASE + β*(π - Δφ_max)²

import config
import numpy as np


def calculate_cost(alpha, v_pi_l, max_dphi, weights=None, targets=None):
    """
    Calculates cost using the piecewise quadratic penalty.
    Returns negative value for BayesOpt maximization.

    Parameters:
    -----------
    alpha : float
        Optical loss (dB/cm). For valid sims: loss at V_pi.
        For failed sims: max loss from sweep.
    v_pi_l : float
        Voltage-length product (V*mm). For valid sims: V_pi * L.
        For failed sims: V_MAX * L.
    max_dphi : float
        Maximum phase shift achieved (radians).
        Used to determine valid (≥π) vs failed (<π) case.
    weights : dict, optional
        {'loss': w_loss, 'vpil': w_vpil}. Defaults to config.FOM_WEIGHTS.
    targets : dict, optional
        {'loss': T_loss, 'vpil': T_vpil}. Defaults to config.TARGETS.

    Returns:
    --------
    float
        Negative cost (for BayesOpt maximization). Lower cost is better,
        so more negative return value means better performance.

    Cost Formula:
    -------------
    If max_dphi >= π (valid simulation):
        cost = w_loss*(α/T_loss)² + w_vpil*(V_π*L/T_vpil)²

    If max_dphi < π (failed simulation):
        cost = C_BASE + β*(π - max_dphi)²

    where C_BASE = 35.0 and β = 9*C_BASE/π² ≈ 31.83
    """
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    # Branch 1: Valid Simulation (Reached Pi)
    if max_dphi >= np.pi:
        norm_loss = alpha / targets['loss']
        norm_vpil = v_pi_l / targets['vpil']
        # Quadratic (squared) terms per Equation 27
        cost = (weights['loss'] * (norm_loss**2)) + (weights['vpil'] * (norm_vpil**2))

    # Branch 2: Failed Simulation (Did not reach Pi)
    else:
        # Quadratic penalty based on distance from Pi
        cost = config.C_BASE + config.BETA * ((np.pi - max_dphi)**2)

    return -cost
