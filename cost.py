# system/cost.py
# Cost function for PIN diode phase shifter optimization
#   Dual-condition piecewise penalty:
#     Branch A — valid (Δφ_max ≥ π AND α ≤ ALPHA_MAX):
#         cost = w_loss*(α/T_loss)² + w_vpil*(V_π*L/T_vpil)²
#     Branch B — failed (either condition unmet):
#         cost = C_BASE + BETA_ELEC*(max(0, π−Δφ_max))² + BETA_OPT*max(0, α−ALPHA_MAX)

import config
import numpy as np


def calculate_cost(alpha, v_pi_l, max_dphi=np.inf, weights=None, targets=None):
    """
    Calculates cost using the dual-condition piecewise penalty.
    Returns negative value for BayesOpt maximization.

    Parameters:
    -----------
    alpha : float
        Optical loss (dB/cm). For valid sims: loss at V_pi.
        For failed sims: max loss from sweep.
    v_pi_l : float
        Voltage-length product (V*mm). For valid sims: V_pi * L.
        For failed sims: V_MAX * L.
    max_dphi : float, optional
        Maximum phase shift achieved (radians). Defaults to np.inf (assumes
        electrically valid when not provided).
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
    Conditions:
        is_electrically_valid = max_dphi >= π
        is_optically_valid    = alpha <= config.ALPHA_MAX

    Branch A — Both conditions True (valid device):
        cost = w_loss*(α/T_loss)² + w_vpil*(V_π*L/T_vpil)²

    Branch B — One or both conditions False (failed device):
        elec_penalty = BETA_ELEC * max(0, π - max_dphi)²  [quadratic]
        opt_penalty  = BETA_OPT  * max(0, α - ALPHA_MAX)  [linear]
        cost = C_BASE + elec_penalty + opt_penalty

    C_BASE is dynamically derived in config.py so it always exceeds the
    theoretical worst Branch A cost, preventing cost inversion.
    """
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    is_electrically_valid = max_dphi >= np.pi
    is_optically_valid = alpha <= config.ALPHA_MAX

    # Branch A: Valid Device (both conditions met)
    if is_electrically_valid and is_optically_valid:
        # Failsafe: Destroy the cost score if Lumerical returns optical gain
        if alpha < 0:
            return -(config.C_BASE + 50.0)  # Massive penalty for unphysical gain results
        # Normalize by targets and apply weights
        norm_loss = alpha / targets['loss']
        norm_vpil = v_pi_l / targets['vpil']
        # Quadratic (squared) terms per Equation 27
        cost = (weights['loss'] * (norm_loss**2)) + (weights['vpil'] * (norm_vpil**2))

    # Branch B: Failed Device (one or both conditions failed)
    else:
        elec_penalty = config.BETA_ELEC * (max(0, np.pi - max_dphi)**2)
        opt_penalty = config.BETA_OPT * max(0, alpha - config.ALPHA_MAX)
        cost = config.C_BASE + elec_penalty + opt_penalty

    return -cost
