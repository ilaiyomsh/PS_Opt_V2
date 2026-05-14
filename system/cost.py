# system/cost.py
# Piecewise penalty for the PIN-diode phase shifter:
#   Branch A — valid (Δφ_max ≥ π AND α ≤ ALPHA_MAX):
#       cost = w_loss·(α/T_loss)² + w_vpil·(V_π·L/T_vpil)²
#   Branch B — failed: α is capped at PHYSICAL_LOSS_CEILING before the penalty.
#       cost = C_BASE + BETA_ELEC·max(0, π−Δφ_max)² + BETA_OPT·max(0, min(α,1000)−ALPHA_MAX)

import config
import numpy as np

PHYSICAL_LOSS_CEILING = 1000.0  # dB/cm — cap to neutralize Lumerical numerical anomalies


def calculate_cost(alpha, v_pi_l, max_dphi=np.inf, weights=None, targets=None):
    """
    Return the **negative** cost so BayesOpt's maximizer minimizes cost.

    alpha       — loss in dB/cm at V_pi (valid) or worst-case from sweep (failed).
    v_pi_l      — V_pi·L in V·mm; for failed sims set to V_MAX·L by the caller.
    max_dphi    — max |Δφ|; defaults to inf (assumes electrically valid if omitted).
    weights     — override config.FOM_WEIGHTS.
    targets     — override config.TARGETS.

    Branch A applies when both electrical (Δφ_max ≥ π) and optical
    (α ≤ ALPHA_MAX) validity hold. Otherwise Branch B (penalty) applies,
    with C_BASE chosen in config.py to exceed the worst Branch A cost.
    """
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    if max_dphi >= np.pi and alpha <= config.ALPHA_MAX:
        if alpha < 0:
            # Lumerical occasionally reports unphysical gain — penalize hard.
            return -(config.C_BASE + 50.0)
        norm_loss = alpha / targets['loss']
        norm_vpil = v_pi_l / targets['vpil']
        cost = weights['loss'] * (norm_loss ** 2) + weights['vpil'] * (norm_vpil ** 2)
    else:
        capped_alpha = min(alpha, PHYSICAL_LOSS_CEILING)
        elec_penalty = config.BETA_ELEC * (max(0, np.pi - max_dphi) ** 2)
        opt_penalty = config.BETA_OPT * max(0, capped_alpha - config.ALPHA_MAX)
        cost = config.C_BASE + elec_penalty + opt_penalty

    return -cost
