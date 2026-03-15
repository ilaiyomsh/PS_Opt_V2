# system/cost.py
# Cost function for PIN diode phase shifter optimization
#   cost = w_loss * (alpha/target_loss) + w_vpil * (vpil/target_vpil)
# Same formula for valid and failed simulations (failed uses worst-case values).

import config
import numpy as np


def calculate_cost(alpha, v_pi_l, weights=None, targets=None):
    """
    Calculates cost. Returns negative value for BayesOpt maximization.

    For valid sims: alpha = loss at V_pi, v_pi_l = V_pi * L
    For failed sims: alpha = max loss from sweep, v_pi_l = V_MAX * L

    Same linear formula in both cases — no discontinuity.
    """
    if weights is None:
        weights = config.FOM_WEIGHTS
    if targets is None:
        targets = config.TARGETS

    norm_loss = alpha / targets['loss']
    norm_vpil = v_pi_l / targets['vpil']
    cost = weights['loss'] * norm_loss + weights['vpil'] * norm_vpil

    return -cost
