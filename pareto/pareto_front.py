# pareto/pareto_front.py
# Build a multi-objective Optuna study from existing simulation results
# and identify / plot the Pareto front for (V_pi*L, optical loss).
#
# Self-contained: no dependency on system/config.py.
#
# Install once:
#     pip install optuna plotly
#
# Usage:
#     cd pareto && python pareto_front.py

import os
import sys

import numpy as np
import pandas as pd

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial


_PARETO_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Constants (mirrored from system/config.py) ----
SWEEP_PARAMETERS = {
    'w_r':    {'min': 350e-9,  'max': 500e-9},
    'h_si':   {'min': 70e-9,   'max': 130e-9},
    'doping': {'min': 1e17,    'max': 1e20},
    'S':      {'min': 0,       'max': 0.8e-6},
    'lambda': {'min': 1260e-9, 'max': 1360e-9},
    'length': {'min': 0.1e-3,  'max': 1.0e-3},
}
FOM_WEIGHTS = {'loss': 0.3, 'vpil': 0.7}
TARGETS = {'loss': 20.0, 'vpil': 1.0}
RESULTS_CSV = os.path.join(_PARETO_DIR, "result.csv")

OBJECTIVES = ("v_pi_l_Vmm", "loss_at_v_pi_dB_per_cm")
TARGET_NAMES = ("V_pi*L (V*mm)", "Loss (dB/cm)")
LOG_PARAMS = {"doping"}
MAX_LOSS_DB_PER_CM = 50.0

PARETO_HTML = os.path.join(_PARETO_DIR, "pareto_front.html")
PARETO_CSV = os.path.join(_PARETO_DIR, "pareto_front.csv")


def build_distributions():
    """Optuna distributions that mirror SWEEP_PARAMETERS bounds."""
    return {
        name: FloatDistribution(low=cfg["min"], high=cfg["max"], log=name in LOG_PARAMS)
        for name, cfg in SWEEP_PARAMETERS.items()
    }


def load_valid_results(csv_path):
    """Keep only sims that reached pi (valid V_pi) and have both objectives."""
    df = pd.read_csv(csv_path)
    needed = list(OBJECTIVES) + ["v_pi_V"]
    valid = df.dropna(subset=needed).copy()
    n_valid_before_loss_filter = len(valid)
    valid = valid[valid[OBJECTIVES[1]] <= MAX_LOSS_DB_PER_CM].copy().reset_index(drop=True)
    n_filtered = n_valid_before_loss_filter - len(valid)
    print(
        f"Loaded {len(df)} rows; {n_valid_before_loss_filter} are valid (reached pi); "
        f"{n_filtered} filtered out with loss > {MAX_LOSS_DB_PER_CM} dB/cm."
    )
    return valid


def build_study(valid_df, distributions):
    """Create a min/min multi-objective study and inject one trial per row."""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="ps_opt_pareto",
    )
    for _, row in valid_df.iterrows():
        params = {}
        for name, dist in distributions.items():
            params[name] = float(np.clip(float(row[name]), dist.low, dist.high))
        trial = create_trial(
            params=params,
            distributions=distributions,
            values=[float(row[OBJECTIVES[0]]), float(row[OBJECTIVES[1]])],
        )
        study.add_trial(trial)
    return study


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        sys.exit(1)

    valid = load_valid_results(RESULTS_CSV)
    if valid.empty:
        print("No valid simulations to analyze. Exiting.")
        sys.exit(1)

    distributions = build_distributions()
    study = build_study(valid, distributions)

    pareto_trials = study.best_trials
    pareto_idx = sorted(t.number for t in pareto_trials)
    pareto_df = valid.iloc[pareto_idx].sort_values(OBJECTIVES[0]).reset_index(drop=True)

    print(f"Pareto front: {len(pareto_df)} non-dominated designs out of {len(valid)} valid sims.")

    try:
        fig = optuna.visualization.plot_pareto_front(
            study, target_names=list(TARGET_NAMES)
        )
        fig.write_html(PARETO_HTML)
        print(f"  -> Plot:  {PARETO_HTML}")
    except (ImportError, ValueError) as e:
        print(f"  [WARN] Plot not written ({e}). Install plotly: pip install plotly")

    pareto_df.to_csv(PARETO_CSV, index=False, float_format="%.6e")
    print(f"  -> CSV:   {PARETO_CSV}")

    cols = ["sim_id", *OBJECTIVES, "v_pi_V", "cost"]
    cols = [c for c in cols if c in pareto_df.columns]
    print("\nPareto-optimal designs (sorted by V_pi*L):")
    print(pareto_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
