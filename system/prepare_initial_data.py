"""
Seed simulation csv/result.csv from an archived run for a BO-only restart.

Reads the first N rows of an archived result_full.csv, recomputes costs using
the current piecewise-quadratic formula, and writes both result.csv and
result_full.csv to simulation csv/.

Usage: python prepare_initial_data.py [archive_path] [n_rows]
Defaults: results_archive/result_20260202_025520/result_full.csv, 60 rows
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cost as cost_module


def prepare_data(archive_path, n_rows):
    df = pd.read_csv(archive_path).head(n_rows).copy()
    failed_mask = df['v_pi_V'].isna()

    # Failed rows: use worst-case loss from the sweep, and V_MAX * L for V_pi·L.
    df.loc[failed_mask, 'loss_at_v_pi_dB_per_cm'] = df.loc[failed_mask, 'loss_max_dB_per_cm']
    df.loc[failed_mask, 'v_pi_l_Vmm'] = config.V_MAX * df.loc[failed_mask, 'length'] * 1e3

    df['cost'] = df.apply(
        lambda row: -cost_module.calculate_cost(
            alpha=row['loss_at_v_pi_dB_per_cm'],
            v_pi_l=row['v_pi_l_Vmm'],
            max_dphi=row['max_dphi_rad'],
        ),
        axis=1,
    )
    df['norm_loss'] = df['loss_at_v_pi_dB_per_cm'] / config.TARGETS['loss']
    df['norm_vpil'] = df['v_pi_l_Vmm'] / config.TARGETS['vpil']

    df.to_csv(config.RESULTS_FULL_CSV_FILE, index=False, float_format='%.6e')
    minimal_cols = [c for c in config.MINIMAL_RESULT_COLUMNS if c in df.columns]
    df[minimal_cols].to_csv(config.RESULTS_CSV_FILE, index=False, float_format='%.6e')

    n_failed = failed_mask.sum()
    print(f"Seeded {len(df)} rows ({len(df) - n_failed} valid, {n_failed} failed) → "
          f"{config.RESULTS_CSV_FILE}")
    print("Set SKIP_INITIAL_SIMS=True in config.py, then run main.py.")


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_archive = os.path.join(base, "results_archive", "result_20260202_025520", "result_full.csv")
    archive_path = sys.argv[1] if len(sys.argv) > 1 else default_archive
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    prepare_data(archive_path, n_rows)
