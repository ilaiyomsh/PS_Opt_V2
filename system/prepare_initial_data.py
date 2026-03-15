"""
Prepare initial data for BO from archived results.

Reads the first N rows from an archived result_full.csv,
recalculates costs using the new linear formula, and saves
to simulation csv/ as result.csv and result_full.csv.

Usage:
    python prepare_initial_data.py [archive_path] [n_rows]

Defaults:
    archive_path = results_archive/result_20260202_025520/result_full.csv
    n_rows = 60
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def prepare_data(archive_path, n_rows):
    """Load first n_rows from archive, recalculate costs, save to simulation csv/."""

    print(f"Loading archive: {archive_path}")
    df = pd.read_csv(archive_path)
    print(f"  Total rows in archive: {len(df)}")

    df = df.head(n_rows).copy()
    print(f"  Using first {len(df)} rows")

    # --- Fix failed rows: use worst-case values instead of NaN ---
    failed_mask = df['v_pi_V'].isna()
    n_failed = failed_mask.sum()
    n_valid = len(df) - n_failed
    print(f"  Valid: {n_valid}, Failed: {n_failed}")

    # For failed rows: loss_at_v_pi = loss_max (worst-case from sweep)
    df.loc[failed_mask, 'loss_at_v_pi_dB_per_cm'] = df.loc[failed_mask, 'loss_max_dB_per_cm']

    # For failed rows: v_pi_l = V_MAX * length * 1e3 (V*mm)
    df.loc[failed_mask, 'v_pi_l_Vmm'] = config.V_MAX * df.loc[failed_mask, 'length'] * 1e3

    # --- Recalculate cost for ALL rows with new linear formula ---
    w_loss = config.FOM_WEIGHTS['loss']
    w_vpil = config.FOM_WEIGHTS['vpil']
    t_loss = config.TARGETS['loss']
    t_vpil = config.TARGETS['vpil']

    df['norm_loss'] = df['loss_at_v_pi_dB_per_cm'] / t_loss
    df['norm_vpil'] = df['v_pi_l_Vmm'] / t_vpil
    df['cost'] = w_loss * df['norm_loss'] + w_vpil * df['norm_vpil']

    # --- Print summary ---
    valid_costs = df.loc[~failed_mask, 'cost']
    failed_costs = df.loc[failed_mask, 'cost']

    print(f"\n  Cost summary (new linear formula):")
    print(f"    Valid  — min: {valid_costs.min():.2f}, max: {valid_costs.max():.2f}, median: {valid_costs.median():.2f}")
    print(f"    Failed — min: {failed_costs.min():.2f}, max: {failed_costs.max():.2f}, median: {failed_costs.median():.2f}")
    print(f"    Ratio (max_failed / max_valid): {failed_costs.max() / valid_costs.max():.1f}x")

    # --- Save ---
    # Full file
    full_path = config.RESULTS_FULL_CSV_FILE
    df.to_csv(full_path, index=False, float_format='%.6e')
    print(f"\n  Saved {len(df)} rows to {full_path}")

    # Minimal file (subset of columns)
    minimal_cols = [c for c in config.MINIMAL_RESULT_COLUMNS if c in df.columns]
    df[minimal_cols].to_csv(config.RESULTS_CSV_FILE, index=False, float_format='%.6e')
    print(f"  Saved {len(df)} rows to {config.RESULTS_CSV_FILE}")

    # Show a few examples
    print(f"\n  Sample valid row:")
    sample_valid = df[~failed_mask].iloc[0]
    print(f"    sim_id={int(sample_valid['sim_id'])}: loss={sample_valid['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
          f"VpiL={sample_valid['v_pi_l_Vmm']:.3f} V*mm, cost={sample_valid['cost']:.2f}")

    print(f"  Sample failed row:")
    sample_failed = df[failed_mask].iloc[0]
    print(f"    sim_id={int(sample_failed['sim_id'])}: loss={sample_failed['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
          f"VpiL={sample_failed['v_pi_l_Vmm']:.3f} V*mm, cost={sample_failed['cost']:.2f}")

    print(f"\nDone. Set SKIP_INITIAL_SIMS=True and MAX_ITERATIONS=10 in config.py, then run main.py")


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_archive = os.path.join(base, "results_archive", "result_20260202_025520", "result_full.csv")

    archive_path = sys.argv[1] if len(sys.argv) > 1 else default_archive
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    prepare_data(archive_path, n_rows)
