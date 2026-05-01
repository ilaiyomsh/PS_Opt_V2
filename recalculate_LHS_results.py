"""
Recalculate cost function for existing simulation CSV results.

This script:
1. Reads result_full.csv from the simulation csv/ directory
2. Recalculates cost, norm_loss, and norm_vpil using the current cost.py / config.py
3. Overwrites result_full.csv and result.csv in-place
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure the system package is importable whether run from repo root or system/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import config
import cost as cost_module


def recalculate_costs():
    """
    Recalculate cost, norm_loss, and norm_vpil for all rows in result_full.csv
    using the current cost function parameters from config.py and cost.py.
    Saves updated result_full.csv and result.csv in-place.
    """

    full_path = config.RESULTS_FULL_CSV_FILE
    minimal_path = config.RESULTS_CSV_FILE

    if not os.path.exists(full_path):
        print(f"Error: File not found: {full_path}")
        sys.exit(1)

    print(f"Loading: {full_path}")
    df = pd.read_csv(full_path)
    df = df.dropna(how='all').copy()
    print(f"  Rows loaded: {len(df)}")

    # --- Determine valid / failed rows ---
    if 'is_valid' in df.columns:
        failed_mask = df['is_valid'].astype(str).str.upper() == 'FALSE'
    else:
        failed_mask = df['v_pi_V'].isna()

    n_failed = int(failed_mask.sum())
    n_valid = len(df) - n_failed
    print(f"  Valid: {n_valid}, Failed: {n_failed}")

    # --- For failed rows: enforce worst-case input values ---
    if 'loss_max_dB_per_cm' in df.columns:
        df.loc[failed_mask, 'loss_at_v_pi_dB_per_cm'] = df.loc[failed_mask, 'loss_max_dB_per_cm']
    df.loc[failed_mask, 'v_pi_l_Vmm'] = config.V_MAX * df.loc[failed_mask, 'length'] * 1e3

    # --- Recalculate cost ---
    # calculate_cost() returns -cost (negative, for maximization).
    # The CSV stores the positive penalty value, so we negate the return.
    print("Recalculating cost, norm_loss, norm_vpil ...")
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

    # --- Print summary ---
    valid_costs = df.loc[~failed_mask, 'cost']
    failed_costs = df.loc[failed_mask, 'cost']

    print(f"\n  Cost summary (piecewise quadratic penalty):")
    if n_valid:
        print(f"    Valid  — min: {valid_costs.min():.4f}, max: {valid_costs.max():.4f}, "
              f"median: {valid_costs.median():.4f}")
    if n_failed:
        print(f"    Failed — min: {failed_costs.min():.4f}, max: {failed_costs.max():.4f}, "
              f"median: {failed_costs.median():.4f}")
    if n_valid and n_failed and valid_costs.max() > 0:
        print(f"    Ratio (max_failed / max_valid): {failed_costs.max() / valid_costs.max():.1f}x")

    # Sample rows
    if n_valid:
        s = df[~failed_mask].iloc[0]
        print(f"\n  Sample valid   sim_id={int(s['sim_id'])}: "
              f"loss={s['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
              f"VpiL={s['v_pi_l_Vmm']:.4f} V*mm, "
              f"max_dphi={s['max_dphi_rad']:.2f} rad, cost={s['cost']:.4f}")
    if n_failed:
        s = df[failed_mask].iloc[0]
        print(f"  Sample failed  sim_id={int(s['sim_id'])}: "
              f"loss={s['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
              f"VpiL={s['v_pi_l_Vmm']:.4f} V*mm, "
              f"max_dphi={s['max_dphi_rad']:.2f} rad, cost={s['cost']:.4f}")

    # --- Save results ---
    df.to_csv(full_path, index=False, float_format='%.6e')
    print(f"\n  Saved {len(df)} rows → {full_path}")

    minimal_cols = [c for c in config.MINIMAL_RESULT_COLUMNS if c in df.columns]
    df[minimal_cols].to_csv(minimal_path, index=False, float_format='%.6e')
    print(f"  Saved {len(df)} rows (minimal columns) → {minimal_path}")

    print(f"\n✓ Done. Set SKIP_INITIAL_SIMS=True in config.py and run main.py")

    return df


if __name__ == '__main__':
    recalculate_costs()