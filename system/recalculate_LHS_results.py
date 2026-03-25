"""
Recalculate cost function for Legacy_LHS_Results.csv

This script:
1. Reads the Legacy_LHS_Results.csv file
2. Recalculates the cost using the corrected cost function
3. Saves the results to simulation csv/ directory for BO usage
"""

import sys
import os
import pandas as pd
import numpy as np

# Add system directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))
import config
import cost as cost_module


def recalculate_legacy_costs(legacy_csv_path, output_dir=None):
    """
    Recalculate costs for legacy LHS results using the corrected cost function.
    
    Parameters:
    -----------
    legacy_csv_path : str
        Path to Legacy_LHS_Results.csv
    output_dir : str, optional
        Output directory. If None, uses config.SIMULATION_CSV_DIR
    """
    
    print(f"Loading legacy results: {legacy_csv_path}")
    df = pd.read_csv(legacy_csv_path)
    
    # Remove empty rows (all NaN)
    df = df.dropna(how='all').copy()
    print(f"  Total rows loaded: {len(df)}")
    
    # --- Fix failed rows: use worst-case values ---
    # Check is_valid column (TRUE/FALSE) or use v_pi_V NaN status
    if 'is_valid' in df.columns:
        failed_mask = df['is_valid'] == False
    else:
        failed_mask = df['v_pi_V'].isna()
    
    n_failed = failed_mask.sum()
    n_valid = len(df) - n_failed
    print(f"  Valid: {n_valid}, Failed: {n_failed}")
    
    # For failed rows: use loss_max instead of loss_at_v_pi
    df.loc[failed_mask, 'loss_at_v_pi_dB_per_cm'] = df.loc[failed_mask, 'loss_max_dB_per_cm']
    
    # For failed rows: v_pi_l = V_MAX * length * 1e3 (V*mm)
    df.loc[failed_mask, 'v_pi_l_Vmm'] = config.V_MAX * df.loc[failed_mask, 'length'] * 1e3
    
    # --- Recalculate cost for ALL rows using the corrected cost function ---
    print("\nRecalculating costs with corrected cost function...")
    
    df['cost'] = df.apply(
        lambda row: -cost_module.calculate_cost(
            alpha=row['loss_at_v_pi_dB_per_cm'],
            v_pi_l=row['v_pi_l_Vmm'],
            max_dphi=row['max_dphi_rad']
        ),
        axis=1
    )
    
    # Recalculate normalized columns (for reporting)
    df['norm_loss'] = df['loss_at_v_pi_dB_per_cm'] / config.TARGETS['loss']
    df['norm_vpil'] = df['v_pi_l_Vmm'] / config.TARGETS['vpil']
    
    # --- Print summary ---
    valid_costs = df.loc[~failed_mask, 'cost']
    failed_costs = df.loc[failed_mask, 'cost']
    
    print(f"\n  Cost summary (corrected piecewise quadratic penalty):")
    print(f"    Valid  — min: {valid_costs.min():.2f}, max: {valid_costs.max():.2f}, median: {valid_costs.median():.2f}")
    print(f"    Failed — min: {failed_costs.min():.2f}, max: {failed_costs.max():.2f}, median: {failed_costs.median():.2f}")
    if valid_costs.max() > 0:
        print(f"    Ratio (max_failed / max_valid): {failed_costs.max() / valid_costs.max():.1f}x")
    
    # --- Save to simulation csv directory ---
    if output_dir is None:
        output_dir = config.SIMULATION_CSV_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    full_path = os.path.join(output_dir, "result_full.csv")
    df.to_csv(full_path, index=False, float_format='%.6e')
    print(f"\n  Saved {len(df)} rows to {full_path}")
    
    # Save minimal results (subset of columns for BO)
    minimal_cols = [c for c in config.MINIMAL_RESULT_COLUMNS if c in df.columns]
    minimal_path = os.path.join(output_dir, "result.csv")
    df[minimal_cols].to_csv(minimal_path, index=False, float_format='%.6e')
    print(f"  Saved {len(df)} rows (minimal columns) to {minimal_path}")
    
    # Show examples
    print(f"\n  Sample valid row:")
    if n_valid > 0:
        sample_valid = df[~failed_mask].iloc[0]
        print(f"    sim_id={int(sample_valid['sim_id'])}: loss={sample_valid['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
              f"VpiL={sample_valid['v_pi_l_Vmm']:.3f} V*mm, max_dphi={sample_valid['max_dphi_rad']:.2f} rad, "
              f"cost={sample_valid['cost']:.2f}")
    
    print(f"\n  Sample failed row:")
    if n_failed > 0:
        sample_failed = df[failed_mask].iloc[0]
        print(f"    sim_id={int(sample_failed['sim_id'])}: loss={sample_failed['loss_at_v_pi_dB_per_cm']:.2f} dB/cm, "
              f"VpiL={sample_failed['v_pi_l_Vmm']:.3f} V*mm, max_dphi={sample_failed['max_dphi_rad']:.2f} rad, "
              f"cost={sample_failed['cost']:.2f}")
    
    print(f"\n✓ Done! You can now use these results for BO.")
    print(f"  Set SKIP_INITIAL_SIMS=True in config.py and run main.py")
    
    return df


if __name__ == '__main__':
    # Get the base directory (PS_Opt_V2)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default path to Legacy_LHS_Results.csv
    default_legacy_path = os.path.join(base_dir, "results_archive", "Legacy_LHS_Results.csv")
    
    # Allow command line override
    legacy_path = sys.argv[1] if len(sys.argv) > 1 else default_legacy_path
    
    # Check if file exists
    if not os.path.exists(legacy_path):
        print(f"Error: File not found: {legacy_path}")
        sys.exit(1)
    
    # Run the recalculation
    recalculate_legacy_costs(legacy_path)