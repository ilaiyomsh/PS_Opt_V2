# test_extract_neff.py
# Test script to run a full simulation and visualize the neff data extraction flow
# Verifies that imaginary part of neff is preserved through np.squeeze()

import sys
import os

# Add Lumerical API to path
sys.path.append("C:\\Program Files\\Lumerical\\v231\\api\\python")

import numpy as np
import pandas as pd

# Import local modules
import config
import sim_handler
import data_processor

try:
    import lumapi
except ImportError:
    print("[ERROR] lumapi not found. Make sure Lumerical is installed.")
    sys.exit(1)

# Output CSV file
OUTPUT_CSV = os.path.join(config.SIMULATION_CSV_DIR, "neff_raw_data.csv")

# Test parameters
TEST_PARAMS = {
    'w_r': 373.4e-9,      # 373.4 nm
    'h_si': 98.3e-9,      # 98.3 nm
    'S': 34.0e-9,         # 34.0 nm
    'doping': 2.91e17,    # 2.91e+17 cm^-3
    'lambda': 1289e-9,    # 1289 nm
    'length': 0.18e-3     # 0.18 mm
}


def main():
    print("=" * 60)
    print("FULL SIMULATION + NEFF DATA EXTRACTION TEST")
    print("=" * 60)

    # Show test parameters
    print("\nTest Parameters:")
    print(f"  | w_r       | {TEST_PARAMS['w_r']*1e9:.1f} nm       |")
    print(f"  | h_si      | {TEST_PARAMS['h_si']*1e9:.1f} nm       |")
    print(f"  | S         | {TEST_PARAMS['S']*1e9:.1f} nm       |")
    print(f"  | doping    | {TEST_PARAMS['doping']:.2e} cm^-3|")
    print(f"  | lambda    | {TEST_PARAMS['lambda']*1e9:.0f} nm       |")
    print(f"  | length    | {TEST_PARAMS['length']*1e3:.2f} mm       |")

    # Path to save charge data
    charge_data_path = config.CHARGE_DATA_FILE

    # ========== A. CHARGE Simulation (Electrical) ==========
    print("\n" + "=" * 60)
    print("[A] CHARGE SIMULATION (Electrical)")
    print("=" * 60)

    print("\n[A1] Opening CHARGE session...")
    try:
        charge = lumapi.DEVICE(hide=False)
        charge.load(config.CHARGE_SIM_FILE)
        print(f"     Loaded: {config.CHARGE_SIM_FILE}")
    except Exception as e:
        print(f"     [ERROR] Failed to open CHARGE: {e}")
        return

    print("\n[A2] Setting CHARGE parameters...")
    try:
        sim_handler.set_charge_parameters(charge, TEST_PARAMS, charge_data_path)
        print("     Parameters set successfully.")
    except Exception as e:
        print(f"     [ERROR] Failed to set parameters: {e}")
        charge.close()
        return

    print("\n[A3] Running CHARGE simulation...")
    try:
        sim_handler.run_charge_simulation(charge)
        print("     CHARGE simulation completed.")
    except Exception as e:
        print(f"     [ERROR] CHARGE simulation failed: {e}")
        charge.close()
        return

    print("\n[A4] Extracting capacitance data...")
    try:
        V_cap, C_total_pF_cm = data_processor.extract_capacitance(charge, sim_id=0)
        print(f"     Extracted {len(V_cap)} voltage points")
        print(f"     Capacitance range: [{C_total_pF_cm.min():.2f}, {C_total_pF_cm.max():.2f}] pF/cm")
    except Exception as e:
        print(f"     [ERROR] Failed to extract capacitance: {e}")
        charge.close()
        return

    charge.close()
    print("     CHARGE session closed.")

    # ========== B. FDE Simulation (Optical) ==========
    print("\n" + "=" * 60)
    print("[B] FDE SIMULATION (Optical)")
    print("=" * 60)

    print("\n[B1] Opening FDE session...")
    try:
        fde = lumapi.MODE(hide=False)
        fde.load(config.FDE_SIM_FILE)
        print(f"     Loaded: {config.FDE_SIM_FILE}")
    except Exception as e:
        print(f"     [ERROR] Failed to open FDE: {e}")
        return

    print("\n[B2] Setting FDE parameters...")
    try:
        sim_handler.set_fde_parameters(fde, TEST_PARAMS)
        print("     Parameters set successfully.")
    except Exception as e:
        print(f"     [ERROR] Failed to set FDE parameters: {e}")
        fde.close()
        return

    print("\n[B3] Importing CHARGE data...")
    try:
        sim_handler.import_charge_data(fde, charge_data_path)
        print("     CHARGE data imported into FDE.")
    except Exception as e:
        print(f"     [ERROR] Failed to import CHARGE data: {e}")
        fde.close()
        return

    print("\n[B4] Running FDE voltage sweep...")
    try:
        sim_handler.run_fde_sweep(fde)
        print("     FDE sweep completed.")
    except Exception as e:
        print(f"     [ERROR] FDE sweep failed: {e}")
        fde.close()
        return

    # ========== C. DATA EXTRACTION ==========
    print("\n" + "=" * 60)
    print("[C] DATA EXTRACTION & VERIFICATION")
    print("=" * 60)

    # Extract raw sweep_result
    print("\n[C1] Extracting raw sweep_result...")
    try:
        sweep_result = fde.getsweepresult("voltage", "neff")
        print(f"     sweep_result type: {type(sweep_result)}")
        print(f"     sweep_result keys: {list(sweep_result.keys())}")
    except Exception as e:
        print(f"     [ERROR] Failed to get sweep result: {e}")
        fde.close()
        return

    # Get raw neff
    print("\n[C2] Extracting neff from sweep_result...")
    neff_raw = sweep_result['neff']

    print(f"     neff_raw type:  {type(neff_raw)}")
    print(f"     neff_raw dtype: {neff_raw.dtype}")
    print(f"     neff_raw shape: {neff_raw.shape}")
    print(f"     Is complex?     {np.iscomplexobj(neff_raw)}")

    # Show first few raw values
    print("\n     First 5 raw values:")
    flat_raw = neff_raw.flatten()[:5]
    for i, val in enumerate(flat_raw):
        print(f"       [{i}] {val}  (real={val.real:.6f}, imag={val.imag:.6e})")

    # Apply np.squeeze
    print("\n[C3] Applying np.squeeze()...")
    neff = np.squeeze(neff_raw)

    print(f"     neff type:  {type(neff)}")
    print(f"     neff dtype: {neff.dtype}")
    print(f"     neff shape: {neff.shape}")
    print(f"     Is complex? {np.iscomplexobj(neff)}")

    # Show first few squeezed values
    print("\n     First 5 squeezed values:")
    for i, val in enumerate(neff[:5]):
        print(f"       [{i}] {val}  (real={val.real:.6f}, imag={val.imag:.6e})")

    # Verify imaginary part preserved
    print("\n[C4] Verification: Imaginary part preservation")
    imag_before = np.imag(neff_raw).flatten()[:5]
    imag_after = np.imag(neff)[:5]

    print("     Comparing imaginary parts (before vs after squeeze):")
    all_match = True
    for i in range(min(5, len(imag_after))):
        match = "✓" if imag_before[i] == imag_after[i] else "✗"
        if imag_before[i] != imag_after[i]:
            all_match = False
        print(f"       [{i}] {imag_before[i]:.6e} vs {imag_after[i]:.6e}  {match}")

    all_match = np.allclose(np.imag(neff_raw).flatten(), np.imag(neff).flatten())
    print(f"\n     All imaginary values match: {all_match}")

    # Show data processing flow
    print("\n[C5] Data processing flow (as used in data_processor.py)...")
    wavelength = TEST_PARAMS['lambda']
    k0 = 2 * np.pi / wavelength

    # Calculate alpha (loss)
    alpha = 2 * k0 * np.imag(neff) * (10 / np.log(10)) * 1e-2

    print(f"     Wavelength: {wavelength*1e9:.0f} nm")
    print(f"     k0: {k0:.2e}")
    print(f"\n     Loss (alpha) for first 5 points:")
    for i in range(min(5, len(alpha))):
        print(f"       V[{i}]: imag(neff)={np.imag(neff[i]):.6e} -> alpha={alpha[i]:.4f} dB/cm")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total voltage points: {len(neff)}")
    print(f"  Real neff range:      [{np.real(neff).min():.6f}, {np.real(neff).max():.6f}]")
    print(f"  Imag neff range:      [{np.imag(neff).min():.6e}, {np.imag(neff).max():.6e}]")
    print(f"  Loss range:           [{alpha.min():.4f}, {alpha.max():.4f}] dB/cm")
    print(f"  Imaginary preserved:  {all_match}")
    print("=" * 60)

    # Save raw data to CSV
    print("\n[C6] Saving raw data to CSV...")

    # Reconstruct voltage array
    num_points = len(neff)
    V = np.linspace(0, config.V_MAX, num_points)

    # Create DataFrame with all data
    df = pd.DataFrame({
        'voltage_V': V,
        'neff_real': np.real(neff),
        'neff_imag': np.imag(neff),
        'neff_complex': [f"{val.real}+{val.imag}j" for val in neff],  # String representation
        'alpha_dB_per_cm': alpha
    })

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"     Saved to: {OUTPUT_CSV}")
    print(f"     Columns: {list(df.columns)}")
    print(f"     Rows: {len(df)}")

    # Keep session open for inspection
    input("\nPress Enter to close FDE session...")
    fde.close()
    print("Done.")


if __name__ == "__main__":
    main()
