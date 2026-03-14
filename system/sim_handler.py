# system/sim_handler.py
# Simulation handler module — the ONLY module that touches lumapi
# Handles both CHARGE (electrical) and FDE/MODE (optical) simulations

import sys
import time
import os
import config
import numpy as np
import pandas as pd

# Import lumapi at module level (only sim_handler touches lumapi)
sys.path.append(config.LUMERICAL_API_PATH)
try:
    import lumapi
except ImportError:
    print("[WARNING] lumapi not found. Lumerical simulations will not work.")
    lumapi = None


class SimulationError(Exception):
    """Custom exception for simulation failures with stage tracking."""
    def __init__(self, stage, message, original_error=None):
        self.stage = stage
        self.original_error = original_error
        super().__init__(f"[{stage}] {message}")


def extract_raw_charge_data(charge_session):
    """Extract V_drain, n, p arrays from CHARGE session via getresult.

    Args:
        charge_session: Lumerical CHARGE session object

    Returns:
        dict: {'V_drain': array, 'n': array, 'p': array}
    """
    charge_data = charge_session.getresult("CHARGE::monitor_charge", "total_charge")
    return {
        'V_drain': charge_data['V_drain'].flatten(),
        'n': charge_data['n'].flatten(),
        'p': charge_data['p'].flatten(),
    }


def extract_raw_optical_data(fde_session):
    """Extract neff array from FDE sweep via getsweepresult.

    Args:
        fde_session: Lumerical FDE/MODE session object

    Returns:
        dict: {'neff': array}
    """
    sweep_result = fde_session.getsweepresult("voltage", "neff")
    return {'neff': np.squeeze(sweep_result['neff'])}


def set_charge_parameters(charge_session, params, charge_file_path):
    """
    Set the CHARGE simulation parameters based on the provided dictionary.

    Args:
        charge_session: Lumerical CHARGE session object
        params (dict): Dictionary containing simulation parameters:
                      - w_r: waveguide width (m)
                      - h_si: silicon height (m)
                      - S: junction offset (m)
                      - doping: doping concentration (cm^-3)
                      - length: device length (m)
        charge_file_path (str): Path to save charge data file (.mat)

    Returns:
        None

    Side effects:
        Modifies the CHARGE simulation geometry and doping parameters.
    """
    # 1. Extract independent variables and CAST TO FLOAT
    # Lumerical API sometimes fails with numpy types
    w_r = float(params['w_r'])
    h_si = float(params['h_si'])
    S = float(params['S'])
    doping_cm3 = float(params['doping'])  # Input is in cm^-3
    length = float(params['length'])

    # Convert doping from cm^-3 to m^-3 (Lumerical uses SI units)
    # 1 cm^-3 = 1e6 m^-3
    doping = doping_cm3 * 1e6  # Now in m^-3 for Lumerical

    # 2. Calculate dependent variable dynamically
    h_r = float(config.WAFER_THICKNESS - h_si)

    print(f"\n--- Setting CHARGE simulation parameters ---")
    charge_session.switchtolayout()

    # Set parameters
    try:
        # Waveguide geometry
        charge_session.select("::model::geometry::waveguide")
        charge_session.set("x span", w_r)
        charge_session.set("y span", length)
        charge_session.set("z max", h_r + h_si)
        charge_session.set("z min", h_si)

        # Pad (slab) geometry
        charge_session.select("::model::geometry::pad")
        charge_session.set("y span", length)
        charge_session.set("z max", h_si)
        charge_session.set("z min", 0.0)

        # Contacts - z_min follows pad top, z_span stays from template
        charge_session.select("::model::geometry::source")
        charge_session.set("y span", length)
        charge_session.set("z min", h_si)  # Bottom aligned with pad top

        charge_session.select("::model::geometry::drain")
        charge_session.set("y span", length)
        charge_session.set("z min", h_si)  # Bottom aligned with pad top

        # Buried Oxide - z_max aligned with pad z_min
        charge_session.select("::model::geometry::buried_oxide")
        charge_session.set("y span", length)
        charge_session.set("z max", 0.0)

        # Surface Oxide - z_min aligned with waveguide z_max
        charge_session.select("::model::geometry::surface_oxide")
        charge_session.set("y span", length)
        charge_session.set("z min", 0.0)

        # Doping regions - S is measured from edge of rib (w_r/2) to doping boundary
        # source_nwell (left side): x_max = -(w_r/2) - S
        # drain_pwell (right side): x_min = (w_r/2) + S

        # --- A. Background Doping (pepi) ---
        # Must cover the entire device volume
        charge_session.select("::model::CHARGE::doping::pepi")
        charge_session.set("y span", length)

        # --- B. Intermediate Wells (Optimization Variables) ---
        # Source N-Well (Left)
        charge_session.select("::model::CHARGE::doping::source_nwell")
        charge_session.set("x min", config.DOPING_X_MIN)  # Fixed outer boundary
        charge_session.set("x max", -(w_r/2) - S)         # Inner boundary depends on w_r and S
        charge_session.set("concentration", doping)
        charge_session.set("z max", h_si)
        charge_session.set("y span", length)

        # Drain P-Well (Right)
        charge_session.select("::model::CHARGE::doping::drain_pwell")
        charge_session.set("x min", (w_r/2) + S)          # Inner boundary depends on w_r and S
        charge_session.set("x max", config.DOPING_X_MAX)  # Fixed outer boundary
        charge_session.set("concentration", doping)
        charge_session.set("z max", h_si)
        charge_session.set("y span", length)

        # --- C. Ohmic Contacts (nplus / pplus) ---
        # Must set concentration higher than well doping to maintain hierarchy:
        #   nplus/pplus (1e20 cm^-3) > wells (1e17-1e18 cm^-3) > pepi (1.5e10 cm^-3)
        # Lumerical uses SI units (m^-3), so 1e20 cm^-3 = 1e26 m^-3

        charge_session.select(f"::model::CHARGE::doping::nplus")
        charge_session.set("concentration", 1e26)  # 1e20 cm^-3 in SI units (m^-3)
        charge_session.set("z max", h_si)
        charge_session.set("y span", length)

        charge_session.select(f"::model::CHARGE::doping::pplus")
        charge_session.set("concentration", 1e26)  # 1e20 cm^-3 in SI units (m^-3)
        charge_session.set("z max", h_si)
        charge_session.set("y span", length)

        # Monitor - set output file path
        charge_session.select("::model::CHARGE::monitor_charge")
        charge_session.set("filename", charge_file_path)

        print(f"  -> Geometry: w_r={w_r:.2e}m, h_si={h_si:.2e}m, h_r={h_r:.2e}m")
        print(f"  -> Doping: S={S:.2e}m, concentration={doping_cm3:.2e}/cm\u00b3 ({doping:.2e}/m\u00b3)")
        print(f"  -> Doping x: source_nwell=[{config.DOPING_X_MIN:.2e}, {-(w_r/2)-S:.2e}]m")
        print(f"  -> Doping x: drain_pwell=[{(w_r/2)+S:.2e}, {config.DOPING_X_MAX:.2e}]m")
        print(f"  -> Length: {length:.2e}m")
        print(f"  -> Contacts: z_min={h_si:.2e}m (z_span from template)")
        print(f"  -> Charge data file: {charge_file_path}")

    except Exception as e:
        print(f"  [ERROR] Failed to set parameters in CHARGE: {e}")
        raise e


def run_charge_simulation(charge_session):
    """
    Runs the CHARGE simulation and verifies that results were generated.

    Args:
        charge_session: Lumerical CHARGE session object

    Returns:
        None

    Raises:
        RuntimeError: If simulation fails or produces no valid results

    Side effects:
        Executes the CHARGE simulation and saves results.
    """
    print("\n--- Running CHARGE simulation ---")
    start_time = time.time()

    try:
        # Save to ensure all parameter changes are written before running
        print("  -> Saving simulation file...", end=" ", flush=True)
        charge_session.save(config.CHARGE_SIM_FILE)
        print("Done.")

        # Generate mesh before running simulation
        print("  -> Generating mesh...", end=" ", flush=True)
        sys.stdout.flush()
        charge_session.mesh()
        print("Done.")

        print("  -> Running solver (this may take several minutes)...", flush=True)
        sys.stdout.flush()
        charge_session.run()

        elapsed = time.time() - start_time
        print(f"  -> CHARGE simulation completed in {elapsed:.1f} seconds.")

        # Verify simulation succeeded by trying to read results
        try:
            test_result = charge_session.getresult("CHARGE::monitor_charge", "total_charge")
            if test_result is None:
                raise RuntimeError("CHARGE simulation produced no results (getresult returned None)")

            # Check if result has data
            if 'V_drain' not in test_result or len(test_result['V_drain']) == 0:
                raise RuntimeError("CHARGE simulation produced empty results (no voltage data)")

            if 'n' not in test_result or 'p' not in test_result:
                raise RuntimeError("CHARGE simulation produced incomplete results (missing charge data)")

            print("  -> CHARGE results verified successfully.")

        except Exception as verify_error:
            elapsed = time.time() - start_time
            error_msg = f"CHARGE simulation failed verification after {elapsed:.1f}s: {verify_error}"
            print(f"  [ERROR] {error_msg}")
            raise RuntimeError(error_msg) from verify_error

    except RuntimeError:
        # Re-raise RuntimeError as-is (already formatted)
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"CHARGE simulation failed after {elapsed:.1f}s: {e}"
        print(f"  [ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e


def set_fde_parameters(fde_session, params):
    """
    Set the FDE/MODE simulation parameters based on the provided dictionary.

    Args:
        fde_session: Lumerical FDE/MODE session object
        params (dict): Dictionary containing simulation parameters:
                      - w_r: waveguide width (m)
                      - h_si: silicon height (m)
                      - length: device length (m)
                      - lambda: wavelength (m)

    Returns:
        None

    Side effects:
        Modifies the FDE/MODE simulation geometry and wavelength parameters.
    """
    # Extract parameters and cast to float
    w_r = float(params['w_r'])
    h_si = float(params['h_si'])
    length = float(params['length'])
    wavelength = float(params['lambda'])

    # Calculate dependent variable
    h_r = float(config.WAFER_THICKNESS - h_si)

    print(f"\n--- Setting FDE simulation parameters ---")
    fde_session.switchtolayout()

    # Set wavelength
    try:
        fde_session.setnamed("FDE", "wavelength", wavelength)
        print(f"  -> Wavelength: {wavelength:.2e}m")
    except Exception as e:
        print(f"  [WARNING] Could not set wavelength: {e}")

    # Set geometry (same as CHARGE, but paths are ::model:: instead of ::model::geometry::)
    try:
        # Waveguide
        fde_session.select("::model::waveguide")
        fde_session.set("x span", w_r)
        fde_session.set("y span", length)
        fde_session.set("z max", h_r + h_si)
        fde_session.set("z min", h_si)

        # Pad (slab)
        fde_session.select("::model::pad")
        fde_session.set("y span", length)
        fde_session.set("z max", h_si)
        fde_session.set("z min", 0.0)

        # Contacts - z_min follows pad top, z_span stays from template
        fde_session.select("::model::source")
        fde_session.set("y span", length)
        fde_session.set("z min", h_si)  # Bottom aligned with pad top

        fde_session.select("::model::drain")
        fde_session.set("y span", length)
        fde_session.set("z min", h_si)  # Bottom aligned with pad top

        # Buried Oxide - z_max aligned with pad z_min
        fde_session.select("::model::buried_oxide")
        fde_session.set("y span", length)
        fde_session.set("z max", 0.0)

        # Surface Oxide - z_min aligned with waveguide z_max
        fde_session.select("::model::surface_oxide")
        fde_session.set("y span", length)
        fde_session.set("z min", 0.0)

        print(f"  -> Geometry: w_r={w_r:.2e}m, h_si={h_si:.2e}m, h_r={h_r:.2e}m")
        print(f"  -> Length: {length:.2e}m")
        print(f"  -> Contacts: z_min={h_si:.2e}m (z_span from template)")

    except Exception as e:
        print(f"  [ERROR] Failed to set FDE parameters: {e}")
        raise e


def run_fde_sweep(fde_session):
    """
    Runs the FDE voltage sweep simulation and verifies that results were generated.

    Args:
        fde_session: Lumerical FDE/MODE session object

    Returns:
        None

    Raises:
        RuntimeError: If sweep fails or produces no valid results

    Side effects:
        Executes the FDE sweep and saves results.
    """
    print("\n--- Running FDE voltage sweep ---")
    start_time = time.time()

    try:
        # Save to ensure all parameter changes are written before running
        print("  -> Saving simulation file...", end=" ", flush=True)
        fde_session.save(config.FDE_SIM_FILE)
        print("Done.")

        print("  -> Generating mesh...", end=" ", flush=True)
        sys.stdout.flush()
        fde_session.mesh()
        print("Done.")

        print("  -> Running voltage sweep (this may take several minutes)...", flush=True)
        sys.stdout.flush()
        fde_session.runsweep("voltage")

        elapsed = time.time() - start_time
        print(f"  -> FDE voltage sweep completed in {elapsed:.1f} seconds.")

        # Verify sweep succeeded by trying to read results
        try:
            test_result = fde_session.getsweepresult("voltage", "neff")
            if test_result is None:
                raise RuntimeError("FDE sweep produced no results (getsweepresult returned None)")

            # Check if result has data
            if 'neff' not in test_result:
                raise RuntimeError("FDE sweep produced incomplete results (missing neff data)")

            neff = test_result['neff']
            if neff is None or len(neff) == 0:
                raise RuntimeError("FDE sweep produced empty results (no neff data)")

            print("  -> FDE sweep results verified successfully.")

        except Exception as verify_error:
            elapsed = time.time() - start_time
            error_msg = f"FDE sweep failed verification after {elapsed:.1f}s: {verify_error}"
            print(f"  [ERROR] {error_msg}")
            raise RuntimeError(error_msg) from verify_error

    except RuntimeError:
        # Re-raise RuntimeError as-is (already formatted)
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"FDE sweep failed after {elapsed:.1f}s: {e}"
        print(f"  [ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e


def import_charge_data(fde_session, charge_file_path):
    """
    Imports CHARGE simulation data into the FDE session.

    Args:
        fde_session: Lumerical FDE/MODE session object
        charge_file_path (str): Path to the charge data file (.mat)

    Returns:
        None

    Side effects:
        Imports charge distribution data into the FDE simulation.
    """
    print("\n--- Importing CHARGE data into FDE ---")
    fde_session.switchtolayout()

    try:
        fde_session.select("::model::np")

        # Use basename for import (Lumerical expects file in CWD or full path)
        charge_filename = os.path.basename(charge_file_path)

        if os.path.exists(charge_file_path):
            fde_session.importdataset(charge_filename)
            print(f"  -> Imported {charge_filename}")
        elif os.path.exists(charge_filename):
            fde_session.importdataset(charge_filename)
            print(f"  -> Imported {charge_filename} from CWD")
        else:
            print(f"  [ERROR] Charge data file not found: {charge_file_path}")
            raise FileNotFoundError(f"Charge data file not found: {charge_file_path}")

    except Exception as e:
        print(f"  [ERROR] Failed to import charge data: {e}")
        raise e


def run_full_simulation(params, sim_id=None):
    """
    Runs a complete CHARGE + FDE simulation pipeline.

    Opens/closes Lumerical sessions internally. Returns raw data arrays.
    Raises SimulationError on failure with stage tracking.

    Args:
        params (dict): Simulation parameters (w_r, h_si, S, doping, length, lambda)
        sim_id (int, optional): Simulation ID for logging

    Returns:
        dict: {
            'V_drain': array, 'n': array, 'p': array,
            'neff': array,
            'charge_time': float, 'fde_time': float
        }

    Raises:
        SimulationError: On any simulation failure (with stage info)
    """
    if lumapi is None:
        raise SimulationError("INIT", "lumapi not available. Cannot run simulation.")

    # Show GUI if: DEBUG=True OR HIDE_GUI=False
    hide_gui = config.HIDE_GUI and not config.DEBUG
    charge_data_path = config.CHARGE_DATA_FILE

    result = {}

    # ========== CHARGE Simulation ==========
    charge = None
    try:
        try:
            charge = lumapi.DEVICE(hide=hide_gui)
            charge.load(config.CHARGE_SIM_FILE)
            print("  CHARGE simulation file loaded.")
        except Exception as e:
            raise SimulationError("CHARGE_SETUP", f"Failed to open/load CHARGE session: {e}", e)

        try:
            set_charge_parameters(charge, params, charge_data_path)
        except Exception as e:
            raise SimulationError("CHARGE_SETUP", f"Failed to set CHARGE parameters: {e}", e)

        charge_time = 0.0
        if config.RUN_SIMULATION:
            try:
                charge_start = time.time()
                run_charge_simulation(charge)
                charge_time = time.time() - charge_start
                print("  CHARGE simulation completed.")
            except Exception as e:
                raise SimulationError("CHARGE_RUN", f"CHARGE simulation run failed: {e}", e)

        try:
            raw_charge = extract_raw_charge_data(charge)
            result.update(raw_charge)
            result['charge_time'] = charge_time
            print(f"  Charge data extracted: {len(raw_charge['V_drain'])} voltage points")
        except Exception as e:
            raise SimulationError("CHARGE_EXTRACT", f"Failed to extract charge data: {e}", e)

    finally:
        if charge is not None:
            try:
                charge.close()
            except Exception:
                pass

    # ========== FDE Simulation ==========
    fde = None
    try:
        try:
            fde = lumapi.MODE(hide=hide_gui)
            fde.load(config.FDE_SIM_FILE)
            print("  FDE simulation file loaded.")
        except Exception as e:
            raise SimulationError("FDE_SETUP", f"Failed to open/load FDE session: {e}", e)

        try:
            set_fde_parameters(fde, params)
        except Exception as e:
            raise SimulationError("FDE_SETUP", f"Failed to set FDE parameters: {e}", e)

        try:
            import_charge_data(fde, charge_data_path)
            print("  CHARGE data imported into FDE.")
        except Exception as e:
            raise SimulationError("FDE_SETUP", f"Failed to import CHARGE data: {e}", e)

        fde_time = 0.0
        if config.RUN_SIMULATION:
            try:
                fde_start = time.time()
                run_fde_sweep(fde)
                fde_time = time.time() - fde_start
                print("  FDE sweep completed.")
            except Exception as e:
                raise SimulationError("FDE_RUN", f"FDE sweep run failed: {e}", e)

        try:
            raw_optical = extract_raw_optical_data(fde)
            result.update(raw_optical)
            result['fde_time'] = fde_time
            print(f"  Optical data extracted: {len(raw_optical['neff'])} points")
        except Exception as e:
            raise SimulationError("FDE_EXTRACT", f"Failed to extract optical data: {e}", e)

    finally:
        if fde is not None:
            try:
                fde.close()
            except Exception:
                pass

    # Build combined DataFrame (charge + optical on shared V axis)
    raw_df = pd.DataFrame({
        'V': result['V_drain'].flatten(),
        'n': result['n'].flatten(),
        'p': result['p'].flatten(),
        'neff_re': np.real(result['neff']).flatten(),
        'neff_im': np.imag(result['neff']).flatten()
    })

    # Save raw CSV
    run_dir = os.path.join(config.RAW_OUTPUT_DIR, f"{config.RUN_TIMESTAMP}_result")
    os.makedirs(run_dir, exist_ok=True)
    raw_csv_path = os.path.join(run_dir, f"{config.RUN_TIMESTAMP}_sim_{sim_id}.csv")
    raw_df.to_csv(raw_csv_path, index=False)

    timing = {'charge_time': result['charge_time'], 'fde_time': result['fde_time']}
    return raw_df, raw_csv_path, timing
