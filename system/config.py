# system/config.py
# Configuration file for PS_Opt_V2 system
# Contains all global variables, paths, parameters, and constants

import os

# --- File Paths ---
# Get the base directory (PS_Opt_V2) - two levels up from this file
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lumerical API paths
LUMERICAL_API_PATH = "C:\\Program Files\\Lumerical\\v231\\api\\python"  # Lumerical Python API path
LUMAPI_PATH = 'lumapi.py'  # Lumerical API module name

# Project file paths (relative to PS_Opt_V2 directory)
LUMERICAL_FILES_DIR = os.path.join(_BASE_DIR, "Lumerical_Files")
FDE_SIM_FILE = os.path.join(LUMERICAL_FILES_DIR, "PIN_Ref_phase_shifter.lms")  # FDE simulation file path
CHARGE_DATA_FILE = os.path.join(LUMERICAL_FILES_DIR, "charge_data.mat")  # CHARGE data file path
CHARGE_SIM_FILE = os.path.join(LUMERICAL_FILES_DIR, "PIN_Ref_paper_Charge.ldev")  # CHARGE simulation file path

# CSV file paths (relative to PS_Opt_V2 directory)
SIMULATION_CSV_DIR = os.path.join(_BASE_DIR, "simulation csv")  # Directory for all CSV files and results
# Input parameters CSV file
# NOTE: The file contains a units row (row 2) that should be skipped when reading.
# Use: pd.read_csv(PARAMS_CSV_FILE, skiprows=[1]) to skip the units row
PARAMS_CSV_FILE = os.path.join(SIMULATION_CSV_DIR, "params.csv")
RESULTS_CSV_FILE = os.path.join(SIMULATION_CSV_DIR, "result.csv")  # Minimal results CSV
RESULTS_FULL_CSV_FILE = os.path.join(SIMULATION_CSV_DIR, "result_full.csv")  # Full results CSV
ERRORS_CSV_FILE = os.path.join(SIMULATION_CSV_DIR, "errors.csv")  # Errors CSV file

# Columns for minimal result file (essential data only)
MINIMAL_RESULT_COLUMNS = [
    'sim_id',
    'w_r', 'h_si', 'doping', 'S', 'lambda', 'length',  # Input params
    'v_pi_V', 'v_pi_l_Vmm', 'loss_at_v_pi_dB_per_cm', 'C_at_v_pi_pF_per_cm',  # Key outputs
    'max_dphi_rad', 'cost'  # Phase shift & BO metric
]

# Results archive
RESULTS_ARCHIVE_DIR = os.path.join(_BASE_DIR, "results_archive")
AUTO_ARCHIVE_RESULTS = True

# --- Simulation Control Flags ---
RUN_SIMULATION = True   # Run simulations (False = use existing data)
HIDE_GUI = True         # Hide Lumerical GUI
DEBUG = False           # Step-by-step analysis mode
SHOW_PLOTS = False      # Display plots after extraction
SKIP_LHS = False        # Skip LHS, use existing params.csv

# --- Cooling Delay ---
DELAY_BETWEEN_RUNS = 600  # seconds between runs (0 = no delay)

# --- LHS Parameters ---
LHS_N_SAMPLES = 5  # Number of LHS samples

LHS_SAMPLING_METHOD = 'random'  # 'random', 'maximin', or 'optimum' (smt library)
LHS_RANDOM_SEED = None          # None = random seed

# Parameter bounds (h_r = WAFER_THICKNESS - h_si)
SWEEP_PARAMETERS = {
    'w_r':     {'min': 350e-9,  'max': 500e-9,  'unit': 'm'},    # Waveguide width (350nm - 500nm)
    'h_si':    {'min': 70e-9,   'max': 130e-9,  'unit': 'm'},    # Silicon height (70nm - 130nm)
    'doping':  {'min': 1e17,    'max': 1e18,    'unit': 'cm^-3'},  # Doping concentration (reduced to 1e18 max to avoid solver error)
    'S':       {'min': 0,       'max': 0.8e-6,  'unit': 'm'},    # Junction offset (0nm - 800nm)
    'lambda':  {'min': 1260e-9, 'max': 1360e-9, 'unit': 'm'},    # Wavelength (1260nm - 1360nm)
    'length':  {'min': 0.1e-3,  'max': 1.0e-3,  'unit': 'm'}     # Device length (0.1mm - 1.0mm)
}


# --- Physical Constants ---
WAFER_THICKNESS = 220e-9  # Standard SOI wafer thickness (m)
ELEMENTARY_CHARGE = 1.60217663e-19  # Elementary charge (C)
V_MAX = 2.5  # Maximum voltage for simulations (V)

# --- Lumerical Template Fixed Values ---
DOPING_X_MIN = -5e-6  # source_nwell x_min (m)
DOPING_X_MAX = 5e-6   # drain_pwell x_max (m)

# --- Bayesian Optimization ---
MAX_ITERATIONS = 3    # BO iterations
BO_KAPPA = 2.0        # UCB kappa (low=exploit, high=explore)

# --- Cost Function (Eq. 27) ---
FOM_WEIGHTS = {'loss': 0.3, 'vpil': 0.7}  # dB/cm, V*mm
TARGETS = {'loss': 2.0, 'vpil': 1.0}      # Normalization targets
C_BASE_DEFAULT = 10.0                      # Fallback penalty base

