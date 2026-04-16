# system/config.py
# Configuration file for PS_Opt_V2 system
# Contains all global variables, paths, parameters, and constants

import os
import numpy as np

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
RAW_OUTPUT_DIR = os.path.join(SIMULATION_CSV_DIR, "raw")  # Raw per-simulation sweep data
RUN_TIMESTAMP = None  # Set at runtime by main.py

# Columns for minimal result file (essential data only)
MINIMAL_RESULT_COLUMNS = [
    'sim_id',
    'w_r', 'h_si', 'doping', 'S', 'lambda', 'length',  # Input params
    'v_pi_V', 'v_pi_l_Vmm', 'loss_at_v_pi_dB_per_cm', 'C_at_v_pi_pF_per_cm',  # Key outputs
    'max_dphi_rad', 'cost', 'kappa'  # Phase shift, BO metric & kappa
]

# --- Simulation Control Flags ---
HIDE_GUI = True         # Hide Lumerical GUI
DEBUG = False           # Step-by-step analysis mode
SHOW_PLOTS = False      # Display plots after extraction
RUN_SIMULATION = True   # Run actual Lumerical simulations (False = setup only, for testing)
SKIP_LHS = False        # Skip LHS, use existing params.csv
SKIP_INITIAL_SIMS = True   # Skip LHS + initial sims, use existing result.csv for BO

# --- Cooling Delay ---
DELAY_BETWEEN_RUNS = 180  # seconds between runs (0 = no delay)

# --- LHS Parameters ---
LHS_N_SAMPLES = 60  # Number of LHS samples

LHS_SAMPLING_METHOD = 'optimum'  # 'random', 'maximin', or 'optimum' (smt library)
LHS_RANDOM_SEED = None          # None = random seed

# Parameter bounds (h_r = WAFER_THICKNESS - h_si)
SWEEP_PARAMETERS = {
    'w_r':     {'min': 350e-9,  'max': 500e-9,  'unit': 'm'},    # Waveguide width (350nm - 500nm)
    'h_si':    {'min': 90e-9,   'max': 130e-9,  'unit': 'm'},    # Silicon height (90nm - 130nm); 70/80nm removed (>90% failure rate)
    'doping':  {'min': 1e17,    'max': 1e21,    'unit': 'cm^-3'},  # Doping concentration
    'S':       {'min': 0,       'max': 0.8e-6,  'unit': 'm'},    # Junction offset (0nm - 800nm)
    'lambda':  {'min': 1260e-9, 'max': 1360e-9, 'unit': 'm'},    # Wavelength (1260nm - 1360nm)
    'length':  {'min': 0.1e-3,  'max': 1.0e-3,  'unit': 'm'}     # Device length (0.1mm - 1.0mm)
}

# --- Discrete Parameter Configuration ---
# Restrict parameters to discrete values (e.g., foundry-specific etching depths)
# Format: parameter_name: {'enabled': bool, 'values': array (sorted, SI units), 'method': 'nearest'}
# Note: Discrete grids must be within corresponding SWEEP_PARAMETERS bounds
DISCRETE_PARAMETERS = {
    'h_si': {
        'enabled': True,  # Set to False to disable discrete snapping for h_si
        'values': np.arange(90e-9, 131e-9, 10e-9),  # [90nm, 100nm, 110nm, 120nm, 130nm] - 70/80nm removed (>90% failure rate)
        'method': 'nearest'  # Snapping method: 'nearest' (only option currently supported)
    },
    # Add more parameters here as needed (e.g., 'lambda', 'w_r', etc.)
}


# --- Physical Constants ---
WAFER_THICKNESS = 220e-9  # Standard SOI wafer thickness (m)
ELEMENTARY_CHARGE = 1.60217663e-19  # Elementary charge (C)
V_MAX = 2.5  # Maximum voltage for simulations (V)

# --- Lumerical Template Fixed Values ---
DOPING_X_MIN = -5e-6  # source_nwell x_min (m)
DOPING_X_MAX = 5e-6   # drain_pwell x_max (m)

# --- Bayesian Optimization ---
MAX_ITERATIONS = 50   # BO iterations (run_3: ~7h budget @ median sim=329s + cooling=180s)
BO_KAPPA = 2.0        # UCB kappa (low=exploit, high=explore)
BO_KAPPA_DECAY = 0.98  # Multiply kappa by this each iteration (1.0 = no decay)
# --- Cost Function (Eq. 27) ---
FOM_WEIGHTS = {'loss': 0.5, 'vpil': 0.5}  # dB/cm, V*mm (equal weighting)
TARGETS = {'loss': 20.0, 'vpil': 1.0}      # Normalization targets

# Piecewise Penalty Constants for failed phase shifts
C_BASE = 35.0  # Theoretical worst-case valid simulation cost baseline
BETA = (9.0 * C_BASE) / (np.pi**2)  # Quadratic penalty coefficient

# --- Optimizer method selector ---
BO_METHOD = 'botorch'              # 'bayes_opt' | 'botorch'

# --- BoTorch MOBO (only used when BO_METHOD='botorch') ---
BOTORCH_MC_SAMPLES = 128           # Sobol QMC sample count for qLogNEHVI
BOTORCH_NUM_RESTARTS = 10          # optimize_acqf restarts
BOTORCH_RAW_SAMPLES = 256          # optimize_acqf initial raw samples
BOTORCH_REF_MULTIPLIER = 1.0       # ref_point = this * (TARGETS['loss'], TARGETS['vpil']) -> (20, 1.0) = TARGETS; tight to push front toward origin
BOTORCH_DEVICE = 'cpu'             # 'cpu' | 'cuda' — CPU is fine at N <= 110
BOTORCH_DTYPE = 'double'           # BoTorch strongly prefers float64
BOTORCH_SEED = 42                  # matches BO.py random_state; applied to torch.manual_seed + SobolQMCNormalSampler

STRIMLIT_URL = "https://pareto-pinmz.streamlit.app/"
