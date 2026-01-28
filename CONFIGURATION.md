# Configuration Guide

All configuration options are in `system/config.py`. This guide explains each setting.

---

## Git LFS (Large File Storage)

This repository uses Git LFS to store large Lumerical simulation template files. You must have Git LFS installed to properly clone and work with this repository.

### Why Git LFS?

Lumerical simulation files are large binary files:
- `PIN_Ref_paper_Charge.ldev` - CHARGE template (~347 MB)
- `PIN_Ref_phase_shifter.lms` - FDE/MODE template (~15 MB)

GitHub has a 100 MB file size limit, so these files are stored via Git LFS.

### Installation

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows
# Download installer from https://git-lfs.github.com/
```

### Setup (One-Time)

After installing, initialize Git LFS:

```bash
git lfs install
```

### Cloning the Repository

```bash
git clone https://github.com/ilaiyomsh/PS_Opt_V2.git
cd PS_Opt_V2
```

Git LFS automatically downloads the large files during clone. If you cloned before installing Git LFS, run:

```bash
git lfs pull
```

### Verifying LFS Files

Check that Lumerical files are properly downloaded (not just pointers):

```bash
ls -lh Lumerical_Files/*.ldev Lumerical_Files/*.lms
```

Expected output:
```
-rw-r--r--  347M  PIN_Ref_paper_Charge.ldev
-rw-r--r--   15M  PIN_Ref_phase_shifter.lms
```

If files show as ~130 bytes, they are LFS pointers and need to be pulled:
```bash
git lfs pull
```

### LFS-Tracked Files

The `.gitattributes` file defines which files are tracked by LFS:

```
*.ldev filter=lfs diff=lfs merge=lfs -text
*.lms filter=lfs diff=lfs merge=lfs -text
```

---

## Lumerical API Path

The most critical setting - must point to your Lumerical Python API installation.

```python
LUMERICAL_API_PATH = "C:\\Program Files\\Lumerical\\v231\\api\\python"
```

### Platform-Specific Paths

| Platform | Typical Path |
|----------|--------------|
| Windows | `C:\Program Files\Lumerical\v231\api\python` |
| Linux | `/opt/lumerical/v231/api/python` |
| macOS | `/Applications/Lumerical/v231/api/python` |

Replace `v231` with your installed version (e.g., `v202`, `v241`).

---

## Simulation Control Flags

| Flag | Default | Description |
|------|---------|-------------|
| `RUN_SIMULATION` | `True` | Run simulations (set `False` to reprocess existing data) |
| `HIDE_GUI` | `True` | Hide Lumerical GUI during runs |
| `DEBUG` | `False` | Step-by-step analysis with prompts |
| `SHOW_PLOTS` | `False` | Display plots after data extraction |
| `SKIP_LHS` | `False` | Skip LHS, use existing `params.csv` |

---

## LHS Parameters

### Sample Count

```python
LHS_N_SAMPLES = 60  # Number of initial samples
```

More samples = better initial coverage but longer runtime.

### Sampling Method

```python
LHS_SAMPLING_METHOD = 'random'  # Options: 'random', 'maximin', 'optimum'
```

| Method | Description | Speed |
|--------|-------------|-------|
| `random` | Standard LHS | Fast |
| `maximin` | Maximizes minimum distance between points | Medium |
| `optimum` | Best space-filling via optimization | Slow |

Note: `maximin` and `optimum` require the `smt` library (`pip install smt`).

### Random Seed

```python
LHS_RANDOM_SEED = None  # Set to integer for reproducibility
```

- `None`: Different samples each run
- Integer (e.g., `42`): Reproducible samples

---

## Bayesian Optimization Parameters

```python
MAX_ITERATIONS = 100  # Maximum BO iterations
BO_KAPPA = 2.0        # Exploration vs exploitation
```

### BO_KAPPA (UCB Parameter)

| Value | Behavior |
|-------|----------|
| 1.0 | Exploitation (refine known good areas) |
| 2.0 | Balanced (default) |
| 2.576 | Exploration (search new areas) |

---

## Parameter Bounds

Defined in `SWEEP_PARAMETERS` dictionary:

```python
SWEEP_PARAMETERS = {
    'w_r':     {'min': 350e-9,  'max': 500e-9,  'unit': 'm'},      # 350-500 nm
    'h_si':    {'min': 70e-9,   'max': 130e-9,  'unit': 'm'},      # 70-130 nm
    'doping':  {'min': 1e17,    'max': 1e18,    'unit': 'cm^-3'},  # 1e17-1e18 cm^-3
    'S':       {'min': 0,       'max': 0.8e-6,  'unit': 'm'},      # 0-800 nm
    'lambda':  {'min': 1260e-9, 'max': 1360e-9, 'unit': 'm'},      # 1260-1360 nm
    'length':  {'min': 0.1e-3,  'max': 1.0e-3,  'unit': 'm'}       # 0.1-1.0 mm
}
```

Note: `h_r` (rib height) is calculated as `WAFER_THICKNESS - h_si` during simulation.

---

## Cooling Delay

For systems that overheat during long simulation runs:

```python
DELAY_BETWEEN_RUNS = 0  # Seconds between simulations
```

| Machine | Recommended |
|---------|-------------|
| High-performance | `0` |
| Standard laptop | `60` - `180` |
| Low-performance | `300` - `600` |

---

## Cost Function

### Weights

Control the trade-off between objectives:

```python
FOM_WEIGHTS = {'loss': 0.3, 'vpil': 0.7}
```

- Higher `vpil` weight: Prioritize low drive voltage
- Higher `loss` weight: Prioritize low optical loss

### Targets

Normalization targets for the cost function:

```python
TARGETS = {'loss': 2.0, 'vpil': 1.0}  # dB/cm, V*mm
```

### Cost Formula

```
cost = 0.3 * (loss / 2.0)² + 0.7 * (vpil / 1.0)²
```

Lower cost = better design.

---

## Output Files

### Minimal Results (`result.csv`)

Contains 13 essential columns:

```python
MINIMAL_RESULT_COLUMNS = [
    'sim_id',
    'w_r', 'h_si', 'doping', 'S', 'lambda', 'length',
    'v_pi_V', 'v_pi_l_Vmm', 'loss_at_v_pi_dB_per_cm', 'C_at_v_pi_pF_per_cm',
    'max_dphi_rad', 'cost'
]
```

### Full Results (`result_full.csv`)

Contains all columns including:
- Timing data: `charge_time_s`, `fde_time_s`, `total_time_s`
- Validity flags: `is_valid`, `max_dphi_rad`
- Normalized metrics: `norm_loss`, `norm_vpil`
- Range data: `loss_min/max`, `C_min/max`
- Geometry: `intrinsic_width_m`

---

## Results Archiving

```python
RESULTS_ARCHIVE_DIR = os.path.join(_BASE_DIR, "results_archive")
AUTO_ARCHIVE_RESULTS = True
```

When enabled, saves timestamped copies of results for cumulative BO training.

---

## Physical Constants

These should generally not be modified:

```python
WAFER_THICKNESS = 220e-9      # SOI wafer thickness (m)
ELEMENTARY_CHARGE = 1.60e-19  # Elementary charge (C)
V_MAX = 2.5                   # Maximum simulation voltage (V)
```

---

## Recommended Settings

### Quick Test Run

Verify setup works before full optimization:

```python
LHS_N_SAMPLES = 3
MAX_ITERATIONS = 2
DELAY_BETWEEN_RUNS = 0
DEBUG = True
HIDE_GUI = False
```

### Full Optimization

Production run for best results:

```python
LHS_N_SAMPLES = 60
MAX_ITERATIONS = 100
DELAY_BETWEEN_RUNS = 0
DEBUG = False
HIDE_GUI = True
LHS_SAMPLING_METHOD = 'optimum'
```

### Low-Performance Machine

For laptops or systems prone to overheating:

```python
LHS_N_SAMPLES = 30
MAX_ITERATIONS = 50
DELAY_BETWEEN_RUNS = 300
HIDE_GUI = True
```

### Resume Previous Run

Continue optimization with existing samples:

```python
SKIP_LHS = True  # Uses existing params.csv
MAX_ITERATIONS = 50  # Additional iterations
```
