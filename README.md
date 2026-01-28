# PS_Opt_V2 - PIN Diode Phase Shifter Optimization

A scientific computing system for optimizing silicon-based PIN diode phase shifters using Lumerical simulations and Bayesian optimization.

## Overview

PIN diode phase shifters are key components in silicon photonics for applications like optical switching and beam steering. This system automates the design optimization process by:

1. **Exploring the parameter space** using Latin Hypercube Sampling (LHS)
2. **Running coupled electro-optical simulations** (CHARGE + FDE) in Lumerical
3. **Optimizing designs** using Bayesian Optimization to find Pareto-optimal solutions

### Optimization Objectives

The system minimizes two competing objectives:

| Metric | Unit | Target | Description |
|--------|------|--------|-------------|
| **V_pi * L** | V*mm | 1.0 | Drive voltage-length product |
| **Optical Loss** | dB/cm | 2.0 | Insertion loss |

---

## System Architecture

```
+------------------------------------------------------------------+
|                       main.py (Entry Point)                       |
+----------------------------------+-------------------------------+
                                   |
          +------------------------+------------------------+
          v                                                 v
  +---------------+                                 +---------------+
  |    LHS.py     |                                 |     BO.py     |
  | Latin Hypercube|                                |   Bayesian    |
  |   Sampling    |                                 | Optimization  |
  +-------+-------+                                 +-------+-------+
          |                                                 |
          +------------------------+------------------------+
                                   v
                     +---------------------------+
                     |    run_simulation.py      |
                     |     (Orchestration)       |
                     +-------------+-------------+
                                   v
                     +---------------------------+
                     |     sim_handler.py        |
                     |   (Lumerical Interface)   |
                     +-------------+-------------+
                                   |
          +------------------------+------------------------+
          v                                                 v
  +---------------+                                 +---------------+
  |  CHARGE.ldev  |                                 |   MODE.lms    |
  |  (Electrical) |  ---- charge_data.mat ------>  |   (Optical)   |
  |  V -> carriers|                                 | carriers->neff|
  +---------------+                                 +---------------+
```

### Workflow

1. **LHS Phase**: Generate initial parameter samples spanning the design space
2. **CHARGE Simulation**: Calculate carrier distributions under applied voltage
3. **FDE Simulation**: Compute effective index change and optical loss from carrier data
4. **BO Phase**: Use results to train a surrogate model and propose optimal next samples

---

## Input Parameters

| Parameter | Symbol | Range | Unit | Description |
|-----------|--------|-------|------|-------------|
| Rib Width | `w_r` | 350 - 500 | nm | Waveguide rib width |
| Silicon Height | `h_si` | 70 - 130 | nm | Silicon layer height |
| Doping | `doping` | 1e17 - 1e18 | cm^-3 | Doping concentration |
| Spacing | `S` | 0 - 800 | nm | Junction offset |
| Wavelength | `lambda` | 1260 - 1360 | nm | Operating wavelength |
| Length | `length` | 0.1 - 1.0 | mm | Device length |

## Output Metrics

| Metric | Column Name | Unit | Description |
|--------|-------------|------|-------------|
| V_pi | `v_pi_V` | V | Half-wave voltage |
| V_pi * L | `v_pi_l_Vmm` | V*mm | Voltage-length product |
| Loss | `loss_at_v_pi_dB_per_cm` | dB/cm | Optical insertion loss at V_pi |
| Capacitance | `C_at_v_pi_pF_per_cm` | pF/cm | Capacitance at V_pi |
| Max Phase | `max_dphi_rad` | rad | Maximum phase shift (π = success) |
| Cost | `cost` | - | Weighted objective (lower is better) |

---

## Requirements

### Software
- Python 3.8+
- Lumerical DEVICE (CHARGE solver)
- Lumerical MODE (FDE solver)
- Git LFS (for cloning Lumerical template files)

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Git LFS

This repository uses **Git Large File Storage (LFS)** to manage large Lumerical simulation files (`.ldev`, `.lms`). These files are essential for running simulations.

#### Installing Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows
# Download from https://git-lfs.github.com/
```

#### Cloning the Repository

After installing Git LFS:

```bash
# Initialize Git LFS (one-time setup)
git lfs install

# Clone the repository (LFS files download automatically)
git clone https://github.com/ilaiyomsh/PS_Opt_V2.git
```

#### Verifying LFS Files

After cloning, verify the Lumerical files were downloaded:

```bash
ls -lh Lumerical_Files/
# Should show:
# PIN_Ref_paper_Charge.ldev  (~347 MB)
# PIN_Ref_phase_shifter.lms  (~15 MB)
```

If files show as small text pointers (~130 bytes), run:
```bash
git lfs pull
```

---

## Quick Start

### 1. Configure Lumerical Path

Edit `system/config.py` and set the path to your Lumerical installation:

```python
# Windows
LUMERICAL_API_PATH = "C:\\Program Files\\Lumerical\\v231\\api\\python"

# Linux
LUMERICAL_API_PATH = "/opt/lumerical/v231/api/python"

# macOS
LUMERICAL_API_PATH = "/Applications/Lumerical/v231/api/python"
```

### 2. Prepare Simulation Files

Ensure these files exist in `Lumerical_Files/`:
- `PIN_Ref_phase_shifter.lms` - FDE simulation template
- `PIN_Ref_paper_Charge.ldev` - CHARGE simulation template

### 3. Run Optimization

```bash
cd system
python main.py
```

---

## Project Structure

```
PS_Opt_V2/
├── system/                    # Python source code
│   ├── main.py               # Entry point
│   ├── config.py             # Configuration settings
│   ├── LHS.py                # Latin Hypercube Sampling
│   ├── BO.py                 # Bayesian Optimization
│   ├── run_simulation.py     # Simulation orchestration
│   ├── sim_handler.py        # Lumerical API interface
│   ├── data_processor.py     # Data extraction & processing
│   └── results_archive.py    # Results archiving utilities
│
├── Lumerical_Files/          # Simulation templates (tracked via Git LFS)
├── simulation csv/           # Input/output CSV files
│   ├── params.csv            # Generated parameters
│   ├── result.csv            # Minimal results (12 columns)
│   ├── result_full.csv       # Full results (all columns)
│   └── errors.csv            # Error log
│
├── results_archive/          # Historical results for BO training
├── requirements.txt          # Python dependencies
├── CONFIGURATION.md          # Detailed configuration guide
└── README.md                 # This file
```

---

## Output Files

| File | Description |
|------|-------------|
| `result.csv` | **Minimal** - Essential columns (13 cols): sim_id, inputs, key outputs, max_dphi, cost |
| `result_full.csv` | **Full** - All columns including timing, ranges, debug data |
| `errors.csv` | Error log with traceback for failed simulations |

---

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options including:
- Parameter bounds customization
- LHS and BO settings
- Cooling delay for thermal management
- Recommended settings for different scenarios

---

## Results Interpretation

After optimization completes, results are saved to `simulation csv/result.csv`:

- **Low V_pi*L + Low Loss**: Ideal designs (Pareto front)
- **Low V_pi*L + High Loss**: Fast switching, high loss
- **High V_pi*L + Low Loss**: Low loss, requires higher voltage

The Bayesian optimizer uses a weighted cost function to balance these trade-offs:
- `cost = 0.3 * (loss/2.0)² + 0.7 * (vpil/1.0)²`

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `lumapi not found` | Check `LUMERICAL_API_PATH` in `config.py` |
| `Lumerical file not found` | Ensure `.lms` and `.ldev` files exist in `Lumerical_Files/` |
| `Module not found` | Run `pip install -r requirements.txt` |
| Simulation crashes | Try increasing `DELAY_BETWEEN_RUNS` in `config.py` |
| CSV column mismatch | Delete old `result.csv` and start fresh |

---

## License

Research and development project for PIN diode phase shifter optimization.
