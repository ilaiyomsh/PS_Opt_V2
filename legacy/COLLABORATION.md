# Collaboration Guide

This guide explains how multiple collaborators can work together on PIN diode phase shifter optimization, sharing simulation results to accelerate Bayesian Optimization learning.

---

## How Collaborative BO Works

```
You run 20 sims → archive has 20 results → push
                                              ↓
Collaborator pulls → BO sees 20 points → runs 20 more → push
                                                          ↓
You pull → BO sees 40 points → runs 20 more → push
                                                ↓
                        ... cumulative learning ...
```

The system automatically:
- Loads **all** archived results for BO training
- Skips duplicate samples (within 1% tolerance)
- Generates unique sim_ids across all runs

---

## Initial Setup (One Time)

### 1. Clone the Repository

```bash
git clone https://github.com/ilaiyomsh/PS_Opt_V2.git
cd PS_Opt_V2
```

### 2. Download Lumerical Template Files

The Lumerical simulation files are too large for regular Git. Download them from the GitHub Release:

1. Go to: https://github.com/ilaiyomsh/PS_Opt_V2/releases
2. Download from the latest release:
   - `PIN_Ref_paper_Charge.ldev` (347 MB)
   - `PIN_Ref_phase_shifter.lms` (15 MB)
3. Place them in the `Lumerical_Files/` directory

Verify the files are correct size:
```bash
ls -lh Lumerical_Files/
# Should show ~347 MB for .ldev and ~15 MB for .lms
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Lumerical API Path

Edit `system/config.py` and set the path to your Lumerical installation:

```python
# Windows
LUMERICAL_API_PATH = "C:\\Program Files\\Lumerical\\v231\\api\\python"

# Linux
LUMERICAL_API_PATH = "/opt/lumerical/v231/api/python"

# macOS
LUMERICAL_API_PATH = "/Applications/Lumerical/v231/api/python"
```

Replace `v231` with your installed version.

---

## Running Simulations

### 1. ALWAYS Pull Before Running

```bash
git pull origin main
```

This ensures BO trains on **all** archived results from all collaborators.

### 2. Configure Run Parameters

Edit `system/config.py`:

```python
LHS_N_SAMPLES = 10      # New LHS samples to explore
MAX_ITERATIONS = 20     # Bayesian Optimization iterations
DELAY_BETWEEN_RUNS = 0  # Seconds between runs (increase if overheating)
```

### 3. Run the Optimization

```bash
cd system
python main.py
```

The system will:
1. Generate LHS samples (skipping duplicates of archived results)
2. Run CHARGE + FDE simulations for each sample
3. Train BO on all results (current + archived)
4. Run BO iterations (skipping duplicates)
5. Archive results automatically

---

## After Run Completes - Share Results

### 1. Pull Any New Results

```bash
git pull origin main
```

### 2. Commit and Push Your Results

```bash
git add results_archive/
git commit -m "Add N simulation results from [your name]"
git push origin main
```

---

## Important Rules

| Do | Don't |
|----|-------|
| `git pull` before every run | Run without pulling first |
| Commit `results_archive/` folder | Commit `simulation csv/` folder |
| Share results after each run | Keep results only on your machine |
| Use unique descriptive commit messages | Commit Lumerical binary files |

---

## File Structure

```
PS_Opt_V2/
├── results_archive/           # SHARED - commit this folder
│   ├── result_YYYYMMDD_HHMMSS.csv    # Archived results
│   └── ...
│
├── simulation csv/            # LOCAL ONLY - do not commit
│   ├── params.csv            # Current run parameters
│   ├── result.csv            # Current run results
│   └── errors.csv            # Current run errors
│
├── Lumerical_Files/          # Download from Release
│   ├── PIN_Ref_paper_Charge.ldev
│   └── PIN_Ref_phase_shifter.lms
│
└── system/                   # Source code
    ├── config.py            # Configuration (edit this)
    ├── main.py              # Entry point
    └── ...
```

---

## Duplicate Detection

The system prevents wasting time on similar simulations:

- **Tolerance:** 1% relative difference (configurable via `DUPLICATE_TOLERANCE` in config.py)
- **Scope:** Checks against ALL archived results (current + all collaborators)
- **Action:** Skips simulation and tries a different point

If you see `[SKIP] Parameters are too similar to an existing result`, this is working correctly.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| BO suggests same point repeatedly | Run `git pull` to get latest archived results |
| "Training GP on X data points" shows low number | Ensure `results_archive/` has CSV files |
| Merge conflicts in results_archive | Use `git pull --no-rebase` then push |
| Lumerical files show as ~130 bytes | Download from Release page, not via Git |

---

## Viewing All Results

To see summary of all archived results:

```python
cd system
python -c "import results_archive; results_archive.list_archives()"
```

---

## Best Practices

1. **Communicate** with collaborators about which parameter regions you're exploring
2. **Pull frequently** to benefit from others' results
3. **Push promptly** after runs complete so others can use your data
4. **Use descriptive commits** like "Add 30 results exploring high doping region"
5. **Check the log** for "Training GP on X data points" to verify archive is loaded
