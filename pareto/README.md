# Pareto Front Analysis & Interactive Dashboard

Self-contained module for multi-objective analysis of PIN diode phase shifter optimization results.  
Deployed on **Streamlit Community Cloud** — no local installation required for viewers.

## Live Dashboard

**[Open Dashboard](https://ilaiyomsh-ps-opt-v2-pareto-dashboard.streamlit.app)**  
*(replace with your actual Streamlit Cloud URL)*

## Files

| File | Description |
|------|-------------|
| `dashboard.py` | Streamlit interactive dashboard |
| `pareto_front.py` | CLI tool — Pareto identification & CSV/HTML export |
| `result_1.csv` | Run 1 results snapshot (60 LHS + 144 BO, with kappa decay) |
| `result_2.csv` | Run 2 results snapshot (second optimization run) |
| `pareto_front.csv` | Exported Pareto-optimal subset (CLI output, Run 1) |
| `pareto_front.html` | Static Plotly HTML plot (CLI output, Run 1) |
| `requirements.txt` | Python dependencies for Streamlit Cloud |

## Running Locally

```bash
cd pareto
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at `http://localhost:8501`.

## Running the CLI Tool

```bash
cd pareto
python pareto_front.py
```

Outputs `pareto_front.csv` and `pareto_front.html`.

## Dashboard Features

### Pareto Front Plot
- Scatter plot of V_π·L vs. optical loss for all valid simulations
- Pareto-optimal points highlighted with color-coded cost and connected by a dotted line
- Non-Pareto points shown in gray (toggleable)
- **Knee point** (best trade-off) detected automatically and marked with a red star
- Hover tooltip shows all design parameters in human-readable units (nm, mm, V)

### Run Selection (sidebar)
- Choose **Run 1**, **Run 2**, or **Both** to compare optimization campaigns
- In "Both" mode, each run is tagged on the plot with a distinct marker symbol (circle / diamond) while sharing a single cost color scale — Pareto front is computed across the combined dataset
- Cost-evolution chart splits into per-run traces (each with its own running-best line)
- Tables show a **Run** column for origin tracking

### Data Filters (sidebar)
- **Only sims that reached π** — exclude simulations that didn't achieve π phase shift
- **Max Loss cutoff** — numeric input to set the maximum loss threshold (default: 50 dB/cm)
- **Loss / V_π·L range sliders** — zoom into a region of the Pareto front

### Dynamic Cost Function (sidebar)
The cost function can be tuned in real-time:

$$C = w_\alpha \left(\frac{\alpha}{T_\alpha}\right)^2 + w_{V_\pi L}\left(\frac{V_\pi L}{T_{V_\pi L}}\right)^2$$

Adjustable parameters:
- `w_loss` / `w_vpil` — relative weights (default: 0.3 / 0.7)
- `T_loss` / `T_vpil` — normalization targets (default: 20 dB/cm / 1.0 V·mm)

Changing these recalculates cost for all simulations instantly, updating the plot colors and table.

### Cost Evolution Chart
- Cost per simulation vs. simulation ID
- Running-best line shows optimization progress over time
- Filterable by sim ID range and cost range

### Results Table
Three viewing modes:
- **Pareto-optimal only** — just the non-dominated designs
- **All valid (filtered)** — all simulations that pass the data filters
- **All results (raw)** — every row in result.csv, including failed simulations

All views support sorting by any column and CSV download.

## Updating Results

When new simulation results are available, copy them into `pareto/` with a numbered name and register the file in `pareto_front.py`:

```bash
# 1. Copy the new run's CSV
cp "simulation csv/result.csv" pareto/result_3.csv

# 2. Add an entry to RUN_FILES in pareto/pareto_front.py:
#    RUN_FILES = {
#        "Run 1": os.path.join(_PARETO_DIR, "result_1.csv"),
#        "Run 2": os.path.join(_PARETO_DIR, "result_2.csv"),
#        "Run 3": os.path.join(_PARETO_DIR, "result_3.csv"),  # <- new
#    }

# 3. (Optional) Add a color/symbol for the new run in dashboard.py
#    RUN_COLORS / RUN_SYMBOLS

# 4. Commit and push
git add pareto/
git commit -m "Add Run 3 results"
git push
```

Streamlit Cloud detects the push and redeploys automatically (~30 seconds).

## How Pareto Identification Works

The dashboard uses [Optuna](https://optuna.org/) for multi-objective optimization analysis.  
Each simulation is injected as a trial with two objectives to minimize: `V_π·L` and `loss`.  
Optuna computes the non-dominated set — designs where no other design is strictly better in both objectives simultaneously.

The **knee point** is found geometrically: both axes are normalized to [0,1], a line is drawn between the two extreme endpoints of the Pareto front, and the point with the maximum perpendicular distance from that line is identified as the best compromise.
