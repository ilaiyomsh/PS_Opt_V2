# System Architecture

## Module Dependency Diagram

```
+-------------------------------------------------------------+
|                          main.py                              |
|          (entry point, BO loop, C_BASE bootstrap)             |
+----------+----------------------+----------------------------+
           |                      |
           v                      v
+---------------------+  +--------------+
|  run_simulation.py   |  |    BO.py     |
|  Orchestration + I/O |  |  Optimizer   |
|  C_BASE maintenance  |  |  (reads CSV) |
+---+----------+-------+  +--------------+
    |          |
    v          v
+----------+ +--------------+ +----------+
|sim_handler| |data_processor| | cost.py  |
|Lumerical  | | Pure math &  | | Cost fn  |
|   API     | | calculations | | & C_BASE |
+----------+ +--------------+ +----------+
    |
    v
 [lumapi]
```

## Module Responsibilities

| Module | Responsibility | Imports lumapi? | Does file I/O? |
|--------|---------------|:---:|:---:|
| **`sim_handler.py`** | Lumerical API -- the ONLY module that touches lumapi sessions | Yes | No |
| **`data_processor.py`** | Processing & calculations -- pure math, plotting | No | No |
| **`run_simulation.py`** | Orchestration & data I/O -- CSV read/write, C_BASE maintenance | No | Yes |
| **`cost.py`** | Cost function & C_BASE state management (Eq. 27) | No | No |
| **`config.py`** | Constants, paths, parameter bounds | No | No |
| **`BO.py`** | Bayesian Optimization -- reads stored cost from CSV, no re-scoring | No | Yes (reads CSV) |
| **`main.py`** | Entry point, BO loop, C_BASE bootstrap at startup | No | No |

## Data Flow

```
run_row(params)
|
+- 1. Simulation (one call)
|     |
|     v
|   sim_handler.run_full_simulation(params)
|     |
|     +- CHARGE: open session -> set params -> run -> extract raw data -> close
|     |          returns: V_drain[], n[], p[]
|     |
|     +- FDE: open session -> set params -> import charge -> run sweep -> extract raw data -> close
|             returns: neff[]
|     |
|     +- Returns: { V_drain, n, p, neff, charge_time, fde_time }
|
+- 2. Processing (two calls)
|     |
|     +- data_processor.process_charge_data(V_drain, n, p)
|     |   Returns: (V_cap, C_total_pF_cm)
|     |
|     +- data_processor.process_optical_data(neff, length, wavelength)
|         Returns: (d_neff, alpha_dB_per_cm, d_phi, v_pi, max_dphi)
|
+- 3. Cost calculation + C_BASE check
|     |
|     +- cost.calculate_cost(alpha, v_pi_l, max_dphi)
|     |   Returns: negative cost for BayesOpt
|     |
|     +- If valid run & cost > C_BASE:
|           cost.update_c_base_if_needed(positive_cost)
|           _re_score_failed_rows()
|
+- 4. Save results (CSV I/O)
|     +- save_raw_sweep_data(...)
|     +- save_single_result_to_csv(...)
|
+- 5. Cooling delay
```

## Module Contents

### `sim_handler.py` -- Lumerical API

The only module that imports `lumapi`. Handles all session lifecycle and raw data extraction.

- `SimulationError(Exception)` -- Custom exception with `stage`, `message`, `original_error`
- `set_charge_parameters(session, params, charge_data_path)` -- Configure CHARGE simulation
- `run_charge_simulation(session)` -- Execute CHARGE simulation
- `extract_raw_charge_data(session)` -- Extract V_drain, n, p arrays via `getresult()`
- `set_fde_parameters(session, params)` -- Configure FDE simulation
- `run_fde_sweep(session)` -- Execute FDE voltage sweep
- `import_charge_data(session, charge_data_path)` -- Import CHARGE results into FDE
- `extract_raw_optical_data(session)` -- Extract neff array via `getsweepresult()`
- `run_full_simulation(params, sim_id)` -- Full CHARGE+FDE pipeline with session lifecycle

### `data_processor.py` -- Pure Math

No lumapi calls, no file I/O. Takes raw arrays, returns processed results.

- `process_charge_data(V_drain, n, p)` -- Raw carrier counts to capacitance (pF/cm)
- `process_optical_data(neff, length, wavelength)` -- Raw neff to optical parameters
- `calc_alpha(neff, wavelength)` -- Optical loss in dB/cm
- `calc_dneff(neff)` -- Effective index change relative to V=0
- `calc_dphi(d_neff, length, wavelength)` -- Phase shift in radians
- `calculate_v_pi(voltages, abs_dphi)` -- V_pi by interpolation
- `plot_capacitance(V, C_total_pF_cm)` -- Capacitance plot
- `plot_optical_results(V, d_neff, alpha, d_phi, v_pi)` -- Optical results plots

### `run_simulation.py` -- Orchestration & I/O

Coordinates simulation, processing, cost calculation, and CSV persistence.

- `run_row(row, sim_id, is_last)` -- Run one simulation end-to-end
- `run_init_file(init_csv_path, result_csv_path)` -- Run all rows from params CSV
- `save_single_result_to_csv(csv_path, result_dict)` -- Append result to CSV
- `save_error_to_csv(sim_id, stage, error, params)` -- Log error to CSV
- `save_raw_sweep_data(sim_id, ...)` -- Save raw sweep arrays
- `_re_score_failed_rows()` -- Re-score failed rows when C_BASE changes
- `cooling_delay()` -- Pause between simulations
- `debug_prompt(message)` -- Interactive debug pause

### `cost.py` -- Cost Function & C_BASE

Implements Eq. 27 cost function with dynamic C_BASE penalty for failed simulations.

- `calculate_cost(alpha, v_pi_l, max_dphi, weights, targets)` -- Compute cost value
- `update_c_base(new_value)` -- Set C_BASE to new value
- `get_c_base()` -- Get current C_BASE value
- `calculate_c_base_from_results(df)` -- Bootstrap C_BASE from results DataFrame
- `update_c_base_if_needed(new_valid_cost)` -- Update C_BASE only if new cost exceeds current

### `BO.py` -- Bayesian Optimization

Reads stored cost from CSV. No re-scoring or recalculation of C_BASE.

- `train_optimizer(result_csv_path)` -- Train GP model from stored results
- `get_next_sample(optimizer, result_csv_path)` -- Suggest next parameter set via UCB
- `get_best_result(result_csv_path)` -- Find best result from CSV

### `main.py` -- Entry Point

1. Generate LHS samples (or skip if `SKIP_LHS=True`)
2. Run initial simulations from params.csv
3. Bootstrap C_BASE from all results
4. BO loop: train optimizer, suggest next point, simulate, repeat

### `config.py` -- Configuration

All constants, file paths, parameter bounds, and optimization settings.

## C_BASE Update Logic

```
Startup (main.py):
  cost.calculate_c_base_from_results(all_results)   <- bootstrap once

Each simulation (run_simulation.py -> run_row):
  cost_val = cost.calculate_cost(alpha, v_pi_l, max_dphi)
  save to CSV
  if valid run (v_pi is not NaN):
    if cost.update_c_base_if_needed(valid_cost):
      _re_score_failed_rows()                        <- C_BASE changed

BO training (BO.py -> train_optimizer):
  reads stored 'cost' column from CSV               <- no re-scoring
```
