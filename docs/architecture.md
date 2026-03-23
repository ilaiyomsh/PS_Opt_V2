# System Architecture

## Module Dependency Diagram

```
+-------------------------------------------------------------+
|                          main.py                              |
|              (entry point, BO loop)                           |
+----------+----------------------+----------------------------+
           |                      |
           v                      v
+---------------------+  +--------------+
|  run_simulation.py   |  |    BO.py     |
|  Orchestration + I/O |  |  Optimizer   |
|                      |  |  (reads CSV) |
+---+----------+-------+  +--------------+
    |          |
    v          v
+----------+ +--------------+ +----------+
|sim_handler| |data_processor| | cost.py  |
|Lumerical  | | Pure math &  | | Piecewise|
|   API     | | calculations | | Quadratic|
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
| **`run_simulation.py`** | Orchestration & data I/O -- CSV read/write | No | Yes |
| **`cost.py`** | Cost function -- piecewise quadratic penalty based on max_dphi | No | No |
| **`config.py`** | Constants, paths, parameter bounds | No | No |
| **`BO.py`** | Bayesian Optimization -- reads stored cost from CSV, no re-scoring | No | Yes (reads CSV) |
| **`main.py`** | Entry point, BO loop | No | No |

---

## Function-Level Data Flow Diagram

Every box is a function. Arrows show calls and what data flows between them.

```
 +==========================================================================+
 |                            main.py::main()                               |
 |                                                                          |
 |  STAGE 1: LHS                                                           |
 |  +---------------------------+                                           |
 |  | LHS.generate_lhs_samples()|-----> params.csv                          |
 |  +---------------------------+       (w_r, h_si, doping, S, lambda, L)   |
 |                                                                          |
 |  STAGE 2: Initial Simulations                                            |
 |  +-------------------------------------+                                 |
 |  | run_simulation.run_init_file()      |                                 |
 |  |   reads params.csv                  |                                 |
 |  |   loops over each row:              |                                 |
 |  |     calls run_row() per row --------+--> (see run_row diagram below)  |
 |  +-------------------------------------+                                 |
 |                                                                          |
 |  STAGE 3: BO Loop (repeats MAX_ITERATIONS times)                         |
 |  +-------------------------------------------------------------------+  |
 |  |                                                                    |  |
 |  |   +-------------------------------+                                |  |
 |  |   | BO.train_optimizer()          |                                |  |
 |  |   |   reads result.csv           |                                |  |
 |  |   |   for each row:              |                                |  |
 |  |   |     params + (-cost) ------->|---> optimizer.register()       |  |
 |  |   |   returns: optimizer         |                                |  |
 |  |   +---------------+--------------+                                |  |
 |  |                   |                                                |  |
 |  |                   | optimizer                                      |  |
 |  |                   v                                                |  |
 |  |   +-------------------------------+                                |  |
 |  |   | BO.get_next_sample(optimizer) |                                |  |
 |  |   |   optimizer.suggest() ------->|---> next_point                 |  |
 |  |   |   np.clip to bounds          |                                |  |
 |  |   |   returns: {w_r, h_si, ...}  |                                |  |
 |  |   +---------------+--------------+                                |  |
 |  |                   |                                                |  |
 |  |                   | next_params                                    |  |
 |  |                   v                                                |  |
 |  |   +-------------------------------+                                |  |
 |  |   | run_simulation.run_row()      |---> (see run_row diagram below)|  |
 |  |   +-------------------------------+                                |  |
 |  |                                                                    |  |
 |  |   +-------------------------------+                                |  |
 |  |   | BO.get_best_result()          |                                |  |
 |  |   |   reads result.csv           |                                |  |
 |  |   |   returns: best row dict     |                                |  |
 |  |   +-------------------------------+                                |  |
 |  +-------------------------------------------------------------------+  |
 +==========================================================================+
```

---

### `run_row()` -- Single Simulation Pipeline

This is the core function. Each numbered step shows the function called,
what data goes in, and what comes out.

```
run_simulation.run_row(row, sim_id, is_last)
|
|   row = {w_r, h_si, doping, S, lambda, length}
|
|
|   STEP 1: SIMULATION (Lumerical)
|   ================================
|
|   params = {w_r, h_si, doping, S, lambda, length}
|         |
|         v
|   +--------------------------------------------------+
|   | sim_handler.run_full_simulation(params, sim_id)   |
|   |                                                   |
|   |   +-- CHARGE PHASE -------------------------+     |
|   |   |                                         |     |
|   |   |   lumapi.DEVICE() --> charge session     |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   set_charge_parameters(session, params) |     |
|   |   |     sets: w_r, h_si, S, doping, length   |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   run_charge_simulation(session)         |     |
|   |   |     session.mesh() --> session.run()     |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   extract_raw_charge_data(session)       |     |
|   |   |     session.getresult("total_charge")    |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |     {V_drain[], n[], p[]}                |     |
|   |   |                                         |     |
|   |   |   charge.close()                         |     |
|   |   +-----------------------------------------+     |
|   |                                                   |
|   |   +-- FDE PHASE ----------------------------+     |
|   |   |                                         |     |
|   |   |   lumapi.MODE() --> fde session          |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   set_fde_parameters(session, params)    |     |
|   |   |     sets: w_r, h_si, length, lambda      |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   import_charge_data(session, .mat path) |     |
|   |   |     imports charge distribution into FDE |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   run_fde_sweep(session)                 |     |
|   |   |     session.mesh()                       |     |
|   |   |     session.runsweep("voltage")          |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |   extract_raw_optical_data(session)      |     |
|   |   |     session.getsweepresult("neff")       |     |
|   |   |     np.squeeze(neff)                     |     |
|   |   |         |                                |     |
|   |   |         v                                |     |
|   |   |     {neff[]}  (complex array)            |     |
|   |   |                                         |     |
|   |   |   fde.close()                            |     |
|   |   +-----------------------------------------+     |
|   |                                                   |
|   |   returns: raw = {                                |
|   |     V_drain[],  n[],  p[],                        |
|   |     neff[],                                       |
|   |     charge_time,  fde_time                        |
|   |   }                                               |
|   +--------------------------------------------------+
|         |
|         | on SimulationError:
|         |   save_error_to_csv() --> errors.csv
|         |   return None
|
|
|   STEP 2: DATA PROCESSING (pure math)
|   =====================================
|
|   raw.V_drain, raw.n, raw.p
|         |
|         v
|   +----------------------------------------------+
|   | data_processor.process_charge_data(V, n, p)  |
|   |                                              |
|   |   Qn = e * n          (charge from carriers) |
|   |   Qp = e * p                                 |
|   |   Cn = dQn/dV         (np.gradient)          |
|   |   Cp = dQp/dV                                |
|   |   C_total = (Cn + Cp) * 1e10  --> pF/cm      |
|   |                                              |
|   |   returns: (V_cap[], C_total_pF_cm[])        |
|   +----------------------------------------------+
|
|   raw.neff, params.length, params.lambda
|         |
|         v
|   +-------------------------------------------------------+
|   | data_processor.process_optical_data(neff, L, lambda)   |
|   |                                                        |
|   |   +---------------------------+                        |
|   |   | calc_dneff(neff)          |                        |
|   |   |   d_neff = Re(neff - neff[0])                      |
|   |   +---------------------------+                        |
|   |         |  d_neff[]                                    |
|   |         v                                              |
|   |   +---------------------------+                        |
|   |   | calc_alpha(neff, lambda)  |                        |
|   |   |   k0 = 2*pi/lambda                                |
|   |   |   alpha = 2*k0*Im(neff) * (10/ln10) * 1e-2        |
|   |   +---------------------------+                        |
|   |         |  alpha_dB_per_cm[]                           |
|   |         v                                              |
|   |   +---------------------------+                        |
|   |   | calc_dphi(d_neff, L, lam) |                        |
|   |   |   d_phi = 2*pi*d_neff*L/lambda                     |
|   |   +---------------------------+                        |
|   |         |  d_phi[]                                     |
|   |         v                                              |
|   |   +-------------------------------+                    |
|   |   | calculate_v_pi(V, |d_phi|)    |                    |
|   |   |   if max(|d_phi|) >= pi:                           |
|   |   |     v_pi = np.interp(pi, |d_phi|, V)              |
|   |   |   else:                                            |
|   |   |     v_pi = NaN                                     |
|   |   +-------------------------------+                    |
|   |         |  v_pi (scalar)                               |
|   |         v                                              |
|   |   max_dphi = max(|d_phi|)                              |
|   |                                                        |
|   |   returns: (d_neff[], alpha[], d_phi[], v_pi, max_dphi)|
|   +-------------------------------------------------------+
|
|
|   STEP 3: BUILD RESULT (derived metrics + cost)
|   ===============================================
|
|   +-----------------------------------------------------+
|   | _build_result(sim_id, params, raw, V_cap, C, ...)    |
|   |                                                      |
|   |   if v_pi is not NaN (valid):                        |
|   |     loss_at_v_pi = interp(v_pi, V, alpha[])         |
|   |     C_at_v_pi    = interp(v_pi, V_cap, C[])         |
|   |     v_pi_l       = v_pi * length * 1e3   (V*mm)     |
|   |   else (failed):                                     |
|   |     loss_at_v_pi = max(alpha[])  (worst-case)        |
|   |     C_at_v_pi    = NaN                               |
|   |     v_pi_l       = V_MAX * length * 1e3  (V*mm)     |
|   |         |                                            |
|   |         v                                            |
|   |   +-------------------------------------------+      |
|   |   | cost.calculate_cost(alpha, v_pi_l, max_dphi) |      |
|   |   |                                           |      |
|   |   |  Branch on max_dphi >= π:                 |      |
|   |   |    Valid:  cost = w*(α/T)² + w*(VπL/T)²   |      |
|   |   |    Failed: cost = C_BASE + β*(π-Δφ)²      |      |
|   |   |                                           |      |
|   |   |  returns: -cost (negative for BO max)     |      |
|   |   +-------------------------------------------+      |
|   |         |                                            |
|   |         v                                            |
|   |   cost_value = -neg_cost  (positive for CSV)         |
|   |                                                      |
|   |   +-------------------------------------------+      |
|   |   | save_raw_sweep_data(sim_id, V, neff, ...) |      |
|   |   |   --> raw/sim_{id}_raw.csv                |      |
|   |   +-------------------------------------------+      |
|   |                                                      |
|   |   returns: result_dict = {                           |
|   |     sim_id, w_r, h_si, doping, S, lambda, length,   |
|   |     v_pi_V, v_pi_l_Vmm, loss_at_v_pi_dB_per_cm,    |
|   |     C_at_v_pi_pF_per_cm, max_dphi_rad,              |
|   |     is_valid, norm_loss, norm_vpil, cost,            |
|   |     charge_time_s, fde_time_s, total_time_s,         |
|   |     intrinsic_width_m, loss/C min/max ranges         |
|   |   }                                                  |
|   +-----------------------------------------------------+
|
|
|   STEP 4: SAVE TO CSV
|   ====================
|
|   result_dict
|         |
|         v
|   +---------------------------------------------------+
|   | save_single_result_to_csv(path, result_dict)      |
|   |                                                    |
|   |   +------------------------------------------+     |
|   |   | _save_to_csv(result.csv, df, MINIMAL_COLS)|    |
|   |   |   appends to result.csv (key columns only)|    |
|   |   +------------------------------------------+     |
|   |                                                    |
|   |   +------------------------------------------+     |
|   |   | _save_to_csv(result_full.csv, df, ALL)   |     |
|   |   |   appends to result_full.csv (all columns)|    |
|   |   +------------------------------------------+     |
|   +---------------------------------------------------+
|
|
|   STEP 5: COOLING DELAY
|   ======================
|
|   +-----------------------------+
|   | cooling_delay(skip=is_last) |
|   |   time.sleep(DELAY)         |
|   +-----------------------------+
|
|   returns: result_dict (or None on error)
```

---

### `LHS.generate_lhs_samples()` -- Parameter Generation

```
+----------------------------------------------+
| LHS.generate_lhs_samples()                   |
|                                              |
|   config.SWEEP_PARAMETERS                    |
|     = {w_r: {min, max}, h_si: {min, max}, ...}
|         |                                    |
|         v                                    |
|   limits = [[min1, max1], [min2, max2], ...] |
|         |                                    |
|         v                                    |
|   if method == 'random':                     |
|     scipy.qmc.LatinHypercube(d=6)            |
|     sampler.random(n=N) --> unit hypercube   |
|     qmc.scale(samples, mins, maxs)           |
|                                              |
|   if method == 'maximin' or 'optimum':       |
|     smt.LHS(xlimits=limits, criterion=...)   |
|     sampling(N)                              |
|         |                                    |
|         v                                    |
|   DataFrame: N rows x 6 params              |
|   round to 4 sig figs                        |
|   add sim_id column (1..N)                   |
|   add units row                              |
|         |                                    |
|         v                                    |
|   --> params.csv                             |
+----------------------------------------------+
```

---

### `BO.train_optimizer()` -- Training

```
+-----------------------------------------------+
| BO.train_optimizer(result_csv_path)           |
|                                               |
|   reads result.csv                            |
|         |                                     |
|         v                                     |
|   for each row in df:                         |
|     params = _normalize_params(raw_params)    |
|       --> {name: (val-min)/(max-min)} [0,1]   |
|     target = -log(row['cost'])                |
|       --> compresses 8000x range to ~12x      |
|         |                                     |
|         v                                     |
|   optimizer.register(params, target)          |
|     BayesianOptimization(                     |
|       pbounds = {name: (0, 1) for all},       |
|       acquisition = UCB(kappa=2.0)            |
|     )                                         |
|         |                                     |
|         v                                     |
|   returns: trained optimizer object           |
|   (created ONCE, reused across iterations)    |
+-----------------------------------------------+
```

### `BO.get_next_sample()` -- Suggestion

```
+--------------------------------------------+
| BO.get_next_sample(optimizer)              |
|                                            |
|   optimizer.suggest()                      |
|     GP model predicts mean + uncertainty   |
|     UCB selects point maximizing            |
|       mu + kappa * sigma                   |
|         |                                  |
|         v                                  |
|   norm_point in [0,1] space               |
|   raw_point = _denormalize_params(norm)    |
|   np.clip each param to [min, max]         |
|         |                                  |
|         v                                  |
|   returns: raw params dict                 |
+--------------------------------------------+
```

### `BO.register_result()` -- In-loop Registration

```
+--------------------------------------------+
| BO.register_result(optimizer, params, cost)|
|                                            |
|   norm_params = _normalize_params(params)  |
|   target = -log(cost)                      |
|   optimizer.register(norm_params, target)  |
+--------------------------------------------+
```

### `BO.get_best_result()` -- Best Lookup

```
+--------------------------------------------+
| BO.get_best_result(result_csv_path)        |
|                                            |
|   reads result.csv                         |
|   df.dropna(subset=['cost'])               |
|   best_idx = df['cost'].idxmin()           |
|     (lowest positive cost = best)          |
|         |                                  |
|         v                                  |
|   returns: row dict or None                |
+--------------------------------------------+
```

---

### `cost.py` -- Cost Function Internals

```
+-----------------------------------------------------------+
| cost.calculate_cost(alpha, v_pi_l, max_dphi)              |
|                                                           |
|   Piecewise quadratic penalty formula:                    |
|                                                           |
|   IF max_dphi >= π (Valid Simulation):                    |
|     norm_loss = alpha / 2.0                               |
|     norm_vpil = v_pi_l / 1.0                              |
|     cost = 0.3 * norm_loss² + 0.7 * norm_vpil²            |
|                                                           |
|   ELSE (Failed Simulation):                               |
|     cost = C_BASE + BETA * (π - max_dphi)²                |
|     where C_BASE = 35.0, BETA ≈ 31.83                     |
|                                                           |
|   return -cost  (negative for BO maximization)            |
+-----------------------------------------------------------+
```

---

### Full System Sequence (one BO iteration)

```
main.main()
  |
  |-- BO.train_optimizer()
  |     reads result.csv
  |     registers all (params, -cost) pairs into GP
  |     returns: optimizer
  |
  |-- BO.get_next_sample(optimizer)
  |     optimizer.suggest() --> {w_r, h_si, doping, S, lambda, length}
  |     returns: next_params
  |
  |-- run_simulation.run_row(next_params, sim_id)
  |     |
  |     |-- sim_handler.run_full_simulation(params)
  |     |     |-- lumapi.DEVICE() --> set_charge_parameters --> run_charge --> extract_raw_charge
  |     |     |     returns: {V_drain[], n[], p[]}
  |     |     |-- lumapi.MODE() --> set_fde_parameters --> import_charge --> run_fde_sweep --> extract_raw_optical
  |     |     |     returns: {neff[]}
  |     |     returns: {V_drain, n, p, neff, charge_time, fde_time}
  |     |
  |     |-- data_processor.process_charge_data(V_drain, n, p)
  |     |     calc: Qn, Qp, dQ/dV, C_total
  |     |     returns: (V_cap[], C_total_pF_cm[])
  |     |
  |     |-- data_processor.process_optical_data(neff, length, lambda)
  |     |     |-- calc_dneff(neff) --> d_neff[]
  |     |     |-- calc_alpha(neff, lambda) --> alpha_dB_per_cm[]
  |     |     |-- calc_dphi(d_neff, length, lambda) --> d_phi[]
  |     |     |-- calculate_v_pi(V, |d_phi|) --> v_pi
  |     |     returns: (d_neff[], alpha[], d_phi[], v_pi, max_dphi)
  |     |
  |     |-- _build_result(...)
  |     |     |-- interp(v_pi, ...) --> loss_at_v_pi, C_at_v_pi, v_pi_l
  |     |     |-- cost.calculate_cost(alpha, v_pi_l) --> neg_cost
  |     |     returns: result_dict
  |     |
  |     |-- save_single_result_to_csv(result_dict) --> result.csv, result_full.csv
  |     |-- cooling_delay()
  |     returns: result_dict
  |
  |-- BO.get_best_result() --> best row from result.csv
```

---

### CSV File Flow

```
                    params.csv
                   (LHS output)
                       |
                       | read by run_init_file()
                       v
    +------------------------------------------+
    |         run_row() x N simulations        |
    +--------+-----------+----------+----------+
             |           |          |
             v           v          v
        result.csv  result_full  raw/sim_N_raw.csv
        (minimal)    .csv         (per-sim sweep)
        (13 cols)   (all cols)    (V, neff, alpha,
             |                     d_phi, v_pi)
             |
             | read by BO.train_optimizer()
             | read by BO.get_best_result()
             v
    +------------------------------------------+
    |   BO loop: train --> suggest --> run_row  |
    +------------------------------------------+
             |
             v
        result.csv (appended)


    On error:
        save_error_to_csv() --> errors.csv
```
