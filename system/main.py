# system/main.py
# Entry point: LHS sampling, initial simulations, and BO loop.

import sys
import os
import logging
from datetime import datetime
import config
import LHS
import run_simulation
import BO
import pandas as pd
import numpy as np


def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(message)s')
    for h in (logging.FileHandler(log_path, mode='w', encoding='utf-8'),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return log_path


def log(msg):
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def debug_prompt(message):
    if not config.DEBUG:
        return True
    response = input(f"[DEBUG] {message} — Enter to continue, 's' skip, 'q' quit: ").strip().lower()
    if response == 'q':
        sys.exit(0)
    return response != 's'


def main():
    log_path = setup_logging()
    config.RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()

    log(f"PS_Opt_V2 run — LHS={config.LHS_N_SAMPLES}, BO iters={config.MAX_ITERATIONS}, "
        f"skip_lhs={config.SKIP_LHS}, skip_init={config.SKIP_INITIAL_SIMS}")

    if config.SKIP_INITIAL_SIMS:
        _skip_initial(config.RESULTS_CSV_FILE)
    else:
        _run_initial(config.PARAMS_CSV_FILE, config.RESULTS_CSV_FILE)

    if not os.path.exists(config.RESULTS_CSV_FILE):
        log("[ERROR] No results file. Cannot start BO.")
        return

    _run_bo_loop(config.RESULTS_CSV_FILE)
    _print_final_results(config.RESULTS_CSV_FILE)

    log(f"Done in {datetime.now() - start_time}. Log: {log_path}")


def _skip_initial(results_path):
    if not os.path.exists(results_path):
        log(f"[ERROR] {results_path} not found. Set SKIP_INITIAL_SIMS=False to generate it.")
        return
    df = pd.read_csv(results_path)
    log(f"Using existing results: {len(df)} rows ({df['v_pi_V'].notna().sum()} valid)")


def _run_initial(params_path, results_path):
    log("Stage 1: LHS")
    if config.SKIP_LHS:
        if not os.path.exists(params_path):
            log("[ERROR] params.csv not found. Set SKIP_LHS=False to generate new samples.")
            return
    else:
        if not debug_prompt("Generate LHS samples"):
            return
        LHS.generate_lhs_samples()

    log("Stage 2: initial simulations")
    if not debug_prompt("Run initial simulations"):
        return
    run_simulation.run_init_file(params_csv_path=params_path, logger_func=log)

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        valid = df['v_pi_V'].notna().sum()
        log(f"Initial sims: {valid}/{len(df)} valid")
        if valid > 0 and 'cost' in df.columns:
            best_row = df[df['v_pi_V'].notna()].loc[lambda d: d['cost'].idxmin()]
            log(f"Best initial: sim_id={int(best_row['sim_id'])}, cost={best_row['cost']:.3f}")


def _run_bo_loop(results_path):
    log("Stage 3: BO")
    if not debug_prompt("Start BO loop"):
        return

    bo_sim_id = len(pd.read_csv(results_path)) + 1
    optimizer = BO.train_optimizer(result_csv_path=results_path)

    for iteration in range(1, config.MAX_ITERATIONS + 1):
        if not debug_prompt(f"BO iteration {iteration}"):
            continue
        try:
            next_params = BO.get_next_sample(optimizer)
            if next_params is None:
                log("[WARNING] Optimizer returned None. Stopping.")
                break

            next_params['sim_id'] = bo_sim_id
            is_last = (iteration == config.MAX_ITERATIONS)
            result = run_simulation.run_row(next_params, sim_id=bo_sim_id, is_last=is_last)

            if result is None:
                log(f"[{iteration}/{config.MAX_ITERATIONS}] sim_id={bo_sim_id} FAILED")
                bo_sim_id += 1
                continue

            result['kappa'] = BO.get_current_kappa(optimizer)
            BO.register_result(optimizer, result, result['cost'])
            _log_bo_result(iteration, bo_sim_id, result)
            bo_sim_id += 1

        except KeyboardInterrupt:
            log("Interrupted by user.")
            break
        except Exception as e:
            log(f"[ERROR] iteration {iteration}: {e}")
            bo_sim_id += 1


def _log_bo_result(iteration, sim_id, result):
    v_pi = result.get('v_pi_V', float('nan'))
    prefix = f"[{iteration}/{config.MAX_ITERATIONS}] sim_id={sim_id}"
    if not np.isnan(v_pi):
        log(f"{prefix} V_pi={v_pi:.2f}V, V_pi*L={result.get('v_pi_l_Vmm', 0):.3f}, "
            f"loss={result.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm, cost={result.get('cost', 0):.3f}")
    else:
        log(f"{prefix} INVALID max_dphi={result.get('max_dphi_rad', 0):.2f}, cost={result.get('cost', 0):.3f}")


def _print_final_results(results_path):
    log("Stage 4: final results")
    if not os.path.exists(results_path):
        return
    df = pd.read_csv(results_path)
    valid = df['v_pi_V'].notna().sum()
    log(f"Total: {len(df)}, valid: {valid}, invalid: {len(df) - valid}")

    best = BO.get_best_result(results_path)
    if not best:
        return
    log(f"Best: sim_id={int(best.get('sim_id', 0))}, "
        f"V_pi={best.get('v_pi_V', 0):.2f}V, V_pi*L={best.get('v_pi_l_Vmm', 0):.3f} V*mm, "
        f"loss={best.get('loss_at_v_pi_dB_per_cm', 0):.2f} dB/cm")


if __name__ == "__main__":
    main()
