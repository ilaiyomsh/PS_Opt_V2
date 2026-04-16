# system/BO_botorch.py
# Multi-objective BO backend. Public surface mirrors BO.py so bo_dispatch.py
# can swap backends. Torch/botorch are imported lazily on first call.

import os
from dataclasses import dataclass, field
from typing import Any

import config
import numpy as np
import pandas as pd


def _normalize_params(params):
    return {name: (params[name] - cfg['min']) / (cfg['max'] - cfg['min'])
            for name, cfg in config.SWEEP_PARAMETERS.items()}


def _denormalize_params(norm_params):
    return {name: norm_params[name] * (cfg['max'] - cfg['min']) + cfg['min']
            for name, cfg in config.SWEEP_PARAMETERS.items()}


def _param_order():
    return list(config.SWEEP_PARAMETERS.keys())


_torch = None
_bt = None           # botorch namespace bag
_dtype = None
_device = None


def _load_torch():
    global _torch, _bt, _dtype, _device
    if _torch is not None:
        return

    try:
        import torch
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement as qLogNEHVI,
        )
        from botorch.acquisition.multi_objective.objective import (
            IdentityMCMultiOutputObjective,
        )
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls import SumMarginalLogLikelihood
    except ImportError as e:
        # Raised here (not in bo_dispatch) because these are lazy imports.
        raise ImportError(
            "BO_METHOD='botorch' requires torch, gpytorch, and botorch. "
            "Install with: pip install torch gpytorch botorch "
            "(requires Python >= 3.11)."
        ) from e

    torch.manual_seed(config.BOTORCH_SEED)

    class _BT:
        pass
    bt = _BT()
    bt.SingleTaskGP = SingleTaskGP
    bt.ModelListGP = ModelListGP
    bt.Standardize = Standardize
    bt.fit_gpytorch_mll = fit_gpytorch_mll
    bt.qLogNEHVI = qLogNEHVI
    bt.IdentityMCMultiOutputObjective = IdentityMCMultiOutputObjective
    bt.optimize_acqf = optimize_acqf
    bt.SobolQMCNormalSampler = SobolQMCNormalSampler
    bt.SumMarginalLogLikelihood = SumMarginalLogLikelihood

    _torch = torch
    _bt = bt
    _dtype = torch.double if config.BOTORCH_DTYPE == 'double' else torch.float
    _device = torch.device(config.BOTORCH_DEVICE)


def _t(data):
    return _torch.tensor(data, dtype=_dtype, device=_device)


@dataclass
class _State:
    train_X_obj: Any
    train_Y: Any
    train_X_con: Any
    train_C: Any
    model: Any
    ref_point: Any
    bounds: Any
    param_names: list = field(default_factory=_param_order)


def _fit_model(X_obj, Y, X_con, C):
    gp_loss = _bt.SingleTaskGP(X_obj, Y[:, 0:1], outcome_transform=_bt.Standardize(m=1))
    gp_vpil = _bt.SingleTaskGP(X_obj, Y[:, 1:2], outcome_transform=_bt.Standardize(m=1))
    gp_con = _bt.SingleTaskGP(X_con, C, outcome_transform=_bt.Standardize(m=1))
    model = _bt.ModelListGP(gp_loss, gp_vpil, gp_con)
    mll = _bt.SumMarginalLogLikelihood(model.likelihood, model)
    _bt.fit_gpytorch_mll(mll)
    return model


def _hypervolume(Y, ref_point):
    try:
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.pareto import is_non_dominated
        if Y.shape[0] == 0:
            return 0.0
        pareto_Y = Y[is_non_dominated(Y)]
        return float(Hypervolume(ref_point=ref_point).compute(pareto_Y))
    except Exception:
        return 0.0


def train_optimizer(result_csv_path=None):
    _load_torch()

    if result_csv_path is None:
        result_csv_path = config.RESULTS_CSV_FILE
    if not os.path.exists(result_csv_path):
        raise FileNotFoundError(f"Results file not found: {result_csv_path}")

    df = pd.read_csv(result_csv_path)
    if len(df) == 0:
        raise ValueError("Results file is empty. Run initial simulations first.")

    print(f"  -> Loaded {len(df)} prior data points from {result_csv_path}")

    param_names = _param_order()
    required = param_names + ['max_dphi_rad', 'loss_at_v_pi_dB_per_cm', 'v_pi_l_Vmm']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in result.csv: {col}")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows with complete data in results file.")

    X_con_rows, C_rows, X_obj_rows, Y_rows = [], [], [], []
    for _, row in df.iterrows():
        raw = {name: float(row[name]) for name in param_names}
        x_vec = [_normalize_params(raw)[name] for name in param_names]
        max_dphi = float(row['max_dphi_rad'])

        X_con_rows.append(x_vec)
        C_rows.append([max_dphi - np.pi])

        # Infeasible rows have placeholder v_pi_l (V_MAX * L) — keep them out
        # of the objective GPs but in the constraint GP.
        if max_dphi >= np.pi:
            X_obj_rows.append(x_vec)
            Y_rows.append([-float(row['loss_at_v_pi_dB_per_cm']),
                           -float(row['v_pi_l_Vmm'])])

    if len(X_obj_rows) == 0:
        raise ValueError(
            "No feasible (max_dphi_rad >= pi) rows found. "
            "BoTorch MOBO needs at least one feasible point; run more LHS "
            "samples or widen SWEEP_PARAMETERS."
        )

    train_X_con = _t(X_con_rows)
    train_C = _t(C_rows)
    train_X_obj = _t(X_obj_rows)
    train_Y = _t(Y_rows)

    print(f"  -> Feasible rows (objective GPs): {train_X_obj.shape[0]}")
    print(f"  -> All rows (constraint GP):      {train_X_con.shape[0]}")

    model = _fit_model(train_X_obj, train_Y, train_X_con, train_C)
    mult = config.BOTORCH_REF_MULTIPLIER
    ref_point = _t([-mult * config.TARGETS['loss'], -mult * config.TARGETS['vpil']])
    bounds = _torch.stack([_t([0]*6), _t([1]*6)])

    hv = _hypervolume(train_Y, ref_point)
    if hv <= 0.0:
        print(f"  [WARNING] Initial hypervolume is 0.0 (no feasible point "
              f"dominates ref_point = {ref_point.tolist()}). qLogNEHVI gradient "
              f"may be weak. Consider raising config.BOTORCH_REF_MULTIPLIER.")
    else:
        print(f"  -> Initial feasible hypervolume: {hv:.4f}")

    return _State(train_X_obj, train_Y, train_X_con, train_C,
                  model, ref_point, bounds)


def get_next_sample(state):
    if state is None:
        raise ValueError("State is None. Call train_optimizer first.")
    _load_torch()

    try:
        sampler = _bt.SobolQMCNormalSampler(
            sample_shape=_torch.Size([config.BOTORCH_MC_SAMPLES]),
            seed=config.BOTORCH_SEED,
        )
        # Constraint convention: BoTorch expects callable(Z) <= 0 for feasible.
        # Our slack is (max_dphi - pi) >= 0 when feasible, so sign-flip it.
        acq = _bt.qLogNEHVI(
            model=state.model,
            ref_point=state.ref_point,
            X_baseline=state.train_X_con,
            sampler=sampler,
            objective=_bt.IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            constraints=[lambda Z: -(Z[..., 2])],
            prune_baseline=True,
        )
        candidate, _ = _bt.optimize_acqf(
            acq_function=acq,
            bounds=state.bounds,
            q=1,
            num_restarts=config.BOTORCH_NUM_RESTARTS,
            raw_samples=config.BOTORCH_RAW_SAMPLES,
        )

        norm_vec = candidate.detach().cpu().numpy().ravel().tolist()
        raw_point = _denormalize_params(dict(zip(state.param_names, norm_vec)))
        for name, cfg in config.SWEEP_PARAMETERS.items():
            raw_point[name] = float(np.clip(raw_point[name], cfg['min'], cfg['max']))

        print(f"  -> Next suggested point: {raw_point}")
        return raw_point

    except Exception as e:
        print(f"  [ERROR] Failed to get next sample: {e}")
        return None


def register_result(state, params, cost_value):
    # cost_value accepted for BO.py signature parity; MOBO reads raw objectives.
    if state is None:
        raise ValueError("State is None. Call train_optimizer first.")
    _load_torch()

    for key in ('loss_at_v_pi_dB_per_cm', 'v_pi_l_Vmm', 'max_dphi_rad'):
        if key not in params:
            print(f"  [WARNING] register_result: missing '{key}' — skipping.")
            return

    raw = {name: float(params[name]) for name in state.param_names}
    x_vec = [_normalize_params(raw)[name] for name in state.param_names]
    x_tensor = _t([x_vec])

    max_dphi = float(params['max_dphi_rad'])
    state.train_X_con = _torch.cat([state.train_X_con, x_tensor], dim=0)
    state.train_C = _torch.cat([state.train_C, _t([[max_dphi - np.pi]])], dim=0)

    if max_dphi >= np.pi:
        y_tensor = _t([[-float(params['loss_at_v_pi_dB_per_cm']),
                        -float(params['v_pi_l_Vmm'])]])
        state.train_X_obj = _torch.cat([state.train_X_obj, x_tensor], dim=0)
        state.train_Y = _torch.cat([state.train_Y, y_tensor], dim=0)

    state.model = _fit_model(state.train_X_obj, state.train_Y,
                             state.train_X_con, state.train_C)


def get_current_kappa(state):
    return None


def get_best_result(result_csv_path=None):
    print("  [INFO] MOBO run — see pareto/ dashboard for the Pareto front.")
    return None
