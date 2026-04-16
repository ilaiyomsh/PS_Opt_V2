# test/unit/test_bo_botorch.py
# Unit tests for BO_botorch.py - BoTorch multi-objective backend.
#
# All tests are gated on `pytest.importorskip('botorch')` so environments
# without torch/botorch skip cleanly rather than failing the suite.

import os
import sys
import importlib
import pytest

# Make the system/ directory importable.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'system'),
)

botorch = pytest.importorskip('botorch')


@pytest.fixture
def results_csv():
    """Path to the real result.csv in the repo. Skip test if missing."""
    import config
    if not os.path.exists(config.RESULTS_CSV_FILE):
        pytest.skip(f"result.csv not found at {config.RESULTS_CSV_FILE}")
    return config.RESULTS_CSV_FILE


@pytest.mark.unit
def test_train_optimizer_on_real_csv(results_csv):
    """train_optimizer should load real results and return a populated _State."""
    import BO_botorch
    state = BO_botorch.train_optimizer(results_csv)

    assert state is not None
    assert state.train_X_con.shape[0] > 0, "constraint tensor must have rows"
    assert state.train_X_con.shape[1] == 6, "6 params in SWEEP_PARAMETERS"
    assert state.train_C.shape == (state.train_X_con.shape[0], 1)
    assert state.train_X_obj.shape[1] == 6
    assert state.train_Y.shape == (state.train_X_obj.shape[0], 2)


@pytest.mark.unit
def test_get_next_sample_returns_valid_dict(results_csv):
    """Suggested point must contain every SWEEP_PARAMETERS key within bounds."""
    import config
    import BO_botorch

    state = BO_botorch.train_optimizer(results_csv)
    next_point = BO_botorch.get_next_sample(state)

    assert next_point is not None
    for name, cfg in config.SWEEP_PARAMETERS.items():
        assert name in next_point, f"missing param {name}"
        assert cfg['min'] <= next_point[name] <= cfg['max'], (
            f"{name}={next_point[name]} out of bounds [{cfg['min']}, {cfg['max']}]"
        )


@pytest.mark.unit
def test_register_result_refits(results_csv):
    """register_result must extend tensors and leave get_next_sample working."""
    import pandas as pd
    import BO_botorch

    state = BO_botorch.train_optimizer(results_csv)
    n_con_before = state.train_X_con.shape[0]
    n_obj_before = state.train_X_obj.shape[0]

    # Synthesize a result dict from the first CSV row.
    df = pd.read_csv(results_csv).dropna(
        subset=['w_r', 'h_si', 'doping', 'S', 'lambda', 'length',
                'max_dphi_rad', 'loss_at_v_pi_dB_per_cm', 'v_pi_l_Vmm']
    )
    row = df.iloc[0].to_dict()
    row['sim_id'] = int(row.get('sim_id', 1))

    BO_botorch.register_result(state, row, cost_value=1.0)

    assert state.train_X_con.shape[0] == n_con_before + 1
    # Objective set grows only if the synthetic row is feasible
    feasible = row['max_dphi_rad'] >= 3.14159
    assert state.train_X_obj.shape[0] == n_obj_before + (1 if feasible else 0)

    # Model must still produce suggestions
    pt = BO_botorch.get_next_sample(state)
    assert pt is not None


@pytest.mark.unit
def test_get_current_kappa_is_none(results_csv):
    """get_current_kappa returns None under MOBO — interface parity only."""
    import BO_botorch
    state = BO_botorch.train_optimizer(results_csv)
    assert BO_botorch.get_current_kappa(state) is None


@pytest.mark.unit
def test_get_best_result_returns_none():
    """Under MOBO, get_best_result must return None (Pareto front has no single best)."""
    import BO_botorch
    assert BO_botorch.get_best_result() is None


@pytest.mark.unit
def test_interface_parity_bayes_opt():
    """bo_dispatch under 'bayes_opt' should expose the same five public names."""
    import config
    original = config.BO_METHOD
    config.BO_METHOD = 'bayes_opt'
    try:
        # Force a fresh import so the dispatcher picks up the current flag.
        sys.modules.pop('bo_dispatch', None)
        import bo_dispatch
        for name in ('train_optimizer', 'get_next_sample', 'register_result',
                     'get_current_kappa', 'get_best_result'):
            assert hasattr(bo_dispatch, name), f"bo_dispatch missing {name}"
    finally:
        config.BO_METHOD = original
        sys.modules.pop('bo_dispatch', None)


@pytest.mark.unit
def test_lazy_import_under_bayes_opt():
    """Importing bo_dispatch under 'bayes_opt' must NOT pull in torch."""
    import config
    original = config.BO_METHOD
    config.BO_METHOD = 'bayes_opt'

    # Clean slate: drop torch-related modules so we can detect a re-import.
    for mod in list(sys.modules):
        if mod == 'torch' or mod.startswith('torch.'):
            sys.modules.pop(mod, None)
    sys.modules.pop('bo_dispatch', None)
    sys.modules.pop('BO_botorch', None)

    try:
        importlib.import_module('bo_dispatch')
        assert 'torch' not in sys.modules, (
            "bo_dispatch pulled in torch under BO_METHOD='bayes_opt'"
        )
    finally:
        config.BO_METHOD = original
        sys.modules.pop('bo_dispatch', None)
