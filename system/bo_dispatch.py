# system/bo_dispatch.py
# Thin dispatcher that re-exports the BO public surface from either
# the single-objective bayes_opt backend (BO.py) or the multi-objective
# BoTorch backend (BO_botorch.py), selected by config.BO_METHOD.

import config

_PUBLIC = ('train_optimizer', 'get_next_sample', 'register_result',
           'get_current_kappa', 'get_best_result')

if config.BO_METHOD == 'botorch':
    # BO_botorch only imports stdlib + numpy/pandas at module load;
    # torch/gpytorch/botorch are pulled in on the first function call
    # via BO_botorch._get_torch_ctx(), which raises a clear install-hint
    # error if any of those dependencies are missing.
    from BO_botorch import (
        train_optimizer,
        get_next_sample,
        register_result,
        get_current_kappa,
        get_best_result,
    )
elif config.BO_METHOD == 'bayes_opt':
    from BO import (
        train_optimizer,
        get_next_sample,
        register_result,
        get_current_kappa,
        get_best_result,
    )
else:
    raise ValueError(
        f"Unknown BO_METHOD: {config.BO_METHOD!r}. Expected 'bayes_opt' or 'botorch'."
    )

__all__ = list(_PUBLIC)
