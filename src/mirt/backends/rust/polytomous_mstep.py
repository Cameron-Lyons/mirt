"""Batched native polytomous optimization.

Fallback mode: optional. Unsupported models retain generic item optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust._helpers import _ensure_f64, _ensure_i32, mirt_rs, rust_enabled

FALLBACK_MODE = "optional"

if TYPE_CHECKING:
    from mirt.estimation._em_context import EMFitContext
    from mirt.models.base import BaseItemModel


def try_polytomous_m_step(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    posterior: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    max_iter: int,
    ftol: float,
    epsilon: float,
    n_jobs: int,
    context: EMFitContext | None = None,
) -> bool:
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        PartialCreditModel,
    )
    from mirt.scoring._common import resolve_n_jobs

    if (
        not rust_enabled()
        or model.n_factors != 1
        or not uses_builtin_model_hooks(model)
        or type(model)
        not in (GradedResponseModel, GeneralizedPartialCredit, PartialCreditModel)
    ):
        return False
    # Nonstandard controls retain the generic optimizer's behavior.
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
        or not np.isfinite(ftol)
        or ftol <= 0
        or not np.isfinite(epsilon)
        or not 0 < epsilon < 0.5
    ):
        return False
    grm = type(model) is GradedResponseModel
    name = "thresholds" if grm else "steps"
    params, masks = model.parameters, model.free_parameter_masks
    packed = np.column_stack((params["discrimination"], params[name]))
    free = np.column_stack((masks["discrimination"], masks[name]))
    lower = np.full(packed.shape[1], -6.0)
    upper = np.full(packed.shape[1], 6.0)
    lower[0], upper[0] = 0.1, 5.0
    if not np.all(np.isfinite(packed)) or np.any(
        free & ((packed < lower) | (packed > upper))
    ):
        return False
    workers = resolve_n_jobs(n_jobs)
    optimized = mirt_rs.m_step_polytomous(
        _ensure_i32(responses),
        _ensure_f64(posterior),
        _ensure_f64(points.ravel()),
        _ensure_f64(packed),
        np.ascontiguousarray(free),
        _ensure_i32(np.asarray(model.n_categories)),
        grm,
        int(max_iter),
        float(ftol),
        float(epsilon),
        workers,
        None if context is None else context.native_pool(workers),
    )
    updated = {name: optimized[:, 1:]}
    if type(model) is not PartialCreditModel:
        updated["discrimination"] = optimized[:, 0]
    model.set_parameters(**updated)
    return True
