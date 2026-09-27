"""Batched MAP/ML optimization for built-in unidimensional logistic models.

Fallback mode: optional. Other models retain the generic Python scorer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust._helpers import _ensure_f64, _ensure_i32, mirt_rs, rust_enabled

FALLBACK_MODE = "optional"

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def try_optimized_scores(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    *,
    bounds: tuple[float, float],
    method: Literal["MAP", "ML"],
    n_jobs: int,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.scoring._common import resolve_n_jobs

    if (
        not rust_enabled()
        or model.n_factors != 1
        or any(
            name in vars(model)
            for name in ("probability", "log_likelihood", "information")
        )
        or type(model)
        not in (
            OneParameterLogistic,
            TwoParameterLogistic,
            ThreeParameterLogistic,
            FourParameterLogistic,
        )
    ):
        return None
    parameters = model.parameters
    packed = np.column_stack(
        (
            parameters["discrimination"],
            parameters["difficulty"],
            parameters.get("guessing", np.zeros(model.n_items)),
            parameters.get("upper", np.ones(model.n_items)),
        )
    )
    return mirt_rs.compute_optimized_scores(
        _ensure_i32(responses),
        _ensure_f64(packed),
        *bounds,
        method == "MAP",
        float(prior_mean),
        float(prior_var),
        resolve_n_jobs(n_jobs),
    )
