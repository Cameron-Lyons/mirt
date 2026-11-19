"""Frequency compression for exchangeable EM response rows."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from mirt._model_defaults import uses_builtin_model_hooks

_Response = TypeVar("_Response", bound=np.generic)


def _group_rows(rows: NDArray[_Response]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    from mirt.backends.rust.patterns import response_pattern_indices

    if rows.dtype.kind in "fu":
        # Use exact byte keys for floats and unsigned integers. Selecting
        # representatives from the original rows preserves fractions, NaNs
        # and unsigned values outside the signed integer range.
        keys = np.ascontiguousarray(rows).view(np.int8).reshape(len(rows), -1)
    elif rows.dtype.kind == "i" and rows.min(initial=-1) < -1:
        keys = np.maximum(rows, -1)
    else:
        keys = rows
    first, _, counts = response_pattern_indices(keys)
    return first, counts


def compress_responses(
    responses: NDArray[_Response],
) -> tuple[NDArray[_Response], NDArray[np.float64] | None]:
    """Group only when a bounded sample predicts substantial repeated work."""
    n_persons = responses.shape[0]
    if n_persons < 256:
        return responses, None
    sample = responses[np.linspace(0, n_persons - 1, min(1024, n_persons), dtype=int)]
    first, _ = _group_rows(sample)
    if first.size > sample.shape[0] // 2:
        return responses, None
    first, counts = _group_rows(responses)
    if first.size > n_persons // 2:
        return responses, None
    patterns = responses[first]
    if patterns.dtype.kind == "i":
        np.maximum(patterns, -1, out=patterns)
    return patterns, counts.astype(np.float64)


def supports_pattern_compression(model: object) -> bool:
    # Exact types protect custom models with person-specific likelihoods.
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
        PartialCreditModel,
    )

    return uses_builtin_model_hooks(model, likelihood=True) and type(model) in (
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
    )
