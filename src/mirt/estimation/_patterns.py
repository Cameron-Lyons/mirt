"""Frequency compression for exchangeable EM response rows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._model_defaults import uses_builtin_model_hooks


def compress_responses(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.float64] | None]:
    """Group only when a bounded sample predicts substantial repeated work."""
    from mirt.backends.rust.patterns import response_pattern_indices

    n_persons = responses.shape[0]
    if n_persons < 256:
        return responses, None
    sample = responses[np.linspace(0, n_persons - 1, min(1024, n_persons), dtype=int)]
    sample = np.where(sample < 0, -1, sample)
    first, _, _ = response_pattern_indices(sample)
    if first.size > sample.shape[0] // 2:
        return responses, None
    normalized = (
        responses if np.all(responses >= -1) else np.where(responses < 0, -1, responses)
    )
    first, _, counts = response_pattern_indices(normalized)
    if first.size > n_persons // 2:
        return responses, None
    return normalized[first], counts.astype(np.float64)


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
