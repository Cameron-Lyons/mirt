"""Shared validation and normal-interval helpers for result objects."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError


def validate_alpha(alpha: float) -> float:
    """Validate and normalize a two-sided significance level."""
    if isinstance(alpha, bool):
        raise MirtValidationError(
            "alpha must be a finite number strictly between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        )
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "alpha must be a finite number strictly between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        ) from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise MirtValidationError(
            "alpha must be a finite number strictly between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        )
    return value


def normal_critical_value(alpha: float) -> float:
    """Return a stable two-sided standard-normal critical value."""
    from scipy import special

    validated = validate_alpha(alpha)
    return float(-special.ndtri_exp(np.log(validated / 2.0)))


def broadcast_cut_scores(
    cut_score: ArrayLike,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Validate finite cut scores and broadcast them to a score shape."""
    expected = f"finite values broadcastable to {shape}"
    raw_cuts = np.asarray(cut_score)
    if raw_cuts.dtype.kind not in {"i", "u", "f"}:
        raise MirtValidationError(
            "cut_score must contain only finite numbers",
            parameter="cut_score",
            value=cut_score,
            expected=expected,
        )
    cuts = raw_cuts.astype(np.float64, copy=False)
    if not np.all(np.isfinite(cuts)):
        raise MirtValidationError(
            "cut_score must contain only finite numbers",
            parameter="cut_score",
            value=cut_score,
            expected=expected,
        )
    try:
        return np.broadcast_to(cuts, shape)
    except ValueError as exc:
        raise MirtValidationError(
            "cut_score must be broadcastable to the score shape",
            parameter="cut_score",
            value=cuts.shape,
            expected=str(shape),
        ) from exc


def validate_classification_confidence(confidence: float) -> float:
    """Validate the probability required for an above/below decision."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        value = math.nan
    else:
        value = float(confidence)
    if not math.isfinite(value) or not 0.5 < value < 1.0:
        raise MirtValidationError(
            "confidence must be a finite number strictly between 0.5 and 1",
            parameter="confidence",
            value=confidence,
            expected="0.5 < confidence < 1",
        )
    return value


def classify_from_probabilities(
    probabilities: NDArray[np.float64],
    confidence: float,
) -> NDArray[np.str_]:
    """Label probabilities of exceeding a cut as above, below, or uncertain.

    A decision is made only when the probability of the chosen side reaches
    ``confidence``; unknown (``NaN``) probabilities remain uncertain.
    """
    resolved = validate_classification_confidence(confidence)
    classifications = np.full(probabilities.shape, "uncertain", dtype="U9")
    classifications[probabilities >= resolved] = "above"
    classifications[probabilities <= 1.0 - resolved] = "below"
    return classifications


def resolve_item_model(model_or_result: Any) -> Any:
    """Return the item model wrapped by a ``FitResult``, or the input itself."""
    from mirt.results.fit_result import FitResult

    if isinstance(model_or_result, FitResult):
        return model_or_result.model
    return model_or_result
