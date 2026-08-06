"""Shared validation and normal-interval helpers for result objects."""

from __future__ import annotations

import math

import numpy as np

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
