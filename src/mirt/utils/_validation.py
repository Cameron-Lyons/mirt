"""Shared scalar and vector argument validators for utility functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError


def validate_finite_scalar(value: float, parameter: str) -> float:
    """Return ``value`` as a float, rejecting Booleans, arrays and non-finite values."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        ) from exc
    if not np.isfinite(result):
        raise MirtValidationError(
            f"{parameter} must be a finite number",
            parameter=parameter,
            value=value,
        )
    return result


def validate_positive_scalar(value: float, parameter: str) -> float:
    """Return a finite, strictly positive scalar as a float."""
    result = validate_finite_scalar(value, parameter)
    if result <= 0.0:
        raise MirtValidationError(
            f"{parameter} must be positive",
            parameter=parameter,
            value=value,
            expected="> 0",
        )
    return result


def validate_alpha(alpha: float) -> float:
    """Return a significance level strictly between 0 and 1."""
    value = validate_finite_scalar(alpha, "alpha")
    if not 0.0 < value < 1.0:
        raise MirtValidationError(
            "alpha must be between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        )
    return value


def as_finite_vector(values: ArrayLike, parameter: str) -> NDArray[np.float64]:
    """Convert a numeric input to a nonempty, finite, one-dimensional array."""
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            f"{parameter} must contain numeric values",
            parameter=parameter,
        ) from exc
    if result.size == 0:
        raise MirtValidationError(
            f"{parameter} must contain at least one value",
            parameter=parameter,
        )
    if not np.all(np.isfinite(result)):
        raise MirtValidationError(
            f"{parameter} must contain only finite values",
            parameter=parameter,
        )
    return result
