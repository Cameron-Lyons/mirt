"""Custom exceptions for the mirt package.

This module defines a hierarchy of exceptions for better error handling
and programmatic error detection.

Exception Hierarchy
-------------------
MirtError
    Base exception for all mirt-related errors.

MirtValidationError
    Raised when input validation fails (data shape, parameter bounds, etc.).

MirtEstimationError
    Raised when model estimation fails to converge or encounters numerical issues.

MirtConvergenceError
    Raised specifically when an iterative algorithm fails to converge.

MirtModelError
    Raised when there are issues with model specification or configuration.

MirtDataError
    Raised when there are issues with input data (missing values, invalid responses, etc.).

Examples
--------
>>> from mirt.exceptions import MirtValidationError
>>> raise MirtValidationError("n_items must be positive", parameter="n_items", value=-1)
MirtValidationError: n_items must be positive (parameter='n_items', value=-1)
"""

from __future__ import annotations

from typing import Any


class MirtError(Exception):
    """Base exception for all mirt-related errors.

    All custom exceptions in the mirt package inherit from this class,
    allowing users to catch all mirt-specific errors with a single
    except clause.

    Parameters
    ----------
    message : str
        Human-readable error message.
    **kwargs
        Additional context to include in the error.

    Examples
    --------
    >>> try:
    ...     # some mirt operation
    ...     pass
    ... except MirtError as e:
    ...     print(f"MIRT error: {e}")
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        self.message = message
        self.context = kwargs
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class MirtValidationError(MirtError):
    """Raised when input validation fails.

    This exception is raised when function arguments or data do not meet
    the required constraints (e.g., wrong shape, out-of-bounds values,
    invalid types).

    Parameters
    ----------
    message : str
        Description of the validation error.
    parameter : str, optional
        Name of the parameter that failed validation.
    value : Any, optional
        The invalid value that was provided.
    expected : str, optional
        Description of what was expected.

    Examples
    --------
    >>> raise MirtValidationError(
    ...     "Invalid number of items",
    ...     parameter="n_items",
    ...     value=-5,
    ...     expected="positive integer"
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
        expected: str | None = None,
        **kwargs: Any,
    ) -> None:
        if parameter is not None:
            kwargs["parameter"] = parameter
        if value is not None:
            kwargs["value"] = value
        if expected is not None:
            kwargs["expected"] = expected
        super().__init__(message, **kwargs)


class MirtEstimationError(MirtError):
    """Raised when model estimation fails.

    This exception is raised when the estimation algorithm encounters
    an error that prevents it from completing successfully.

    Parameters
    ----------
    message : str
        Description of the estimation error.
    iteration : int, optional
        The iteration at which the error occurred.
    log_likelihood : float, optional
        The log-likelihood value at failure.

    Examples
    --------
    >>> raise MirtEstimationError(
    ...     "Numerical overflow in likelihood computation",
    ...     iteration=15,
    ...     log_likelihood=float('-inf')
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        iteration: int | None = None,
        log_likelihood: float | None = None,
        **kwargs: Any,
    ) -> None:
        if iteration is not None:
            kwargs["iteration"] = iteration
        if log_likelihood is not None:
            kwargs["log_likelihood"] = log_likelihood
        super().__init__(message, **kwargs)


class MirtConvergenceError(MirtEstimationError):
    """Raised when an iterative algorithm fails to converge.

    This exception is raised when the maximum number of iterations
    is reached without meeting the convergence criterion.

    Parameters
    ----------
    message : str
        Description of the convergence failure.
    max_iterations : int, optional
        The maximum number of iterations that was attempted.
    final_change : float, optional
        The parameter change at the final iteration.
    tolerance : float, optional
        The convergence tolerance that was not met.

    Examples
    --------
    >>> raise MirtConvergenceError(
    ...     "EM algorithm did not converge",
    ...     max_iterations=100,
    ...     final_change=0.01,
    ...     tolerance=0.001
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        max_iterations: int | None = None,
        final_change: float | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> None:
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        if final_change is not None:
            kwargs["final_change"] = final_change
        if tolerance is not None:
            kwargs["tolerance"] = tolerance
        super().__init__(message, **kwargs)


class MirtModelError(MirtError):
    """Raised when there are issues with model specification.

    This exception is raised when a model is incorrectly specified
    or configured (e.g., incompatible options, missing required
    parameters).

    Parameters
    ----------
    message : str
        Description of the model error.
    model_type : str, optional
        The type of model that encountered the error.

    Examples
    --------
    >>> raise MirtModelError(
    ...     "Bifactor model requires at least 2 specific factors",
    ...     model_type="BifactorModel"
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        if model_type is not None:
            kwargs["model_type"] = model_type
        super().__init__(message, **kwargs)


class MirtDataError(MirtError):
    """Raised when there are issues with input data.

    This exception is raised when the input data has problems that
    prevent analysis (e.g., all missing values, invalid response
    codes, insufficient variance).

    Parameters
    ----------
    message : str
        Description of the data error.
    n_persons : int, optional
        Number of persons in the data.
    n_items : int, optional
        Number of items in the data.
    item_idx : int, optional
        Index of the problematic item, if applicable.

    Examples
    --------
    >>> raise MirtDataError(
    ...     "Item has no variance (all responses identical)",
    ...     item_idx=5,
    ...     n_persons=100
    ... )
    """

    def __init__(
        self,
        message: str,
        *,
        n_persons: int | None = None,
        n_items: int | None = None,
        item_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        if n_persons is not None:
            kwargs["n_persons"] = n_persons
        if n_items is not None:
            kwargs["n_items"] = n_items
        if item_idx is not None:
            kwargs["item_idx"] = item_idx
        super().__init__(message, **kwargs)


__all__ = [
    "MirtError",
    "MirtValidationError",
    "MirtEstimationError",
    "MirtConvergenceError",
    "MirtModelError",
    "MirtDataError",
]
