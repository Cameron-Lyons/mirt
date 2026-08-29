"""CAT result classes for tracking adaptive testing state and outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError


def _integer_list(values: Any, *, name: str) -> list[int]:
    """Return a validated one-dimensional list of non-negative integers."""
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise MirtValidationError(
                f"{name} must be one-dimensional",
                parameter=name,
                value=values.shape,
                expected="one-dimensional integer sequence",
            )
        candidates = values.tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        candidates = list(values)
    else:
        raise MirtValidationError(
            f"{name} must be a sequence of integers",
            parameter=name,
            value=type(values).__name__,
            expected="one-dimensional integer sequence",
        )

    result: list[int] = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise MirtValidationError(
                f"{name} must contain only integers",
                parameter=name,
                value=value,
                expected="non-negative integers",
            )
        normalized = int(value)
        if normalized < 0:
            raise MirtValidationError(
                f"{name} cannot contain negative values",
                parameter=name,
                value=normalized,
                expected=">= 0",
            )
        result.append(normalized)
    return result


def _float_array(values: Any, *, name: str, ndim: int) -> NDArray[np.float64]:
    """Return a copied floating-point array with the requested rank."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MirtValidationError(
            f"{name} must contain numeric values",
            parameter=name,
            expected=f"{ndim}-dimensional numeric array",
        ) from error
    if result.ndim != ndim:
        raise MirtValidationError(
            f"{name} must be {ndim}-dimensional",
            parameter=name,
            value=result.shape,
            expected=f"{ndim}-dimensional numeric array",
        )
    return result.copy()


def _float_history(values: Any, *, name: str, limit: int) -> list[float]:
    """Normalize a scalar history and ensure it fits the administration."""
    result = _float_array(values, name=name, ndim=1)
    if result.size > limit:
        raise MirtValidationError(
            f"{name} cannot be longer than the administered item sequence",
            parameter=name,
            value=result.size,
            expected=f"at most {limit} values",
        )
    return result.tolist()


def _array_history(
    values: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    limit: int,
) -> list[NDArray[np.float64]]:
    """Normalize a history of arrays with a fixed per-step shape."""
    if isinstance(values, np.ndarray):
        candidates = list(values)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        candidates = list(values)
    else:
        raise MirtValidationError(
            f"{name} must be a sequence of arrays",
            parameter=name,
            value=type(values).__name__,
            expected=f"at most {limit} arrays with shape {shape}",
        )
    if len(candidates) > limit:
        raise MirtValidationError(
            f"{name} cannot be longer than the administered item sequence",
            parameter=name,
            value=len(candidates),
            expected=f"at most {limit} arrays",
        )

    result = []
    for position, value in enumerate(candidates):
        array = _float_array(value, name=f"{name}[{position}]", ndim=len(shape))
        if array.shape != shape:
            raise MirtValidationError(
                f"{name} entries must match the final result shape",
                parameter=name,
                value=array.shape,
                expected=str(shape),
            )
        result.append(array)
    return result


def _normalize_common_result(result: CATResult | MCATResult) -> None:
    """Validate and normalize fields shared by CAT result objects."""
    items = _integer_list(result.items_administered, name="items_administered")
    responses = _integer_list(result.responses, name="responses")
    if len(responses) != len(items):
        raise MirtValidationError(
            "responses must contain one value per administered item",
            parameter="responses",
            value=len(responses),
            expected=str(len(items)),
        )
    count = result.n_items_administered
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise MirtValidationError(
            "n_items_administered must be a non-negative integer",
            parameter="n_items_administered",
            value=count,
            expected=">= 0",
        )
    if int(count) != len(items):
        raise MirtValidationError(
            "n_items_administered must match the administered item sequence",
            parameter="n_items_administered",
            value=count,
            expected=str(len(items)),
        )
    if (
        not isinstance(result.stopping_reason, str)
        or not result.stopping_reason.strip()
    ):
        raise MirtValidationError(
            "stopping_reason must be a non-empty string",
            parameter="stopping_reason",
            value=result.stopping_reason,
            expected="non-empty string",
        )

    result.items_administered = items
    result.responses = np.asarray(responses, dtype=np.int_)
    result.n_items_administered = int(count)
    result.stopping_reason = result.stopping_reason.strip()


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    kind: str,
    allowed: set[str],
    required: set[str],
) -> None:
    """Reject malformed mappings before constructing a result."""
    if not isinstance(payload, Mapping):
        raise MirtValidationError(
            f"{kind} payload must be a mapping",
            parameter="payload",
            value=type(payload).__name__,
            expected="mapping",
        )
    unknown = set(payload) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise MirtValidationError(
            f"{kind} payload contains unknown fields: {names}",
            parameter="payload",
            value=names,
            expected=f"fields produced by {kind}.to_dict()",
        )
    missing = required - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise MirtValidationError(
            f"{kind} payload is missing required fields: {names}",
            parameter="payload",
            value=names,
            expected=f"fields produced by {kind}.to_dict()",
        )


def _load_json(value: str | bytes | bytearray, *, kind: str) -> Any:
    """Decode a JSON result payload with package-specific validation errors."""
    import json

    if not isinstance(value, (str, bytes, bytearray)):
        raise MirtValidationError(
            f"{kind} JSON must be a string or bytes",
            parameter="value",
            value=type(value).__name__,
            expected="str, bytes, or bytearray",
        )
    try:
        return json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise MirtValidationError(
            f"{kind} JSON must contain a valid JSON object",
            parameter="value",
            expected=f"JSON object produced by {kind}.to_json()",
        ) from error


@dataclass
class MCATState:
    """Current state during multidimensional CAT administration.

    This class tracks the evolving state of an MCAT session, including the
    current ability estimates across multiple dimensions, items administered,
    and whether the test has reached a stopping condition.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Current ability estimates, shape (n_factors,).
    covariance : NDArray[np.float64]
        Posterior covariance matrix, shape (n_factors, n_factors).
    standard_error : NDArray[np.float64]
        Standard errors for each dimension (sqrt of diagonal of covariance).
    items_administered : list[int]
        Indices of items that have been administered.
    responses : list[int]
        Responses to administered items.
    n_items : int
        Number of items administered so far.
    is_complete : bool
        Whether the CAT session has reached a stopping condition.
    next_item : int | None
        Index of the next item to administer, or None if complete.
    """

    theta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    items_administered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    n_items: int = 0
    is_complete: bool = False
    next_item: int | None = None

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        return len(self.theta)

    @property
    def trace_covariance(self) -> float:
        """Trace of the covariance matrix (sum of variances)."""
        return float(np.trace(self.covariance))

    @property
    def det_covariance(self) -> float:
        """Determinant of the covariance matrix."""
        return float(np.linalg.det(self.covariance))

    def __repr__(self) -> str:
        theta_str = ", ".join(f"{t:.3f}" for t in self.theta)
        se_str = ", ".join(f"{s:.3f}" for s in self.standard_error)
        return (
            f"MCATState(theta=[{theta_str}], "
            f"se=[{se_str}], "
            f"n_items={self.n_items}, "
            f"complete={self.is_complete})"
        )


@dataclass
class MCATResult:
    """Final result of a completed multidimensional CAT session.

    This class contains the complete record of an MCAT administration,
    including the final ability estimates across all dimensions, all items
    administered, responses given, and the history of ability estimates.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Final ability estimates, shape (n_factors,).
    covariance : NDArray[np.float64]
        Final posterior covariance matrix, shape (n_factors, n_factors).
    standard_error : NDArray[np.float64]
        Standard errors for each dimension.
    items_administered : list[int]
        Indices of all items administered in order.
    responses : numpy.ndarray
        Array of responses to administered items.
    n_items_administered : int
        Total number of items administered.
    stopping_reason : str
        Description of why the test stopped.
    theta_history : list[NDArray[np.float64]]
        History of ability estimates after each item.
    se_history : list[NDArray[np.float64]]
        History of standard errors after each item.
    covariance_history : list[NDArray[np.float64]]
        History of covariance matrices after each item.
    item_info_history : list[float]
        History of item information values for selected items.
    """

    theta: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    items_administered: list[int]
    responses: NDArray[np.int_]
    n_items_administered: int
    stopping_reason: str
    theta_history: list[NDArray[np.float64]] = field(default_factory=list)
    se_history: list[NDArray[np.float64]] = field(default_factory=list)
    covariance_history: list[NDArray[np.float64]] = field(default_factory=list)
    item_info_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate dimensions and copy caller-owned result data."""
        _normalize_common_result(self)
        theta = _float_array(self.theta, name="theta", ndim=1)
        if theta.size == 0:
            raise MirtValidationError(
                "theta must contain at least one factor",
                parameter="theta",
                value=theta.shape,
                expected="(n_factors,) with n_factors > 0",
            )
        covariance = _float_array(self.covariance, name="covariance", ndim=2)
        expected_covariance_shape = (theta.size, theta.size)
        if covariance.shape != expected_covariance_shape:
            raise MirtValidationError(
                "covariance must contain one row and column per factor",
                parameter="covariance",
                value=covariance.shape,
                expected=str(expected_covariance_shape),
            )
        if not np.allclose(covariance, covariance.T, equal_nan=True):
            raise MirtValidationError(
                "covariance must be symmetric",
                parameter="covariance",
                expected="symmetric matrix",
            )
        standard_error = _float_array(
            self.standard_error,
            name="standard_error",
            ndim=1,
        )
        if standard_error.shape != theta.shape:
            raise MirtValidationError(
                "standard_error must contain one value per factor",
                parameter="standard_error",
                value=standard_error.shape,
                expected=str(theta.shape),
            )
        if np.any(standard_error < 0.0):
            raise MirtValidationError(
                "standard_error cannot contain negative values",
                parameter="standard_error",
                expected=">= 0, NaN, or infinity",
            )

        count = self.n_items_administered
        self.theta = theta
        self.covariance = covariance
        self.standard_error = standard_error
        self.theta_history = _array_history(
            self.theta_history,
            name="theta_history",
            shape=theta.shape,
            limit=count,
        )
        self.se_history = _array_history(
            self.se_history,
            name="se_history",
            shape=theta.shape,
            limit=count,
        )
        self.covariance_history = _array_history(
            self.covariance_history,
            name="covariance_history",
            shape=covariance.shape,
            limit=count,
        )
        self.item_info_history = _float_history(
            self.item_info_history,
            name="item_info_history",
            limit=count,
        )

    @property
    def n_factors(self) -> int:
        """Number of latent factors."""
        return len(self.theta)

    def to_dict(self) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible result representation."""
        return {
            "theta": self.theta.tolist(),
            "covariance": self.covariance.tolist(),
            "standard_error": self.standard_error.tolist(),
            "n_factors": self.n_factors,
            "items_administered": list(self.items_administered),
            "responses": self.responses.tolist(),
            "n_items_administered": self.n_items_administered,
            "stopping_reason": self.stopping_reason,
            "theta_history": [value.tolist() for value in self.theta_history],
            "se_history": [value.tolist() for value in self.se_history],
            "covariance_history": [value.tolist() for value in self.covariance_history],
            "item_info_history": list(self.item_info_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a multidimensional result from :meth:`to_dict` output."""
        allowed = {
            "theta",
            "covariance",
            "standard_error",
            "n_factors",
            "items_administered",
            "responses",
            "n_items_administered",
            "stopping_reason",
            "theta_history",
            "se_history",
            "covariance_history",
            "item_info_history",
        }
        required = {
            "theta",
            "covariance",
            "standard_error",
            "items_administered",
            "responses",
            "n_items_administered",
            "stopping_reason",
        }
        _validate_payload(
            payload,
            kind=cls.__name__,
            allowed=allowed,
            required=required,
        )
        result = cls(
            theta=payload["theta"],
            covariance=payload["covariance"],
            standard_error=payload["standard_error"],
            items_administered=payload["items_administered"],
            responses=payload["responses"],
            n_items_administered=payload["n_items_administered"],
            stopping_reason=payload["stopping_reason"],
            theta_history=payload.get("theta_history", []),
            se_history=payload.get("se_history", []),
            covariance_history=payload.get("covariance_history", []),
            item_info_history=payload.get("item_info_history", []),
        )
        if "n_factors" in payload:
            value = payload["n_factors"]
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) != result.n_factors
            ):
                raise MirtValidationError(
                    "n_factors does not match the reconstructed result shape",
                    parameter="n_factors",
                    value=value,
                    expected=str(result.n_factors),
                )
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the multidimensional result to JSON."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Reconstruct a multidimensional result from :meth:`to_json` output."""
        return cls.from_dict(_load_json(value, kind=cls.__name__))

    def summary(self) -> str:
        """Return a formatted summary of the MCAT result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "MCAT Result Summary",
            "=" * 50,
            f"Dimensions:            {self.n_factors}",
            f"Items administered:    {self.n_items_administered}",
            f"Stopping reason:       {self.stopping_reason}",
            "",
            "Final theta estimates:",
        ]
        for i, (t, se) in enumerate(zip(self.theta, self.standard_error, strict=True)):
            lines.append(f"  Dimension {i + 1}: {t:7.4f} (SE: {se:.4f})")

        lines.extend(
            [
                "",
                f"Trace(Cov):            {np.trace(self.covariance):.4f}",
                f"Det(Cov):              {np.linalg.det(self.covariance):.6f}",
                "",
                "Response pattern:",
                f"  Correct: {np.sum(self.responses == 1)} / {self.n_items_administered}",
            ]
        )
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert MCAT history to a DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns for step, item, response, and theta/se per dimension.
        """
        from mirt.utils.dataframe import create_dataframe

        n = len(self.items_administered)
        data: dict[str, Any] = {
            "step": list(range(1, n + 1)),
            "item": self.items_administered,
            "response": list(self.responses),
        }

        for d in range(self.n_factors):
            theta_key = f"theta_{d + 1}"
            se_key = f"se_{d + 1}"
            if self.theta_history:
                data[theta_key] = [th[d] for th in self.theta_history[:n]]
            else:
                data[theta_key] = [np.nan] * n
            if self.se_history:
                data[se_key] = [se[d] for se in self.se_history[:n]]
            else:
                data[se_key] = [np.nan] * n

        if self.item_info_history:
            data["info"] = self.item_info_history[:n]

        return create_dataframe(data)

    def plot_convergence(self) -> Any:
        """Plot theta and SE convergence for each dimension.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with subplots showing theta and SE history for each dimension.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        n_factors = self.n_factors
        fig, axes = plt.subplots(
            n_factors, 2, figsize=(10, 3 * n_factors), squeeze=False
        )

        steps = list(range(1, len(self.theta_history) + 1))
        colors = plt.cm.tab10(np.linspace(0, 1, n_factors))

        for d in range(n_factors):
            theta_d = [th[d] for th in self.theta_history]
            se_d = [se[d] for se in self.se_history]

            axes[d, 0].plot(steps, theta_d, "-o", markersize=4, color=colors[d])
            axes[d, 0].axhline(y=self.theta[d], color="r", linestyle="--", alpha=0.7)
            axes[d, 0].set_ylabel(f"Theta {d + 1}")
            axes[d, 0].grid(True, alpha=0.3)

            axes[d, 1].plot(steps, se_d, "-o", markersize=4, color=colors[d])
            axes[d, 1].axhline(
                y=self.standard_error[d], color="r", linestyle="--", alpha=0.7
            )
            axes[d, 1].set_ylabel(f"SE {d + 1}")
            axes[d, 1].grid(True, alpha=0.3)

        axes[0, 0].set_title("Theta Convergence")
        axes[0, 1].set_title("SE Convergence")
        axes[-1, 0].set_xlabel("Items Administered")
        axes[-1, 1].set_xlabel("Items Administered")

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        theta_str = ", ".join(f"{t:.3f}" for t in self.theta)
        return (
            f"MCATResult(theta=[{theta_str}], "
            f"n_items={self.n_items_administered}, "
            f"reason='{self.stopping_reason}')"
        )


@dataclass
class CATState:
    """Current state during CAT administration.

    This class tracks the evolving state of a CAT session, including the
    current ability estimate, items administered, and whether the test
    has reached a stopping condition.

    Attributes
    ----------
    theta : float
        Current ability estimate.
    standard_error : float
        Standard error of the current ability estimate.
    items_administered : list[int]
        Indices of items that have been administered.
    responses : list[int]
        Responses to administered items (0/1 for dichotomous, 0..k for polytomous).
    n_items : int
        Number of items administered so far.
    is_complete : bool
        Whether the CAT session has reached a stopping condition.
    next_item : int | None
        Index of the next item to administer, or None if complete.
    """

    theta: float
    standard_error: float
    items_administered: list[int] = field(default_factory=list)
    responses: list[int] = field(default_factory=list)
    n_items: int = 0
    is_complete: bool = False
    next_item: int | None = None

    def __repr__(self) -> str:
        return (
            f"CATState(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items}, "
            f"complete={self.is_complete})"
        )


@dataclass
class CATResult:
    """Final result of a completed CAT session.

    This class contains the complete record of a CAT administration,
    including the final ability estimate, all items administered,
    responses given, and the history of ability estimates.

    Attributes
    ----------
    theta : float
        Final ability estimate.
    standard_error : float
        Standard error of the final ability estimate.
    items_administered : list[int]
        Indices of all items administered in order.
    responses : numpy.ndarray
        Array of responses to administered items.
    n_items_administered : int
        Total number of items administered.
    stopping_reason : str
        Description of why the test stopped (e.g., "SE threshold reached").
    theta_history : list[float]
        History of ability estimates after each item.
    se_history : list[float]
        History of standard errors after each item.
    item_info_history : list[float]
        History of item information values for selected items.
    """

    theta: float
    standard_error: float
    items_administered: list[int]
    responses: NDArray[np.int_]
    n_items_administered: int
    stopping_reason: str
    theta_history: list[float] = field(default_factory=list)
    se_history: list[float] = field(default_factory=list)
    item_info_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate dimensions and copy caller-owned result data."""
        _normalize_common_result(self)
        try:
            theta = float(self.theta)
            standard_error = float(self.standard_error)
        except (TypeError, ValueError) as error:
            raise MirtValidationError(
                "theta and standard_error must be numeric scalars",
                expected="numeric scalars",
            ) from error
        if standard_error < 0.0:
            raise MirtValidationError(
                "standard_error cannot be negative",
                parameter="standard_error",
                value=standard_error,
                expected=">= 0, NaN, or infinity",
            )

        count = self.n_items_administered
        self.theta = theta
        self.standard_error = standard_error
        self.theta_history = _float_history(
            self.theta_history,
            name="theta_history",
            limit=count,
        )
        self.se_history = _float_history(
            self.se_history,
            name="se_history",
            limit=count,
        )
        self.item_info_history = _float_history(
            self.item_info_history,
            name="item_info_history",
            limit=count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a dependency-free, JSON-compatible result representation."""
        return {
            "theta": self.theta,
            "standard_error": self.standard_error,
            "items_administered": list(self.items_administered),
            "responses": self.responses.tolist(),
            "n_items_administered": self.n_items_administered,
            "stopping_reason": self.stopping_reason,
            "theta_history": list(self.theta_history),
            "se_history": list(self.se_history),
            "item_info_history": list(self.item_info_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a unidimensional result from :meth:`to_dict` output."""
        allowed = {
            "theta",
            "standard_error",
            "items_administered",
            "responses",
            "n_items_administered",
            "stopping_reason",
            "theta_history",
            "se_history",
            "item_info_history",
        }
        required = {
            "theta",
            "standard_error",
            "items_administered",
            "responses",
            "n_items_administered",
            "stopping_reason",
        }
        _validate_payload(
            payload,
            kind=cls.__name__,
            allowed=allowed,
            required=required,
        )
        return cls(
            theta=payload["theta"],
            standard_error=payload["standard_error"],
            items_administered=payload["items_administered"],
            responses=payload["responses"],
            n_items_administered=payload["n_items_administered"],
            stopping_reason=payload["stopping_reason"],
            theta_history=payload.get("theta_history", []),
            se_history=payload.get("se_history", []),
            item_info_history=payload.get("item_info_history", []),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the unidimensional result to JSON."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Reconstruct a unidimensional result from :meth:`to_json` output."""
        return cls.from_dict(_load_json(value, kind=cls.__name__))

    def summary(self) -> str:
        """Return a formatted summary of the CAT result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "CAT Result Summary",
            "=" * 40,
            f"Final theta estimate:  {self.theta:.4f}",
            f"Standard error:        {self.standard_error:.4f}",
            f"Items administered:    {self.n_items_administered}",
            f"Stopping reason:       {self.stopping_reason}",
            "",
            "Response pattern:",
            f"  Correct: {np.sum(self.responses == 1)} / {self.n_items_administered}",
            f"  Items:   {self.items_administered}",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Convert CAT history to a DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with columns: item, response, theta, se, info.
        """
        from mirt.utils.dataframe import create_dataframe

        n = len(self.items_administered)
        data: dict[str, Any] = {
            "step": list(range(1, n + 1)),
            "item": self.items_administered,
            "response": list(self.responses),
            "theta": self.theta_history[:n] if self.theta_history else [np.nan] * n,
            "se": self.se_history[:n] if self.se_history else [np.nan] * n,
        }

        if self.item_info_history:
            data["info"] = self.item_info_history[:n]

        return create_dataframe(data)

    def to_array(self) -> NDArray[np.float64]:
        """Convert result to numpy array.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_items, 4) containing [item, response, theta, se].
        """
        n = len(self.items_administered)
        arr = np.zeros((n, 4), dtype=np.float64)
        arr[:, 0] = self.items_administered
        arr[:, 1] = self.responses
        if self.theta_history:
            arr[:, 2] = self.theta_history[:n]
        if self.se_history:
            arr[:, 3] = self.se_history[:n]
        return arr

    def plot_convergence(self) -> Any:
        """Plot theta and SE convergence over items.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with two subplots showing theta and SE history.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from e

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        steps = list(range(1, len(self.theta_history) + 1))

        ax1.plot(steps, self.theta_history, "b-o", markersize=4)
        ax1.axhline(y=self.theta, color="r", linestyle="--", alpha=0.7)
        ax1.set_ylabel("Theta Estimate")
        ax1.set_title("CAT Convergence")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, self.se_history, "g-o", markersize=4)
        ax2.axhline(y=self.standard_error, color="r", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Items Administered")
        ax2.set_ylabel("Standard Error")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"CATResult(theta={self.theta:.3f}, "
            f"se={self.standard_error:.3f}, "
            f"n_items={self.n_items_administered}, "
            f"reason='{self.stopping_reason}')"
        )
