"""Clinical change statistics for IRT score comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError

RCIMethod = Literal["jacobson", "hageman", "iverson"]

_RCI_METHODS = ("jacobson", "hageman", "iverson")


@dataclass
class RCIResult:
    """Result of a Reliable Change Index computation.

    ``se_diff`` retains the historical scalar summary (the mean standard
    error), while ``se_diff_by_person`` contains the denominator actually
    used for each RCI. This preserves compatibility while making
    heterogeneous measurement precision visible.
    """

    rci: NDArray[np.float64]
    significant: NDArray[np.bool_]
    direction: NDArray[np.str_]
    se_diff: float
    critical_value: float
    change: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    se_diff_by_person: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    ci_lower: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    ci_upper: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    method: str = "jacobson"
    higher_is_better: bool = True

    @property
    def n_persons(self) -> int:
        """Number of paired scores in the result."""
        return int(self.rci.size)

    def summary(self) -> dict[str, int]:
        """Return counts for each reliable-change classification."""
        return {
            "n_persons": self.n_persons,
            "significant": int(np.count_nonzero(self.significant)),
            "improved": int(np.count_nonzero(self.direction == "improved")),
            "declined": int(np.count_nonzero(self.direction == "declined")),
            "unchanged": int(np.count_nonzero(self.direction == "unchanged")),
        }


def RCI(
    theta_pre: ArrayLike,
    theta_post: ArrayLike,
    sem_pre: ArrayLike | None = None,
    sem_post: ArrayLike | None = None,
    reliability: float | None = None,
    sd_theta: float = 1.0,
    alpha: float = 0.05,
    method: RCIMethod = "jacobson",
    higher_is_better: bool = True,
) -> RCIResult:
    """Compute person-specific Reliable Change Indices.

    Parameters
    ----------
    theta_pre, theta_post : array-like
        Paired pre- and post-treatment scores.
    sem_pre, sem_post : array-like or float, optional
        Standard errors of measurement. Scalars are broadcast to every
        person. If ``sem_pre`` is omitted, it is derived from ``reliability``
        and ``sd_theta``. If ``sem_post`` is omitted, ``sem_pre`` is reused.
    reliability : float, optional
        Reliability in ``[0, 1)`` used when ``sem_pre`` is omitted.
    sd_theta : float
        Positive score standard deviation used with ``reliability``.
    alpha : float
        Two-sided significance level in ``(0, 1)``.
    method : {"jacobson", "hageman", "iverson"}
        Standard-error convention. Jacobson uses ``sqrt(2) * sem_pre``;
        Hageman and Iverson combine pre- and post-test measurement errors.
    higher_is_better : bool
        Whether positive score change represents improvement.

    Returns
    -------
    RCIResult
        Person-level RCIs, uncertainty intervals, and classifications.
    """
    from scipy import stats

    pre = _as_finite_vector(theta_pre, "theta_pre")
    post = _as_finite_vector(theta_post, "theta_post")
    if pre.size != post.size:
        raise MirtValidationError(
            "theta_pre and theta_post must have the same length",
            parameter="theta_post",
            value=post.size,
            expected=str(pre.size),
        )

    alpha_value = _validate_alpha(alpha)
    reliability_value = _validate_reliability(reliability)
    sd_value = _validate_positive_scalar(sd_theta, "sd_theta")
    _validate_method(method)
    orientation = _validate_boolean(higher_is_better, "higher_is_better")

    if sem_pre is None:
        if reliability_value is None:
            raise MirtValidationError(
                "sem_pre or reliability must be provided",
                parameter="sem_pre",
            )
        derived_sem = sd_value * np.sqrt(1.0 - reliability_value)
        pre_sem = np.full(pre.size, derived_sem, dtype=np.float64)
    else:
        pre_sem = _as_sem_vector(sem_pre, pre.size, "sem_pre")

    if sem_post is None:
        post_sem = pre_sem.copy()
    else:
        post_sem = _as_sem_vector(sem_post, post.size, "sem_post")

    if method == "jacobson":
        se_by_person = np.sqrt(2.0) * pre_sem
    else:
        se_by_person = np.sqrt(pre_sem**2 + post_sem**2)

    change = post - pre
    rci = change / se_by_person
    critical_value = float(stats.norm.ppf(1.0 - alpha_value / 2.0))
    significant = np.abs(rci) > critical_value

    oriented_rci = rci if orientation else -rci
    direction = np.where(
        oriented_rci > critical_value,
        "improved",
        np.where(oriented_rci < -critical_value, "declined", "unchanged"),
    )
    margin = critical_value * se_by_person

    return RCIResult(
        rci=rci,
        significant=significant,
        direction=direction,
        se_diff=float(np.mean(se_by_person)),
        critical_value=critical_value,
        change=change,
        se_diff_by_person=se_by_person,
        ci_lower=change - margin,
        ci_upper=change + margin,
        method=method,
        higher_is_better=orientation,
    )


def clinical_significance(
    theta_pre: ArrayLike,
    theta_post: ArrayLike,
    cutoff: float,
    reliability: float,
    sd_theta: float = 1.0,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict[str, NDArray[np.bool_]]:
    """Classify reliable change and recovery relative to a clinical cutoff.

    ``higher_is_better=False`` supports symptom scales where lower scores
    indicate better functioning. The returned masks are mutually exclusive
    and collectively exhaustive.
    """
    cutoff_value = _validate_finite_scalar(cutoff, "cutoff")
    orientation = _validate_boolean(higher_is_better, "higher_is_better")
    pre = _as_finite_vector(theta_pre, "theta_pre")
    post = _as_finite_vector(theta_post, "theta_post")

    result = RCI(
        pre,
        post,
        reliability=reliability,
        sd_theta=sd_theta,
        alpha=alpha,
        higher_is_better=orientation,
    )

    reliable_improvement = result.direction == "improved"
    reliable_decline = result.direction == "declined"
    if orientation:
        crossed_to_normal = (pre < cutoff_value) & (post >= cutoff_value)
    else:
        crossed_to_normal = (pre > cutoff_value) & (post <= cutoff_value)

    recovered = reliable_improvement & crossed_to_normal
    improved = reliable_improvement & ~crossed_to_normal
    unchanged = ~result.significant
    deteriorated = reliable_decline

    return {
        "recovered": recovered,
        "improved": improved,
        "unchanged": unchanged,
        "deteriorated": deteriorated,
    }


def _as_finite_vector(values: ArrayLike, parameter: str) -> NDArray[np.float64]:
    """Convert a numeric input to a nonempty finite one-dimensional array."""
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


def _as_sem_vector(
    values: ArrayLike,
    n_persons: int,
    parameter: str,
) -> NDArray[np.float64]:
    """Validate and broadcast person-level standard errors."""
    result = _as_finite_vector(values, parameter)
    if result.size == 1:
        result = np.full(n_persons, result.item(), dtype=np.float64)
    elif result.size != n_persons:
        raise MirtValidationError(
            f"{parameter} must be scalar or match the number of persons",
            parameter=parameter,
            value=result.size,
            expected=f"1 or {n_persons}",
        )
    if np.any(result <= 0):
        raise MirtValidationError(
            f"{parameter} must contain only positive values",
            parameter=parameter,
        )
    return result


def _validate_alpha(alpha: float) -> float:
    value = _validate_finite_scalar(alpha, "alpha")
    if not 0.0 < value < 1.0:
        raise MirtValidationError(
            "alpha must be between 0 and 1",
            parameter="alpha",
            value=alpha,
            expected="0 < alpha < 1",
        )
    return value


def _validate_reliability(reliability: float | None) -> float | None:
    if reliability is None:
        return None
    value = _validate_finite_scalar(reliability, "reliability")
    if not 0.0 <= value < 1.0:
        raise MirtValidationError(
            "reliability must be at least 0 and less than 1",
            parameter="reliability",
            value=reliability,
            expected="0 <= reliability < 1",
        )
    return value


def _validate_positive_scalar(value: float, parameter: str) -> float:
    result = _validate_finite_scalar(value, parameter)
    if result <= 0.0:
        raise MirtValidationError(
            f"{parameter} must be positive",
            parameter=parameter,
            value=value,
            expected="> 0",
        )
    return result


def _validate_finite_scalar(value: float, parameter: str) -> float:
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


def _validate_method(method: str) -> None:
    if method not in _RCI_METHODS:
        raise MirtValidationError(
            "Unknown RCI method",
            parameter="method",
            value=method,
            expected=", ".join(_RCI_METHODS),
        )


def _validate_boolean(value: bool, parameter: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise MirtValidationError(
            f"{parameter} must be a boolean",
            parameter=parameter,
            value=value,
        )
    return bool(value)
