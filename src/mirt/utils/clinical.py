"""Clinical change statistics for IRT score comparisons."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError
from mirt.utils._validation import (
    as_finite_vector,
    validate_alpha,
    validate_finite_scalar,
    validate_positive_scalar,
)

RCIMethod = Literal["jacobson", "hageman", "iverson"]

_RCI_METHODS = ("jacobson", "hageman", "iverson")


@dataclass
class RCIResult:
    """Result of a Reliable Change Index computation.

    ``se_diff`` retains the historical scalar summary (the mean standard
    error), while ``se_diff_by_person`` contains the denominator actually
    used for each RCI. This preserves compatibility while making
    heterogeneous measurement precision visible. ``change`` is the observed
    difference ``theta_post - theta_pre``; the interval ``ci_lower`` to
    ``ci_upper`` surrounds the change estimate the index tests, which for
    ``method="hageman"`` is the reliability-weighted difference rather than
    the observed one.
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
    reliability_post: float | None = None,
) -> RCIResult:
    """Compute person-specific Reliable Change Indices.

    Parameters
    ----------
    theta_pre, theta_post : array-like
        Paired pre- and post-treatment scores.
    sem_pre, sem_post : array-like or float, optional
        Standard errors of measurement. Scalars are broadcast to every
        person. If ``sem_pre`` is omitted, it is ``sd_theta *
        sqrt(1 - reliability)``. If ``sem_post`` is omitted, it is derived
        the same way from ``reliability_post`` when given, and is otherwise
        ``sem_pre``.
    reliability : float, optional
        Reliability in ``[0, 1)`` of the pre-test scores, used when
        ``sem_pre`` is omitted and required by ``method="hageman"``.
    sd_theta : float
        Positive score standard deviation used with ``reliability``.
    alpha : float
        Two-sided significance level in ``(0, 1)``.
    method : {"jacobson", "hageman", "iverson"}
        Reliable change index:

        - ``"jacobson"`` (Jacobson & Truax, 1991): ``(post - pre) /
          (sqrt(2) * sem_pre)``. Only the pre-test error is used, so
          ``sem_post`` and ``reliability_post`` are ignored with a warning.
        - ``"iverson"`` (Iverson, Lovell & Collins, 2003): ``(post - pre) /
          sqrt(sem_pre**2 + sem_post**2)``, combining both measurement
          errors as R's ``mirt::RCI`` does.
        - ``"hageman"`` (Hageman & Arrindell, 1993, RC_ID):
          ``[(post - pre) * r_DD + (M_post - M_pre) * (1 - r_DD)] /
          sqrt(r_DD * (sem_pre**2 + sem_post**2))``, which regresses each
          difference toward the mean change by the reliability of the
          difference scores ``r_DD = (s_pre**2 r_pre + s_post**2 r_post -
          2 s_pre s_post r_pp) / (s_pre**2 + s_post**2 - 2 s_pre s_post
          r_pp)``. The means ``M``, standard deviations ``s`` and pre-post
          correlation ``r_pp`` come from the supplied scores, ``r_pre`` is
          ``reliability`` and ``r_post`` is ``reliability_post`` (default
          ``reliability``).
    higher_is_better : bool
        Whether positive score change represents improvement.
    reliability_post : float, optional
        Reliability in ``[0, 1)`` of the post-test scores. Defaults to
        ``reliability``. Not used by ``method="jacobson"``.

    Returns
    -------
    RCIResult
        Person-level RCIs, uncertainty intervals, and classifications.

    Raises
    ------
    MirtValidationError
        If an argument is invalid, ``method="hageman"`` lacks
        ``reliability``, or the difference scores have no positive
        reliability.

    Warns
    -----
    UserWarning
        If ``method="jacobson"`` receives ``sem_post`` or
        ``reliability_post``, which it does not use.

    References
    ----------
    Jacobson, N. S., & Truax, P. (1991). Clinical significance: A statistical
    approach to defining meaningful change in psychotherapy research.
    *Journal of Consulting and Clinical Psychology, 59*(1), 12-19.

    Hageman, W. J. J. M., & Arrindell, W. A. (1993). A further refinement of
    the reliable change (RC) index by improving the pre-post difference
    score: Introducing RC_ID. *Behaviour Research and Therapy, 31*(7),
    693-700.

    Iverson, G. L., Lovell, M. R., & Collins, M. W. (2003). Interpreting
    change on ImPACT following sport concussion. *The Clinical
    Neuropsychologist, 17*(4), 460-467.
    """
    from scipy import stats

    pre = as_finite_vector(theta_pre, "theta_pre")
    post = as_finite_vector(theta_post, "theta_post")
    if pre.size != post.size:
        raise MirtValidationError(
            "theta_pre and theta_post must have the same length",
            parameter="theta_post",
            value=post.size,
            expected=str(pre.size),
        )

    alpha_value = validate_alpha(alpha)
    reliability_value = _validate_reliability(reliability)
    post_reliability_value = _validate_reliability(reliability_post, "reliability_post")
    sd_value = validate_positive_scalar(sd_theta, "sd_theta")
    _validate_method(method)
    orientation = _validate_boolean(higher_is_better, "higher_is_better")

    if method == "jacobson":
        ignored = [
            name
            for name, value in (
                ("sem_post", sem_post),
                ("reliability_post", reliability_post),
            )
            if value is not None
        ]
        if ignored:
            warnings.warn(
                f"method='jacobson' uses sqrt(2) * sem_pre and ignores "
                f"{' and '.join(ignored)}; use method='iverson' to combine "
                "pre- and post-test measurement errors",
                UserWarning,
                stacklevel=2,
            )

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

    if sem_post is not None:
        post_sem = _as_sem_vector(sem_post, post.size, "sem_post")
    elif post_reliability_value is not None:
        post_sem = np.full(
            post.size,
            sd_value * np.sqrt(1.0 - post_reliability_value),
            dtype=np.float64,
        )
    else:
        post_sem = pre_sem.copy()

    change = post - pre
    estimate = change
    if method == "jacobson":
        se_by_person = np.sqrt(2.0) * pre_sem
    elif method == "iverson":
        se_by_person = np.sqrt(pre_sem**2 + post_sem**2)
    else:
        if reliability_value is None:
            raise MirtValidationError(
                "method='hageman' requires the pre-test reliability",
                parameter="reliability",
                expected="0 <= reliability < 1",
            )
        difference_reliability = _difference_reliability(
            pre,
            post,
            reliability_value,
            reliability_value
            if post_reliability_value is None
            else post_reliability_value,
        )
        mean_change = float(np.mean(post) - np.mean(pre))
        estimate = change * difference_reliability + mean_change * (
            1.0 - difference_reliability
        )
        se_by_person = np.sqrt(difference_reliability * (pre_sem**2 + post_sem**2))

    rci = estimate / se_by_person
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
        ci_lower=estimate - margin,
        ci_upper=estimate + margin,
        method=method,
        higher_is_better=orientation,
    )


def _difference_reliability(
    pre: NDArray[np.float64],
    post: NDArray[np.float64],
    reliability_pre: float,
    reliability_post: float,
) -> float:
    """Reliability of the pre-post difference scores (Hageman & Arrindell)."""
    if pre.size < 3:
        raise MirtValidationError(
            "method='hageman' needs at least three paired scores to estimate "
            "the reliability of the differences",
            parameter="theta_pre",
            value=pre.size,
            expected=">= 3",
        )
    sd_pre = float(np.std(pre, ddof=1))
    sd_post = float(np.std(post, ddof=1))
    if sd_pre == 0.0 or sd_post == 0.0:
        raise MirtValidationError(
            "method='hageman' needs pre- and post-test scores that vary",
            parameter="theta_pre",
        )
    correlation = float(np.corrcoef(pre, post)[0, 1])
    cross = 2.0 * sd_pre * sd_post * correlation
    numerator = sd_pre**2 * reliability_pre + sd_post**2 * reliability_post - cross
    denominator = sd_pre**2 + sd_post**2 - cross
    if not denominator > 0.0 or not numerator > 0.0:
        raise MirtValidationError(
            "the pre-post difference scores have no positive reliability, so "
            "method='hageman' is undefined",
            parameter="reliability",
            value=reliability_pre,
        )
    return min(numerator / denominator, 1.0)


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
    cutoff_value = validate_finite_scalar(cutoff, "cutoff")
    orientation = _validate_boolean(higher_is_better, "higher_is_better")
    pre = as_finite_vector(theta_pre, "theta_pre")
    post = as_finite_vector(theta_post, "theta_post")

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


def _as_sem_vector(
    values: ArrayLike,
    n_persons: int,
    parameter: str,
) -> NDArray[np.float64]:
    """Validate and broadcast person-level standard errors."""
    result = as_finite_vector(values, parameter)
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


def _validate_reliability(
    reliability: float | None, parameter: str = "reliability"
) -> float | None:
    if reliability is None:
        return None
    value = validate_finite_scalar(reliability, parameter)
    if not 0.0 <= value < 1.0:
        raise MirtValidationError(
            f"{parameter} must be at least 0 and less than 1",
            parameter=parameter,
            value=reliability,
            expected=f"0 <= {parameter} < 1",
        )
    return value


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
