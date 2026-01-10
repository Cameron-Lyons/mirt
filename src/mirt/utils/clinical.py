"""Clinical utility functions for IRT models.

Provides functions for computing clinically meaningful change
indices and related statistics.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class RCIResult:
    """Result of Reliable Change Index computation.

    Attributes
    ----------
    rci : NDArray[np.float64]
        RCI values for each person.
    significant : NDArray[np.bool_]
        Whether change is statistically significant.
    direction : NDArray[np.str_]
        Direction of change ("improved", "declined", "unchanged").
    se_diff : float
        Standard error of difference scores.
    critical_value : float
        Critical value used for significance.
    """

    rci: NDArray[np.float64]
    significant: NDArray[np.bool_]
    direction: NDArray[np.str_]
    se_diff: float
    critical_value: float


def RCI(
    theta_pre: NDArray[np.float64],
    theta_post: NDArray[np.float64],
    sem_pre: NDArray[np.float64] | float | None = None,
    sem_post: NDArray[np.float64] | float | None = None,
    reliability: float | None = None,
    sd_theta: float = 1.0,
    alpha: float = 0.05,
    method: str = "jacobson",
) -> RCIResult:
    """Compute Reliable Change Index for pre-post comparisons.

    The RCI indicates whether observed change exceeds measurement error,
    suggesting true (clinically meaningful) change.

    Parameters
    ----------
    theta_pre : NDArray[np.float64]
        Pre-treatment theta estimates. Shape: (n_persons,).
    theta_post : NDArray[np.float64]
        Post-treatment theta estimates. Shape: (n_persons,).
    sem_pre : NDArray or float, optional
        Standard error of measurement at pre-test.
        If None, computed from reliability and sd_theta.
    sem_post : NDArray or float, optional
        Standard error of measurement at post-test.
        If None, uses sem_pre.
    reliability : float, optional
        Test reliability (required if sem_pre not provided).
    sd_theta : float
        Standard deviation of theta. Default 1.0.
    alpha : float
        Significance level. Default 0.05.
    method : str
        Method for computing SE of difference:
        - "jacobson": Jacobson & Truax (1991) method
        - "hageman": Hageman & Arrindell (1999) method
        - "iverson": Uses pooled SEM

    Returns
    -------
    RCIResult
        Container with RCI values and significance indicators.

    Examples
    --------
    >>> # Pre-post comparison with known reliability
    >>> rci_result = RCI(theta_pre, theta_post, reliability=0.85)
    >>> print(f"Significant changes: {np.sum(rci_result.significant)}")
    >>> print(f"Improved: {np.sum(rci_result.direction == 'improved')}")

    Notes
    -----
    The Jacobson-Truax RCI is computed as:
        RCI = (X2 - X1) / SE_diff

    where SE_diff = sqrt(2) * SEM for the Jacobson method.

    An RCI > 1.96 (for alpha=0.05) indicates reliable improvement,
    while RCI < -1.96 indicates reliable decline.

    References
    ----------
    Jacobson, N. S., & Truax, P. (1991). Clinical significance: A statistical
    approach to defining meaningful change in psychotherapy research.
    Journal of Consulting and Clinical Psychology, 59(1), 12-19.
    """
    from scipy import stats

    theta_pre = np.atleast_1d(theta_pre).ravel()
    theta_post = np.atleast_1d(theta_post).ravel()

    if len(theta_pre) != len(theta_post):
        raise ValueError("theta_pre and theta_post must have same length")

    if sem_pre is None:
        if reliability is None:
            raise ValueError("Must provide either sem_pre or reliability")
        sem_pre = sd_theta * np.sqrt(1 - reliability)

    if sem_post is None:
        sem_post = sem_pre

    sem_pre = np.atleast_1d(sem_pre)
    sem_post = np.atleast_1d(sem_post)

    if len(sem_pre) == 1:
        sem_pre = np.full(len(theta_pre), sem_pre[0])
    if len(sem_post) == 1:
        sem_post = np.full(len(theta_post), sem_post[0])

    if method == "jacobson":
        se_diff = np.sqrt(2) * np.mean(sem_pre)
    elif method == "hageman":
        se_diff = np.sqrt(sem_pre**2 + sem_post**2)
        se_diff = np.mean(se_diff)
    elif method == "iverson":
        pooled_sem = np.sqrt((sem_pre**2 + sem_post**2) / 2)
        se_diff = np.sqrt(2) * np.mean(pooled_sem)
    else:
        raise ValueError(f"Unknown method: {method}")

    diff = theta_post - theta_pre
    rci = diff / se_diff

    critical_value = stats.norm.ppf(1 - alpha / 2)

    significant = np.abs(rci) > critical_value

    direction = np.where(
        rci > critical_value,
        "improved",
        np.where(rci < -critical_value, "declined", "unchanged"),
    )

    return RCIResult(
        rci=rci,
        significant=significant,
        direction=direction,
        se_diff=float(se_diff) if np.isscalar(se_diff) else float(np.mean(se_diff)),
        critical_value=float(critical_value),
    )


def clinical_significance(
    theta_pre: NDArray[np.float64],
    theta_post: NDArray[np.float64],
    cutoff: float,
    reliability: float,
    sd_theta: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, NDArray[np.bool_]]:
    """Classify individuals by clinical significance criteria.

    Combines RCI with cutoff-based classification following
    Jacobson & Truax (1991) criteria.

    Parameters
    ----------
    theta_pre : NDArray[np.float64]
        Pre-treatment theta estimates.
    theta_post : NDArray[np.float64]
        Post-treatment theta estimates.
    cutoff : float
        Clinical cutoff score (e.g., between clinical and normative).
    reliability : float
        Test reliability coefficient.
    sd_theta : float
        Standard deviation of theta. Default 1.0.
    alpha : float
        Significance level for RCI. Default 0.05.

    Returns
    -------
    dict
        Classification results with keys:
        - "recovered": Reliable change AND crossed cutoff to normative
        - "improved": Reliable change but still in clinical range
        - "unchanged": No reliable change
        - "deteriorated": Reliable negative change
    """
    rci_result = RCI(
        theta_pre, theta_post, reliability=reliability, sd_theta=sd_theta, alpha=alpha
    )

    theta_pre = np.atleast_1d(theta_pre).ravel()
    theta_post = np.atleast_1d(theta_post).ravel()

    reliable_improvement = rci_result.rci > rci_result.critical_value
    reliable_decline = rci_result.rci < -rci_result.critical_value

    crossed_to_normal = (theta_pre < cutoff) & (theta_post >= cutoff)

    recovered = reliable_improvement & crossed_to_normal
    improved = reliable_improvement & ~crossed_to_normal
    unchanged = ~reliable_improvement & ~reliable_decline
    deteriorated = reliable_decline

    return {
        "recovered": recovered,
        "improved": improved,
        "unchanged": unchanged,
        "deteriorated": deteriorated,
    }
