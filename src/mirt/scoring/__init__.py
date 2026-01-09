from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.results.score_result import ScoreResult
from mirt.scoring.eap import EAPScorer
from mirt.scoring.eapsum import EAPSumScorer, eapsum, sum_score_to_theta
from mirt.scoring.map import MAPScorer
from mirt.scoring.ml import MLScorer
from mirt.scoring.wle import WLEScorer

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def fscores(
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    method: Literal["EAP", "MAP", "ML", "WLE", "EAPsum"] = "EAP",
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    person_ids: list[Any] | None = None,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> ScoreResult:
    """Compute ability (theta) estimates for respondents.

    This is the main function for estimating latent trait scores from
    response data using a fitted IRT model.

    Parameters
    ----------
    model_or_result : BaseItemModel | FitResult
        A fitted IRT model or a FitResult from fit_mirt().
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses should be coded as -1.
    method : {"EAP", "MAP", "ML", "WLE", "EAPsum"}, default="EAP"
        Scoring method to use:

        - "EAP": Expected A Posteriori (Bayesian mean)
        - "MAP": Maximum A Posteriori (Bayesian mode)
        - "ML": Maximum Likelihood
        - "WLE": Weighted Likelihood Estimation (Warm's estimator)
        - "EAPsum": EAP based on sum scores (Lord-Wingersky)

    n_quadpts : int, default=49
        Number of quadrature points for EAP/EAPsum methods.
    prior_mean : ndarray, optional
        Prior mean for Bayesian methods. Default is 0.
    prior_cov : ndarray, optional
        Prior covariance for Bayesian methods. Default is identity.
    person_ids : list, optional
        Identifiers for each person in the output.
    bounds : tuple of float, default=(-6.0, 6.0)
        Bounds for theta estimation (used by WLE).

    Returns
    -------
    ScoreResult
        Object containing:

        - theta: Ability estimates, shape (n_persons, n_factors)
        - standard_error: Standard errors, shape (n_persons, n_factors)
        - person_ids: Person identifiers if provided

    Raises
    ------
    ValueError
        If model is not fitted or responses shape is invalid.

    Examples
    --------
    >>> from mirt import fit_mirt, fscores
    >>> result = fit_mirt(data, model="2PL")
    >>> scores = fscores(result, data, method="EAP")
    >>> print(scores.theta[:5])
    """
    from mirt.results.fit_result import FitResult

    if isinstance(model_or_result, FitResult):
        model = model_or_result.model
    else:
        model = model_or_result

    if not model.is_fitted:
        raise ValueError("Model must be fitted before scoring")

    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError(f"responses must be 2D, got {responses.ndim}D")
    if responses.shape[1] != model.n_items:
        raise ValueError(
            f"responses has {responses.shape[1]} items, expected {model.n_items}"
        )

    if method == "EAP":
        scorer = EAPScorer(
            n_quadpts=n_quadpts,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
    elif method == "EAPsum":
        scorer = EAPSumScorer(
            n_quadpts=n_quadpts,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
    elif method == "MAP":
        scorer = MAPScorer(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
    elif method == "ML":
        scorer = MLScorer()
    elif method == "WLE":
        scorer = WLEScorer(bounds=bounds)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    result = scorer.score(model, responses)
    result.person_ids = person_ids

    return result


__all__ = [
    "fscores",
    "EAPScorer",
    "EAPSumScorer",
    "MAPScorer",
    "MLScorer",
    "WLEScorer",
    "eapsum",
    "sum_score_to_theta",
]
