"""Person scoring methods for IRT models."""

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mirt.results.score_result import ScoreResult
from mirt.scoring.eap import EAPScorer
from mirt.scoring.map import MAPScorer
from mirt.scoring.ml import MLScorer

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


def fscores(
    model_or_result: Union["BaseItemModel", "FitResult"],
    responses: NDArray[np.int_],
    method: Literal["EAP", "MAP", "ML"] = "EAP",
    n_quadpts: int = 49,
    prior_mean: Optional[NDArray[np.float64]] = None,
    prior_cov: Optional[NDArray[np.float64]] = None,
    person_ids: Optional[list] = None,
) -> ScoreResult:
    """Estimate person abilities (theta scores).

    This function computes ability estimates for each respondent using
    various scoring methods.

    Parameters
    ----------
    model_or_result : BaseItemModel or FitResult
        Either a fitted IRT model or a FitResult from model fitting.
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing values coded as -1.
    method : {'EAP', 'MAP', 'ML'}, default='EAP'
        Scoring method:
        - 'EAP': Expected A Posteriori (posterior mean)
        - 'MAP': Maximum A Posteriori (posterior mode)
        - 'ML': Maximum Likelihood (no prior)
    n_quadpts : int, default=49
        Number of quadrature points for EAP scoring.
    prior_mean : ndarray, optional
        Mean of the prior distribution. Default: zeros.
    prior_cov : ndarray, optional
        Covariance of the prior. Default: identity.
    person_ids : list, optional
        Identifiers for each person in the output.

    Returns
    -------
    ScoreResult
        Object containing theta estimates and standard errors.

    Examples
    --------
    >>> result = mirt.fit_mirt(responses, model='2PL')
    >>> scores = mirt.fscores(result, responses, method='EAP')
    >>> print(scores.theta)
    >>> print(scores.to_dataframe())

    >>> # Using MAP scoring
    >>> scores_map = mirt.fscores(result, responses, method='MAP')

    >>> # Maximum likelihood without prior
    >>> scores_ml = mirt.fscores(result, responses, method='ML')
    """
    # Extract model from FitResult if needed
    from mirt.results.fit_result import FitResult

    if isinstance(model_or_result, FitResult):
        model = model_or_result.model
    else:
        model = model_or_result

    if not model.is_fitted:
        raise ValueError("Model must be fitted before scoring")

    # Validate responses
    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError(f"responses must be 2D, got {responses.ndim}D")
    if responses.shape[1] != model.n_items:
        raise ValueError(
            f"responses has {responses.shape[1]} items, expected {model.n_items}"
        )

    # Select scorer
    if method == "EAP":
        scorer = EAPScorer(
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
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    # Compute scores
    result = scorer.score(model, responses)
    result.person_ids = person_ids

    return result


__all__ = [
    "fscores",
    "EAPScorer",
    "MAPScorer",
    "MLScorer",
]
