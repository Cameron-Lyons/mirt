from typing import TYPE_CHECKING, Any, Literal

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
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    method: Literal["EAP", "MAP", "ML"] = "EAP",
    n_quadpts: int = 49,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    person_ids: list[Any] | None = None,
) -> ScoreResult:
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
    elif method == "MAP":
        scorer = MAPScorer(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
    elif method == "ML":
        scorer = MLScorer()
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    result = scorer.score(model, responses)
    result.person_ids = person_ids

    return result


__all__ = [
    "fscores",
    "EAPScorer",
    "MAPScorer",
    "MLScorer",
]
