from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult
    from mirt.results.score_result import ScoreResult


_LAZY_IMPORTS = {
    "EAPScorer": ("mirt.scoring.eap", "EAPScorer"),
    "EAPSumScorer": ("mirt.scoring._eapsum", "EAPSumScorer"),
    "MAPScorer": ("mirt.scoring.map", "MAPScorer"),
    "MLScorer": ("mirt.scoring.ml", "MLScorer"),
    "SumScoreTable": ("mirt.scoring._eapsum", "SumScoreTable"),
    "WLEScorer": ("mirt.scoring.wle", "WLEScorer"),
    "ability_posterior": ("mirt.scoring.eap", "ability_posterior"),
    "eapsum": ("mirt.scoring._eapsum", "eapsum"),
    "eapsum_table": ("mirt.scoring._eapsum", "eapsum_table"),
    "sum_score_to_theta": ("mirt.scoring._eapsum", "sum_score_to_theta"),
}


def fscores(
    model_or_result: BaseItemModel | FitResult,
    responses: NDArray[np.int_],
    method: Literal["EAP", "MAP", "ML", "WLE", "EAPsum"] = "EAP",
    n_quadpts: int | None = None,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    person_ids: list[Any] | NDArray[Any] | None = None,
    bounds: tuple[float, float] = (-6.0, 6.0),
    n_jobs: int = 1,
    batch_size: int | None = None,
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

    n_quadpts : int, optional
        Number of quadrature points per latent dimension for EAP/EAPsum
        methods. The default depends on the number of factors because EAP
        integrates over a full tensor-product grid: 49 points for one or two
        factors, 21 for three, 9 for four, 7 for five, and 5 for six or more.
        Coarse grids are less accurate when posteriors are concentrated, so
        pass a larger value when precision matters more than run time.
        EAPsum is unidimensional and defaults to 49 points.
    prior_mean : ndarray, optional
        Prior mean for Bayesian methods. Default is 0.
    prior_cov : ndarray, optional
        Prior covariance for Bayesian methods. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to the
        identity otherwise.
    person_ids : list or 1-D ndarray, optional
        Identifiers for each person in the output, one per response row.
    bounds : tuple of float, default=(-6.0, 6.0)
        Bounds for theta estimation used by MAP, ML, and WLE.
    n_jobs : int, default=1
        Number of response patterns to optimize in parallel for MAP, ML, and
        WLE scoring. ``-1`` uses all available CPU cores. Unidimensional
        models, and MAP for built-in multidimensional models, score every
        pattern in one vectorized optimization that does not use a thread
        pool.
    batch_size : int, optional
        Maximum response rows per EAP likelihood batch. Repetition-heavy data
        are compressed to unique patterns when beneficial and expanded back to
        respondents. The default chooses a memory-bounded size automatically.
        Other scoring methods ignore this option.

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
    MirtValidationError
        If ``person_ids`` does not contain one identifier per response row.

    Examples
    --------
    >>> from mirt import fit_mirt, fscores
    >>> result = fit_mirt(data, model="2PL")
    >>> scores = fscores(result, data, method="EAP")
    >>> print(scores.theta[:5])
    """
    import numpy as np

    from mirt.results._common import resolve_latent_prior

    model, prior_mean, prior_cov = resolve_latent_prior(
        model_or_result, prior_mean, prior_cov
    )

    if not model.is_fitted:
        raise ValueError("Model must be fitted before scoring")

    responses = np.asarray(responses)
    if responses.ndim != 2:
        raise ValueError(f"responses must be 2D, got {responses.ndim}D")
    if responses.shape[1] != model.n_items:
        raise ValueError(
            f"responses has {responses.shape[1]} items, expected {model.n_items}"
        )
    person_ids = _validated_person_ids(person_ids, responses.shape[0])

    if method == "EAP":
        from mirt.scoring.eap import EAPScorer

        scorer = EAPScorer(
            n_quadpts=n_quadpts,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            batch_size=batch_size,
        )
    elif method == "EAPsum":
        from mirt.scoring._eapsum import EAPSumScorer

        scorer = EAPSumScorer(
            n_quadpts=49 if n_quadpts is None else n_quadpts,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
    elif method == "MAP":
        from mirt.scoring.map import MAPScorer

        scorer = MAPScorer(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            theta_bounds=bounds,
            n_jobs=n_jobs,
        )
    elif method == "ML":
        from mirt.scoring.ml import MLScorer

        scorer = MLScorer(theta_bounds=bounds, n_jobs=n_jobs)
    elif method == "WLE":
        from mirt.scoring.wle import WLEScorer

        scorer = WLEScorer(bounds=bounds, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    result = scorer.score(model, responses)
    if person_ids is None:
        return result
    return dataclasses.replace(result, person_ids=person_ids)


def _validated_person_ids(
    person_ids: list[Any] | NDArray[Any] | None,
    n_persons: int,
) -> list[Any] | None:
    """Check identifiers before scoring so a mismatch fails without work."""
    if person_ids is None:
        return None

    import numpy as np

    from mirt.exceptions import MirtValidationError

    if isinstance(person_ids, np.ndarray) and person_ids.ndim != 1:
        raise MirtValidationError(
            "person_ids must be one-dimensional",
            parameter="person_ids",
            value=person_ids.shape,
            expected=f"({n_persons},)",
        )
    resolved = list(person_ids)
    if len(resolved) != n_persons:
        raise MirtValidationError(
            "person_ids must contain one identifier per score row",
            parameter="person_ids",
            value=len(resolved),
            expected=str(n_persons),
        )
    return resolved


__all__ = [
    "fscores",
    "ability_posterior",
    "EAPScorer",
    "EAPSumScorer",
    "MAPScorer",
    "MLScorer",
    "WLEScorer",
    "SumScoreTable",
    "eapsum",
    "eapsum_table",
    "sum_score_to_theta",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.scoring' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
