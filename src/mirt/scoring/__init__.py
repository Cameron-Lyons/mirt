from __future__ import annotations

import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from mirt.models.base import BaseItemModel
    from mirt.multigroup.results import MultigroupFitResult
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
    model_or_result: BaseItemModel | FitResult | MultigroupFitResult,
    responses: ArrayLike,
    method: Literal["EAP", "MAP", "ML", "WLE", "EAPsum"] = "EAP",
    n_quadpts: int | None = None,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    person_ids: list[Any] | NDArray[Any] | None = None,
    bounds: tuple[float, float] = (-6.0, 6.0),
    n_jobs: int = 1,
    batch_size: int | None = None,
    *,
    groups: ArrayLike | None = None,
) -> ScoreResult:
    """Compute ability (theta) estimates for respondents.

    This is the main function for estimating latent trait scores from
    response data using a fitted IRT model.

    Parameters
    ----------
    model_or_result : BaseItemModel | FitResult | MultigroupFitResult
        A fitted IRT model, a FitResult from fit_mirt(), or a
        MultigroupFitResult from fit_multigroup(), which scores each row
        under its group's items and latent distribution (see
        :meth:`~mirt.multigroup.MultigroupFitResult.fscores`).
    responses : array-like of shape (n_persons, n_items)
        Response matrix, as an array or a pandas or polars DataFrame.
        Negative codes, ``NaN`` and the nulls of nullable DataFrame columns
        denote missing responses, as in :func:`mirt.fit_mirt`. Array columns
        are matched to the model's items by position. DataFrame columns are
        matched by name when the model's items have names, as after fitting
        a DataFrame: columns may then come in any order, and columns that
        are not items, such as respondent identifiers, are ignored (see
        :func:`mirt.utils.data.align_response_columns`).
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
        Bifactor models, such as :func:`mirt.bfactor` results, are scored by
        dimension reduction on one two-dimensional (general by specific) grid
        per specific factor when their prior keeps the specific factors
        independent given the general factor, as the default prior does, and
        then default to 49 points for any number of factors.
        EAPsum is unidimensional and defaults to 49 points.
    prior_mean : ndarray, optional
        Prior mean for Bayesian methods. Defaults to the ``latent_mean`` of a
        ``FitResult`` when it has one, and to zero otherwise.
    prior_cov : ndarray, optional
        Prior covariance for Bayesian methods. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to the
        identity otherwise.
    person_ids : list or 1-D ndarray, optional
        Identifiers for each person in the output, one per response row.
    bounds : tuple of float, default=(-6.0, 6.0)
        Bounds for theta estimation used by MAP, ML, and WLE. A pattern
        whose observed responses are all in the lowest category, or all in
        each item's highest category, has no finite ML estimate; ML reports
        it at the lower or upper bound in every coordinate with an infinite
        standard error, for every model family. For multidimensional models,
        ML likewise reports a coordinate whose likelihood still increases at
        a bound, such as a factor whose items are all answered in their
        lowest or highest category, at that bound with an infinite standard
        error. R's mirt reports ``-Inf``/``Inf`` with missing standard errors
        instead.
    n_jobs : int, default=1
        Number of response patterns to optimize in parallel for MAP, ML, and
        WLE scoring. ``-1`` uses all available CPU cores. Unidimensional
        models and built-in multidimensional models score every pattern in
        one vectorized optimization that does not use a thread pool.
    batch_size : int, optional
        Maximum response rows per EAP likelihood batch. Repetition-heavy data
        are compressed to unique patterns when beneficial and expanded back to
        respondents. The default chooses a memory-bounded size automatically.
        Other scoring methods ignore this option.
    groups : array-like of shape (n_persons,), optional
        Group label of each row, required for a ``MultigroupFitResult`` and
        rejected otherwise. The groups' fitted latent distributions are the
        priors, so ``prior_mean`` and ``prior_cov`` must be omitted then.

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
        If the model is not fitted or ``method`` is unknown.
    MirtDataError
        If ``responses`` has the wrong shape or invalid response codes, or is
        a DataFrame without columns for some of the model's named items.
        ``MirtDataError`` is a ``ValueError``.
    MirtValidationError
        If ``person_ids`` does not contain one identifier per response row.

    Notes
    -----
    WLE for multidimensional models maximizes the likelihood weighted by the
    square root of the determinant of the observed items' Fisher information
    matrix, as R's mirt does, and reports standard errors from the inverse of
    that matrix.

    Examples
    --------
    >>> from mirt import fit_mirt, fscores
    >>> result = fit_mirt(data, model="2PL")
    >>> scores = fscores(result, data, method="EAP")
    >>> print(scores.theta[:5])
    """
    from mirt.exceptions import MirtValidationError
    from mirt.multigroup.results import MultigroupFitResult
    from mirt.results._common import resolve_latent_prior, validate_person_ids
    from mirt.scoring._common import validate_scoring_responses

    if isinstance(model_or_result, MultigroupFitResult):
        if groups is None or prior_mean is not None or prior_cov is not None:
            raise MirtValidationError(
                "multigroup results score each row under its group's fitted "
                "latent distribution: pass groups and omit prior_mean and "
                "prior_cov",
                parameter="groups" if groups is None else "prior_mean",
            )
        return model_or_result.fscores(
            responses,
            groups,
            method=method,
            n_quadpts=n_quadpts,
            person_ids=person_ids,
            bounds=bounds,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )
    if groups is not None:
        raise MirtValidationError(
            "groups applies only to multigroup results", parameter="groups"
        )

    model, prior_mean, prior_cov = resolve_latent_prior(
        model_or_result, prior_mean, prior_cov
    )

    if not model.is_fitted:
        raise ValueError("Model must be fitted before scoring")

    responses = validate_scoring_responses(model, responses)

    # Check identifiers before scoring so a mismatch fails without work.
    person_ids = validate_person_ids(person_ids, responses.shape[0], row_label="score")

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
