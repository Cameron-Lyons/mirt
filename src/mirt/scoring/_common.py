"""Shared helper utilities for scoring implementations."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError
from mirt.utils.data import _missing_coded_responses, align_response_columns

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

# Response patterns optimized together by row-batched scorers; bounds the
# temporary likelihood arrays without giving up vectorization.
BATCHED_SCORING_CHUNK_SIZE = 8192


def _observed_item_masks(
    model: BaseItemModel,
    n_theta: int,
    observed_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_] | None, NDArray[np.intp]]:
    """Return the per-row mask, if any, and the items observed by any row."""
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    if observed_mask.shape == (model.n_items,):
        return None, np.flatnonzero(observed_mask)
    if observed_mask.shape == (n_theta, model.n_items):
        return observed_mask, np.flatnonzero(observed_mask.any(axis=0))
    raise ValueError(
        f"observed_mask must have shape ({model.n_items},) or "
        f"({n_theta}, {model.n_items}), got {observed_mask.shape}"
    )


def observed_test_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return test information contributed by observed response items.

    ``observed_mask`` is either one item mask of shape ``(n_items,)`` shared
    by every theta row, or an ``(n_theta, n_items)`` mask that pairs each
    theta row with its own observed items.
    """
    n_theta = theta.shape[0]
    row_mask, observed_items = _observed_item_masks(model, n_theta, observed_mask)

    if observed_items.size == 0:
        return np.zeros(n_theta, dtype=np.float64)

    # Dichotomous models expose an itemwise information matrix, so use their
    # vectorized path. Polytomous models expose only total information in bulk.
    if not model.is_polytomous:
        item_information = np.asarray(model.information(theta), dtype=np.float64)
        if item_information.shape == (n_theta, model.n_items):
            if row_mask is None:
                return item_information[:, observed_items].sum(axis=1)
            return np.where(row_mask, item_information, 0.0).sum(axis=1)

    # Accumulate item by item so a row-specific mask adds exactly the same
    # terms, in the same order, as scoring that row with its own 1-D mask.
    information = np.zeros(n_theta, dtype=np.float64)
    for item_idx in observed_items:
        item_information = np.asarray(
            model.information(theta, int(item_idx)), dtype=np.float64
        ).reshape(n_theta, -1)
        item_total = item_information.sum(axis=1)
        if row_mask is not None:
            item_total = np.where(row_mask[:, item_idx], item_total, 0.0)
        information += item_total

    return information


def _rank_one_slopes(model: BaseItemModel) -> NDArray[np.float64] | None:
    """Return slope rows of an unmodified model with rank-one item matrices.

    The compensatory logistic, graded and generalized partial credit models
    give every item one ability gradient direction, its slope vector ``a``,
    so the item Fisher matrix is ``I_j a a' / ||a||^2`` for the scalar item
    information ``I_j`` (the matrix trace). Returns None for other models.
    """
    from mirt._model_defaults import uses_builtin_model_hooks, uses_original_model_hook
    from mirt.models.bifactor import BifactorModel
    from mirt.models.dichotomous import TwoParameterLogistic
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel

    model_type = type(model)
    if (
        model_type
        not in (
            TwoParameterLogistic,
            MultidimensionalModel,
            BifactorModel,
            GradedResponseModel,
            GeneralizedPartialCredit,
        )
        or not uses_builtin_model_hooks(model)
        or not uses_original_model_hook(model, "information")
    ):
        return None
    if model_type is BifactorModel:
        slopes = model.get_loading_matrix()
    elif model_type is MultidimensionalModel:
        slopes = model.parameters["slopes"]
    else:
        slopes = model.parameters["discrimination"]
    return np.asarray(slopes, dtype=np.float64).reshape(model.n_items, model.n_factors)


def _item_probability_columns(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Return one item's category probabilities with shape ``(n_theta, C)``."""
    probability = np.asarray(model.probability(theta, item_idx), dtype=np.float64)
    if probability.ndim == 1 or probability.shape[-1:] == (1,):
        correct = probability.reshape(theta.shape[0])
        return np.column_stack((1.0 - correct, correct))
    return probability.reshape(theta.shape[0], -1)


def _numerical_item_information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Fisher matrices ``sum_c grad p_c grad p_c' / p_c`` by central differences.

    Used for models that do not define ``item_information_matrix``.
    """
    from mirt.constants import PROB_EPSILON

    n_theta, n_factors = theta.shape
    probabilities = _item_probability_columns(model, theta, item_idx)
    steps = 1e-5 * np.maximum(1.0, np.abs(theta))
    gradient = np.empty(probabilities.shape + (n_factors,), dtype=np.float64)
    for factor in range(n_factors):
        shift = np.zeros_like(theta)
        shift[:, factor] = steps[:, factor]
        upper = _item_probability_columns(model, theta + shift, item_idx)
        lower = _item_probability_columns(model, theta - shift, item_idx)
        gradient[:, :, factor] = (upper - lower) / (2.0 * steps[:, factor, None])
    scaled = gradient / np.maximum(probabilities, PROB_EPSILON)[:, :, None]
    return np.einsum("ncf,ncg->nfg", scaled, gradient)


def _item_information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Return one item's Fisher matrices with shape ``(n_theta, F, F)``."""
    from mirt.exceptions import MirtModelError

    n_theta, n_factors = theta.shape
    native = getattr(model, "item_information_matrix", None)
    if callable(native):
        try:
            matrices = np.asarray(native(theta, item_idx), dtype=np.float64)
        except MirtModelError:
            # Mixed-format components without exact matrices.
            pass
        else:
            return matrices.reshape(n_theta, n_factors, n_factors)
    return _numerical_item_information_matrix(model, theta, item_idx)


def observed_information_matrix(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return Fisher information matrices contributed by observed items.

    The result has shape ``(n_theta, n_factors, n_factors)``. Its trace is
    :func:`observed_test_information`, and for one factor the two agree.
    ``observed_mask`` is a shared ``(n_items,)`` mask or one mask per theta
    row, as for :func:`observed_test_information`. Models without an
    ``item_information_matrix`` method use ``sum_c grad p_c grad p_c' / p_c``
    with central-difference probability gradients.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n_theta, n_factors = theta.shape
    if n_factors == 1:
        return observed_test_information(model, theta, observed_mask)[:, None, None]
    row_mask, observed_items = _observed_item_masks(model, n_theta, observed_mask)
    matrices = np.zeros((n_theta, n_factors, n_factors), dtype=np.float64)
    if observed_items.size == 0:
        return matrices

    slopes = _rank_one_slopes(model)
    if slopes is not None:
        # Polytomous models return test totals from ``information``; their
        # item columns come from the shared vectorized hook.
        item_columns = getattr(model, "_information_by_item", model.information)
        information = np.asarray(item_columns(theta), dtype=np.float64)
        if information.shape == (n_theta, model.n_items):
            squared_norms = np.einsum("jf,jf->j", slopes, slopes)
            weights = np.divide(
                information,
                squared_norms,
                out=np.zeros_like(information),
                where=squared_norms > 0.0,
            )
            if row_mask is None:
                row_mask = np.zeros((1, model.n_items), dtype=np.bool_)
                row_mask[0, observed_items] = True
            weights = np.where(row_mask, weights, 0.0)
            return np.einsum("nj,jf,jg->nfg", weights, slopes, slopes)

    for item_idx in observed_items.tolist():
        item_matrices = _item_information_matrix(model, theta, item_idx)
        if row_mask is not None:
            item_matrices = np.where(
                row_mask[:, item_idx, None, None], item_matrices, 0.0
            )
        matrices += item_matrices
    return matrices


def measured_factors(
    model: BaseItemModel,
    observed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return which factors each response row's observed items measure.

    ``observed`` has shape ``(n_rows, n_items)``. A factor is measured when
    an observed item carries information about it; otherwise the row's
    likelihood does not depend on that coordinate at all.
    """
    n_rows = observed.shape[0]
    information = observed_information_matrix(
        model, np.zeros((n_rows, model.n_factors)), observed
    )
    return np.diagonal(information, axis1=1, axis2=2) > 0.0


def resolve_prior_distribution(
    *,
    n_factors: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve and validate a finite, nondegenerate normal scoring prior."""
    if prior_mean is None:
        mean = np.zeros(n_factors, dtype=np.float64)
    else:
        mean = np.asarray(prior_mean, dtype=np.float64)

    if prior_cov is None:
        cov = np.eye(n_factors, dtype=np.float64)
    else:
        cov = np.asarray(prior_cov, dtype=np.float64)

    if mean.shape != (n_factors,):
        raise ValueError(f"prior_mean must have shape ({n_factors},)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("prior_mean must contain only finite values")
    if cov.shape != (n_factors, n_factors):
        raise ValueError(f"prior_cov must have shape ({n_factors}, {n_factors})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("prior_cov must contain only finite values")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=0.0):
        raise ValueError("prior_cov must be symmetric")
    # Make negligible roundoff symmetric before Cholesky: LAPACK otherwise uses
    # only the lower triangle, whereas MAP would invert the entire matrix.
    # Difference-based averaging preserves equal subnormal variances that
    # would be rounded to zero if each diagonal entry were halved first.
    cov = cov + 0.5 * (cov.T - cov)
    lower = np.tril_indices(n_factors, k=-1)
    cov[lower] = cov.T[lower]
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_cov must be positive definite") from exc

    return mean, cov


def build_quadrature(
    *,
    n_quadpts: int,
    n_factors: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build Gauss-Hermite quadrature nodes and weights from prior settings."""
    if (
        isinstance(n_quadpts, (bool, np.bool_))
        or not isinstance(n_quadpts, (int, np.integer))
        or n_quadpts < 1
    ):
        raise ValueError("n_quadpts must be a positive integer")
    mean, cov = resolve_prior_distribution(
        n_factors=n_factors,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    quadrature = GaussHermiteQuadrature(
        n_points=n_quadpts,
        n_dimensions=n_factors,
        mean=mean,
        cov=cov,
    )
    return quadrature.nodes, quadrature.weights


def validate_scoring_responses(
    model: BaseItemModel,
    responses: ArrayLike,
) -> NDArray[np.int_]:
    """Validate a scoring matrix and normalize every missing code to -1.

    Negative codes, ``NaN`` and the missing entries of nullable DataFrame
    columns are missing responses, as in :func:`mirt.fit_mirt`. A DataFrame
    with named columns is first matched to the model's item names by
    :func:`mirt.utils.data.align_response_columns`.

    Raises
    ------
    MirtDataError
        If the matrix is not two-dimensional, has the wrong number of items,
        or contains non-numeric, non-finite, fractional or out-of-range
        response codes.
    """
    raw = _missing_coded_responses(
        align_response_columns(responses, getattr(model, "item_names", None))
    )
    if raw.ndim != 2:
        raise MirtDataError(f"responses must be 2D, got {raw.ndim}D")
    if raw.shape[1] != model.n_items:
        raise MirtDataError(
            f"responses has {raw.shape[1]} items, expected {model.n_items}",
            n_items=raw.shape[1],
        )
    if raw.dtype.kind not in "biuf":
        raise MirtDataError("responses must contain numeric values")

    if raw.dtype.kind in "biu":
        values = raw
    else:
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise MirtDataError("responses must contain only finite values")
    observed = values >= 0.0
    # Missing codes must be integers too, as in mirt.fit_mirt.
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise MirtDataError("response codes must be integer-valued")

    if model.is_polytomous:
        category_counts = np.asarray(model.n_categories, dtype=np.float64)
        invalid = observed & (values >= category_counts[None, :])
        if np.any(invalid):
            item_idx = int(np.flatnonzero(np.any(invalid, axis=0))[0])
            raise MirtDataError(
                f"responses for item {item_idx} must be below "
                f"{int(category_counts[item_idx])}, the item's category range",
                item_idx=item_idx,
            )
    elif np.any(observed & (values > 1.0)):
        raise MirtDataError(
            "dichotomous responses must be coded as 0 or 1; only 0, 1, or "
            "missing values are allowed"
        )

    if values.dtype == np.dtype(np.int_) and np.all(observed | (values == -1)):
        return values
    return np.where(observed, values, -1.0).astype(np.int_, copy=False)


def extreme_response_rows(
    model: BaseItemModel,
    patterns: NDArray[np.int_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Flag rows answered only in the lowest or only in the highest category.

    ``patterns`` are validated responses with missing entries coded
    negative. A row is extreme when it observes at least one item and every
    observed response is category zero (``all_lowest``) or every one is its
    item's highest category (``all_highest``). Rows meeting both conditions
    observe only single-category items, whose responses carry no
    information, and are flagged as neither.
    """
    observed = patterns >= 0
    if model.is_polytomous:
        top = np.asarray(model.n_categories, dtype=np.int_) - 1
    else:
        top = np.ones(model.n_items, dtype=np.int_)
    informative = observed.any(axis=1)
    all_lowest = informative & np.all(~observed | (patterns == 0), axis=1)
    all_highest = informative & np.all(~observed | (patterns == top[None, :]), axis=1)
    both = all_lowest & all_highest
    return all_lowest & ~both, all_highest & ~both


def unique_response_patterns(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.intp]]:
    """Collapse duplicate response rows in first-appearance order."""
    from mirt.backends.rust.patterns import response_pattern_indices

    first_indices, inverse, _ = response_pattern_indices(responses)
    return responses[first_indices], inverse


def finite_difference_se(
    objective: Callable[[float], float],
    estimate: float,
    *,
    step: float = 1e-4,
) -> float:
    """Estimate SE from second finite difference of a scalar objective.

    The default step is near ``eps ** 0.25``, which balances the truncation
    and rounding errors of the second difference.
    """
    f_plus = objective(estimate + step)
    f_minus = objective(estimate - step)
    f_center = objective(estimate)
    hessian = (f_plus - 2 * f_center + f_minus) / (step**2)

    if hessian > 0:
        return float(np.sqrt(1.0 / hessian))
    return float(np.nan)


def finite_difference_se_rows(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    estimate: NDArray[np.float64],
    *,
    center: NDArray[np.float64] | None = None,
    step: float = 1e-4,
) -> NDArray[np.float64]:
    """Row-wise :func:`finite_difference_se` for independent scalar objectives.

    ``objective(x)`` evaluates every row at the matching entry of ``x``.
    ``center`` may supply objective values already known at ``estimate``.
    """
    f_plus = objective(estimate + step)
    f_minus = objective(estimate - step)
    f_center = objective(estimate) if center is None else center
    hessian = (f_plus - 2 * f_center + f_minus) / (step**2)

    standard_error = np.full(estimate.shape, np.nan, dtype=np.float64)
    positive = hessian > 0
    standard_error[positive] = np.sqrt(1.0 / hessian[positive])
    return standard_error


def supports_row_batched_scoring(model: BaseItemModel) -> bool:
    """Return whether stacked rows can share one log-likelihood evaluation.

    Optimizer scorers may then evaluate many ``(pattern, theta)`` pairs in a
    single call. Custom likelihoods and instance-level hook overrides keep the
    per-pattern path, because they need not treat theta rows independently.
    """
    from mirt._model_defaults import uses_original_model_hook
    from mirt.models.mixed_format import MixedItemModel

    namespace = getattr(model, "__dict__", None)
    if namespace is None or any(
        name in namespace for name in ("probability", "log_likelihood", "information")
    ):
        return False
    # A mixed-format likelihood sums its components' likelihoods.
    if isinstance(model, MixedItemModel) and not all(
        map(supports_row_batched_scoring, model.component_models)
    ):
        return False
    return uses_original_model_hook(model, "log_likelihood")


def score_pattern_chunks(
    patterns: NDArray[np.int_],
    score_chunk: Callable[
        [NDArray[np.int_]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ],
    chunk_size: int = BATCHED_SCORING_CHUNK_SIZE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run a row-batched scorer over bounded consecutive pattern chunks."""
    results = [
        score_chunk(patterns[start : start + chunk_size])
        for start in range(0, max(patterns.shape[0], 1), chunk_size)
    ]
    if len(results) == 1:
        return results[0]
    return (
        np.concatenate([theta for theta, _ in results]),
        np.concatenate([standard_error for _, standard_error in results]),
    )


def score_responses_parallel(
    *,
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_jobs: int,
    score_person: Callable[
        [int],
        tuple[float | NDArray[np.float64], float | NDArray[np.float64]],
    ],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute per-person scoring either serially or with a thread pool."""
    from mirt.utils._parallel import resolve_n_jobs

    n_persons = responses.shape[0]
    n_factors = model.n_factors

    theta_values = np.zeros((n_persons, n_factors), dtype=np.float64)
    se_values = np.zeros((n_persons, n_factors), dtype=np.float64)

    if n_persons == 0:
        if n_factors == 1:
            return theta_values.ravel(), se_values.ravel()
        return theta_values, se_values

    worker_count = resolve_n_jobs(n_jobs)

    if worker_count == 1:
        results = map(score_person, range(n_persons))
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, n_persons)) as executor:
            results = executor.map(score_person, range(n_persons))

    for i, (theta_est, se_est) in enumerate(results):
        if n_factors == 1:
            theta_values[i, 0] = float(
                np.asarray(theta_est, dtype=np.float64).ravel()[0]
            )
            se_values[i, 0] = float(np.asarray(se_est, dtype=np.float64).ravel()[0])
        else:
            theta_values[i] = np.asarray(theta_est, dtype=np.float64).reshape(n_factors)
            se_values[i] = np.asarray(se_est, dtype=np.float64).reshape(n_factors)

    if n_factors == 1:
        return theta_values.ravel(), se_values.ravel()
    return theta_values, se_values
