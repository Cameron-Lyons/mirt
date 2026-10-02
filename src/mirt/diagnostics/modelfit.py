"""Model fit statistics for IRT models.

This module provides limited-information goodness-of-fit statistics:
- M2 statistic (Maydeu-Olivares & Joe, 2005)
- RMSEA (Root Mean Square Error of Approximation)
- CFI (Comparative Fit Index)
- TLI (Tucker-Lewis Index)
- SRMSR (Standardized Root Mean Square Residual)
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_MOMENT_CHUNK_ELEMENTS = 262_144


@dataclass(frozen=True)
class _ScoreMoments:
    """Final score moments and pairwise means on the same observations."""

    univariate: NDArray[np.float64]
    bivariate: NDArray[np.float64]
    correlation: NDArray[np.float64]
    pair_means: NDArray[np.float64]


class _SampleMomentAccumulator:
    """Accumulate score moments with shared pairwise observation counts."""

    def __init__(self, n_items: int) -> None:
        self.sums = np.zeros(n_items)
        self.products = np.zeros((n_items, n_items))
        self.pair_sums = np.zeros_like(self.products)
        self.pair_seconds = np.zeros_like(self.products)

    def add(
        self,
        means: NDArray[np.float64],
        seconds: NDArray[np.float64],
        mask: NDArray[np.float64] | None,
    ) -> None:
        if mask is not None:
            means = means * mask
            seconds = seconds * mask
        sums = means.sum(axis=0)
        self.sums += sums
        self.products += means.T @ means
        if mask is None:
            self.pair_sums += sums[:, None]
            self.pair_seconds += seconds.sum(axis=0)[:, None]
        else:
            self.pair_sums += means.T @ mask
            self.pair_seconds += seconds.T @ mask

    def finish(self, pair_counts: NDArray[np.float64]) -> _ScoreMoments:
        univariate = _safe_divide(self.sums, pair_counts.diagonal())
        bivariate = _safe_divide(self.products, pair_counts)
        pair_means = _safe_divide(self.pair_sums, pair_counts)
        pair_seconds = _safe_divide(self.pair_seconds, pair_counts)
        return _ScoreMoments(
            univariate,
            bivariate,
            _score_correlations(bivariate, pair_means, pair_seconds),
            pair_means,
        )


def _score_correlations(
    bivariate: NDArray[np.float64],
    pair_means: NDArray[np.float64],
    pair_seconds: NDArray[np.float64],
) -> NDArray[np.float64]:
    covariance = bivariate - pair_means * pair_means.T
    variances = np.maximum(pair_seconds - pair_means**2, 0.0)
    return _safe_divide(covariance, np.sqrt(variances * variances.T))


@dataclass(frozen=True)
class _FitMoments:
    """Observed and model-implied first- and second-order score moments."""

    observed_uni: NDArray[np.float64]
    observed_bi: NDArray[np.float64]
    expected_uni: NDArray[np.float64]
    expected_bi: NDArray[np.float64]
    observed_corr: NDArray[np.float64]
    expected_corr: NDArray[np.float64]
    observed_pair_means: NDArray[np.float64]
    uni_counts: NDArray[np.float64]
    pair_counts: NDArray[np.float64]


def compute_m2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> dict[str, float]:
    """Compute M2 limited-information fit statistic.

    The statistic weights first- and second-order score residuals by their
    model-implied covariance and projects out the free parameter tangent.
    Binary items use M2; ordinal items use the collapsed M2* formulation of
    Cai and Hansen (2013). Degrees of freedom use numerical covariance and
    tangent ranks. When no testable dimensions remain, inferential outputs
    are NaN instead of reporting a fictitious chi-square test.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix (n_persons, n_items)
    theta : NDArray, optional
        Fixed person abilities defining a conditional response model. If
        omitted, moments are integrated against a standard normal latent
        distribution. Abilities estimated from these responses do not satisfy
        the fixed-design assumption needed for chi-square calibration.
    n_quadpts : int
        Number of quadrature points for integration

    Returns
    -------
    dict
        Dictionary with:
        - 'M2': M2 statistic value
        - 'df': Degrees of freedom
        - 'p_value': P-value
        - 'M2_df_ratio': M2/df ratio
    """
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
        theta,
        n_quadpts,
    )
    return _m2_from_moments(model, moments, response_values, theta, n_quadpts)


def compute_fit_indices(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_quadpts: int = 21,
) -> dict[str, float]:
    """Compute model fit indices (RMSEA, CFI, TLI, SRMSR).

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Fixed person abilities defining a conditional response model. If
        omitted, moments are integrated against a standard normal latent
        distribution. Response-derived ability estimates invalidate the
        fixed-design chi-square calibration.
    n_quadpts : int
        Number of quadrature points

    Returns
    -------
    dict
        Dictionary with:
        - 'RMSEA': Root Mean Square Error of Approximation
        - 'RMSEA_CI_lower': Lower bound of 90% CI for RMSEA
        - 'RMSEA_CI_upper': Upper bound of 90% CI for RMSEA
        - 'CFI': Comparative Fit Index
        - 'TLI': Tucker-Lewis Index (NNFI)
        - 'SRMSR': Standardized Root Mean Square Residual
    """
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    n_persons = response_values.shape[0]
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
        theta,
        n_quadpts,
    )
    design = _moment_design(response_values)
    m2_result = _m2_from_moments(
        model, moments, response_values, theta, n_quadpts, design
    )
    M2 = m2_result["M2"]
    df = m2_result["df"]

    M2_0, df_0 = _baseline_m2(moments, response_values, design)

    rmsea = _compute_rmsea(M2, df, n_persons)
    rmsea_ci = _compute_rmsea_ci(M2, df, n_persons)

    cfi = _compute_cfi(M2, df, M2_0, df_0)

    tli = _compute_tli(M2, df, M2_0, df_0)

    srmsr = _srmsr_from_moments(moments)

    return {
        "RMSEA": rmsea,
        "RMSEA_CI_lower": rmsea_ci[0],
        "RMSEA_CI_upper": rmsea_ci[1],
        "CFI": cfi,
        "TLI": tli,
        "SRMSR": srmsr,
        "M2": M2,
        "M2_df": df,
        "M2_p": m2_result["p_value"],
    }


def _compute_expected_margins(
    model: BaseItemModel,
    n_quadpts: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected score moments under the model using quadrature."""
    moments, _ = _integrate_model_moments(model, n_quadpts)
    return moments.univariate, moments.bivariate


def _baseline_m2(
    moments: _FitMoments,
    responses: NDArray[np.float64],
    design: _MomentDesign | None = None,
) -> tuple[float, int]:
    """Use the same covariance-weighted test for the independence baseline."""
    n_items = responses.shape[1]
    if design is None:
        design = _moment_design(responses)
    # Arbitrary marginal score distributions are nuisance parameters. Their
    # means affect the tested moments, while their variances only set weights.
    means = np.nan_to_num(moments.observed_uni)[None, :]
    seconds = np.nan_to_num(np.diag(moments.observed_bi))[None, :]
    expected = _score_features(means)[0]
    covariance = _conditional_covariance_sum(means, seconds, np.ones(1))
    covariance *= _safe_divide(design.overlap, np.outer(design.counts, design.counts))
    jacobian = np.zeros((len(expected), n_items))
    jacobian[:n_items] = np.eye(n_items)
    left, right = np.triu_indices(n_items, 1)
    jacobian[n_items + np.arange(len(left)), left] = means[0, right]
    jacobian[n_items + np.arange(len(left)), right] = means[0, left]
    observed = _flatten_score_moments(moments.observed_uni, moments.observed_bi)
    selected = design.counts > 0
    return _projected_chi_square(
        (observed - expected)[selected],
        covariance[np.ix_(selected, selected)],
        jacobian[selected],
    )


def _validate_diagnostic_inputs(
    model: BaseItemModel,
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.float64], float]:
    """Validate response blocks without retaining a full mask or float copy."""
    from mirt.models.cdm_advanced import HigherOrderCDM
    from mirt.models.mixture import MixtureIRT

    if isinstance(model, MixtureIRT):
        raise ValueError(
            "M2 does not support MixtureIRT: class-marginal item probabilities "
            "do not identify the required joint response moments"
        )
    if isinstance(model, HigherOrderCDM):
        raise ValueError(
            "M2 does not support HigherOrderCDM: shared mastery-pattern "
            "integration is required for the joint response moments"
        )
    values = np.asarray(responses)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            "responses must be a two-dimensional matrix with at least "
            "2 persons and 1 item"
        )
    if values.shape[1] != model.n_items:
        raise ValueError(
            f"responses have {values.shape[1]} items, expected {model.n_items}"
        )
    rows_per_chunk = max(1, _MOMENT_CHUNK_ELEMENTS // model.n_items)
    category_counts = np.asarray(
        model.n_categories if model.is_polytomous else [2] * model.n_items
    )
    max_observed = -1.0
    for start in range(0, values.shape[0], rows_per_chunk):
        block = np.asarray(values[start : start + rows_per_chunk], dtype=np.float64)
        if np.any(np.isinf(block)):
            raise ValueError("responses must not contain infinite values")
        observed = block[np.isfinite(block) & (block >= 0)]
        if np.any(observed != np.floor(observed)):
            raise ValueError("observed responses must be integer category codes")
        max_observed = max(max_observed, float(np.max(observed, initial=-1.0)))
        if np.any(np.isfinite(block) & (block >= category_counts)):
            raise ValueError(
                "observed response categories must be between 0 and "
                f"{int(np.max(category_counts)) - 1}, within each item's category range"
            )
    if max_observed < 0:
        raise ValueError("responses contain no observed values")
    return values, max_observed


def _prepare_theta(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_persons: int,
) -> NDArray[np.float64]:
    """Normalize and validate person ability values."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim == 1 and model.n_factors == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("theta must be a two-dimensional ability matrix")
    if values.shape != (n_persons, model.n_factors):
        raise ValueError(
            f"theta must have shape ({n_persons}, {model.n_factors}), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain only finite values")
    return values


def _validate_quadrature_count(n_quadpts: int) -> None:
    """Validate the requested quadrature resolution."""
    if isinstance(n_quadpts, bool) or not isinstance(n_quadpts, (int, np.integer)):
        raise ValueError("n_quadpts must be an integer")
    if n_quadpts < 2:
        raise ValueError("n_quadpts must be at least 2")


def _normalized_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return finite nonnegative integration weights summing to one."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(values))
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or total <= 0
    ):
        raise ValueError("quadrature weights must be finite and nonnegative")
    return values / total


def _conditional_score_moments(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Compute conditional item score means and second moments in one pass."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    n_rows = theta.shape[0]

    if probabilities.ndim == 1 and model.n_items == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.ndim == 2:
        if probabilities.shape != (n_rows, model.n_items):
            raise ValueError(
                f"model probability output must have shape ({n_rows}, {model.n_items})"
            )
        expected_scores = probabilities
        expected_squares = probabilities
        max_score = 1
    elif probabilities.ndim == 3:
        if probabilities.shape[:2] != (n_rows, model.n_items):
            raise ValueError(
                "polytomous probability output must start with shape "
                f"({n_rows}, {model.n_items})"
            )
        category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
        expected_scores = probabilities @ category_scores
        expected_squares = probabilities @ category_scores**2
        max_score = probabilities.shape[2] - 1
        if not np.allclose(np.sum(probabilities, axis=2), 1.0, atol=1e-8):
            raise ValueError("polytomous probabilities must sum to one")
    else:
        raise ValueError("model probability output has an unsupported shape")

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < -PROB_EPSILON)
        or np.any(probabilities > 1.0 + PROB_EPSILON)
    ):
        raise ValueError("model probabilities must be finite and between zero and one")
    return expected_scores, expected_squares, max_score


def _safe_divide(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Divide arrays while marking unsupported moments as missing."""
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def _moment_rows_per_chunk(model: BaseItemModel) -> int:
    """Budget probability storage by the widest item category count."""
    width = max(model.n_categories) if model.is_polytomous else 1
    return max(1, _MOMENT_CHUNK_ELEMENTS // (model.n_items * width))


def _integrate_model_moments(
    model: BaseItemModel, n_quadpts: int
) -> tuple[_ScoreMoments, int]:
    """Integrate model moments without retaining full-grid probability arrays."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    _validate_quadrature_count(n_quadpts)
    quadrature = GaussHermiteQuadrature(
        n_points=n_quadpts, n_dimensions=model.n_factors
    )
    weights = _normalized_weights(quadrature.weights)
    univariate = np.zeros(model.n_items)
    seconds = np.zeros(model.n_items)
    bivariate = np.zeros((model.n_items, model.n_items))
    rows_per_chunk = _moment_rows_per_chunk(model)
    max_score = 0
    for start in range(0, weights.size, rows_per_chunk):
        stop = start + rows_per_chunk
        means, conditional_seconds, max_score = _conditional_score_moments(
            model, quadrature.nodes[start:stop]
        )
        block_weights = weights[start:stop]
        univariate += block_weights @ means
        seconds += block_weights @ conditional_seconds
        bivariate += (means * block_weights[:, None]).T @ means
    pair_means = univariate[:, None]
    return _ScoreMoments(
        univariate,
        bivariate,
        _score_correlations(bivariate, pair_means, seconds[:, None]),
        pair_means,
    ), max_score


def _prepare_fit_moments(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    max_observed: float,
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
) -> _FitMoments:
    """Stream moments, sharing observation counts and complete-block shortcuts."""
    observed = _SampleMomentAccumulator(model.n_items)
    expected = _SampleMomentAccumulator(model.n_items) if theta is not None else None
    pair_counts = np.zeros((model.n_items, model.n_items))
    theta_values = (
        _prepare_theta(model, theta, responses.shape[0]) if theta is not None else None
    )
    if theta is None:
        expected_moments, max_score = _integrate_model_moments(model, n_quadpts)

    rows_per_chunk = (
        _moment_rows_per_chunk(model)
        if theta is not None
        else max(1, _MOMENT_CHUNK_ELEMENTS // model.n_items)
    )
    for start in range(0, responses.shape[0], rows_per_chunk):
        stop = start + rows_per_chunk
        block = np.asarray(responses[start:stop], dtype=np.float64)
        valid = np.isfinite(block) & (block >= 0)
        mask = None
        if np.all(valid):
            pair_counts += block.shape[0]
            scores = block
        else:
            mask = valid.astype(np.float64)
            pair_counts += mask.T @ mask
            scores = np.where(valid, block, 0.0)
        observed.add(scores, scores**2, mask)
        if expected is not None and theta_values is not None:
            means, seconds, max_score = _conditional_score_moments(
                model, theta_values[start:stop]
            )
            expected.add(means, seconds, mask)
        if max_observed > max_score:
            raise ValueError(
                f"observed response categories must be between 0 and {max_score}"
            )

    observed_moments = observed.finish(pair_counts)
    if expected is not None:
        expected_moments = expected.finish(pair_counts)
    return _FitMoments(
        observed_uni=observed_moments.univariate,
        observed_bi=observed_moments.bivariate,
        expected_uni=expected_moments.univariate,
        expected_bi=expected_moments.bivariate,
        observed_corr=observed_moments.correlation,
        expected_corr=expected_moments.correlation,
        observed_pair_means=observed_moments.pair_means,
        uni_counts=pair_counts.diagonal(),
        pair_counts=pair_counts,
    )


def _flatten_score_moments(
    univariate: NDArray[np.float64], bivariate: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.concatenate([univariate, bivariate[np.triu_indices(len(univariate), 1)]])


def _score_features(means: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conditional expectations of (Y_j, Y_j Y_k), in a fixed ordering."""
    left, right = np.triu_indices(means.shape[1], 1)
    return np.concatenate([means, means[:, left] * means[:, right]], axis=1)


@dataclass(frozen=True)
class _MomentDesign:
    """Observation counts and overlaps for pairwise available-case moments."""

    counts: NDArray[np.float64]
    overlap: NDArray[np.float64]


def _moment_design(responses: NDArray[np.float64]) -> _MomentDesign:
    n_items = responses.shape[1]
    n_moments = n_items * (n_items + 1) // 2
    counts = np.zeros(n_moments)
    overlap = np.zeros((n_moments, n_moments))
    chunk_rows = max(1, _MOMENT_CHUNK_ELEMENTS // n_moments)
    for start in range(0, len(responses), chunk_rows):
        block = responses[start : start + chunk_rows]
        valid = np.isfinite(block) & (block >= 0)
        if np.all(valid):
            counts += len(block)
            overlap += len(block)
        else:
            present = _score_features(valid.astype(np.float64))
            counts += present.sum(axis=0)
            overlap += present.T @ present
    return _MomentDesign(counts, overlap)


def _conditional_covariance_sum(
    means: NDArray[np.float64],
    seconds: NDArray[np.float64],
    weights: NDArray[np.float64],
    present: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sum exact conditional score-feature covariances in bounded blocks.

    Conditional independence makes disjoint features uncorrelated. Each item
    contributes Var(Y_j) times an outer product of the other conditional
    means. Pair variances additionally require Var(Y_j) Var(Y_k). This includes
    repeated item powers in E(g g'), which cannot be replaced by E(g) E(g)'.
    """
    n_items = means.shape[1]
    left, right = np.triu_indices(n_items, 1)
    n_moments = n_items + len(left)
    result = np.zeros((n_moments, n_moments))
    variances = np.maximum(seconds - means**2, 0.0)
    for item in range(n_items):
        pairs = np.flatnonzero((left == item) | (right == item))
        others = np.where(left[pairs] == item, right[pairs], left[pairs])
        indices = np.concatenate([[item], n_items + pairs])
        factors = np.column_stack([np.ones(len(means)), means[:, others]])
        if present is not None:
            factors *= present[:, indices]
        result[np.ix_(indices, indices)] += (
            factors * (weights * variances[:, item])[:, None]
        ).T @ factors
    products = variances[:, left] * variances[:, right]
    if present is not None:
        products *= present[:, n_items:]
    pair_indices = n_items + np.arange(len(left))
    result[pair_indices, pair_indices] += weights @ products
    return result


def _integration_blocks(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
) -> Iterator[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]
]:
    """Yield bounded quadrature blocks or fixed-design person blocks."""
    if theta is None:
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        _validate_quadrature_count(n_quadpts)
        quadrature = GaussHermiteQuadrature(n_quadpts, model.n_factors)
        nodes = quadrature.nodes
        weights = _normalized_weights(quadrature.weights)
    else:
        nodes = _prepare_theta(model, theta, len(responses))
    n_moments = model.n_items * (model.n_items + 1) // 2
    chunk_rows = min(
        _moment_rows_per_chunk(model),
        max(1, _MOMENT_CHUNK_ELEMENTS // n_moments),
    )
    for start in range(0, len(nodes), chunk_rows):
        stop = start + chunk_rows
        present = None
        if theta is not None:
            block = responses[start:stop]
            valid = np.isfinite(block) & (block >= 0)
            if not np.all(valid):
                present = _score_features(valid.astype(np.float64))
        block_weights = (
            weights[start:stop] if theta is None else np.ones(len(nodes[start:stop]))
        )
        yield nodes[start:stop], block_weights, present


def _model_sample_covariance(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
    design: _MomentDesign,
) -> NDArray[np.float64]:
    n_moments = len(design.counts)
    conditional = np.zeros((n_moments, n_moments))
    raw = np.zeros_like(conditional)
    mean = np.zeros(n_moments)
    for nodes, weights, present in _integration_blocks(
        model, responses, theta, n_quadpts
    ):
        means, seconds, _ = _conditional_score_moments(model, nodes)
        conditional += _conditional_covariance_sum(means, seconds, weights, present)
        if theta is None:
            features = _score_features(means)
            mean += weights @ features
            raw += (features * weights[:, None]).T @ features
    if theta is None:
        # Law of total covariance: integrate within-person variability and
        # between-ability variability. Masks are exogenous/MCAR under this null.
        conditional += raw - np.outer(mean, mean)
        conditional *= design.overlap
    return _safe_divide(conditional, np.outer(design.counts, design.counts))


def _conditional_item_mean(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item: int,
) -> NDArray[np.float64]:
    """Evaluate an item-local derivative without allocating all-item curves."""
    probabilities = np.asarray(model.probability(theta, item), dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -PROB_EPSILON) | (probabilities > 1 + PROB_EPSILON)
    ):
        raise ValueError("model probabilities must be finite and between zero and one")
    if model.is_polytomous:
        if probabilities.ndim != 2 or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-8
        ):
            raise ValueError("polytomous probabilities must sum to one")
        return probabilities @ np.arange(probabilities.shape[1])
    return probabilities.reshape(len(theta))


def _model_moment_jacobian(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
    design: _MomentDesign,
) -> NDArray[np.float64]:
    """Differentiate actual free parameters on an isolated model instance."""
    from mirt._model_defaults import uses_builtin_model_hooks

    parameters = model.parameters
    free = model.free_parameter_masks
    coordinates = [
        (name, index) for name, mask in free.items() for index in np.flatnonzero(mask)
    ]
    jacobian = np.zeros((len(design.counts), len(coordinates)))
    if not coordinates:
        return jacobian
    worker = deepcopy(model)
    builtin = uses_builtin_model_hooks(model)
    left, right = np.triu_indices(model.n_items, 1)
    for nodes, weights, present in _integration_blocks(
        model, responses, theta, n_quadpts
    ):
        means, _, _ = _conditional_score_moments(model, nodes)
        original = _score_features(means)
        for column, (name, index) in enumerate(coordinates):
            values = parameters[name]
            center = values.flat[index]
            step = np.cbrt(np.finfo(float).eps) * max(abs(center), 1.0)
            # One-dimensional thresholds are shared by rating-scale models.
            # Other built-in item arrays have leading axis n_items. Custom
            # curves always take the general full-model derivative route.
            local = (
                builtin
                and values.ndim > 0
                and values.shape[0] == model.n_items
                and name not in {"feature_weights", "testlet_variances"}
                and not (
                    name in {"thresholds", "class_proportions"} and values.ndim == 1
                )
            )
            item = int(np.unravel_index(index, values.shape)[0]) if local else None
            candidates: list[NDArray[np.float64] | None] = []
            for offset in (step, -step):
                candidate = values.copy()
                candidate.flat[index] = center + offset
                try:
                    candidate = worker._canonical_parameter_values(name, candidate)
                except ValueError:
                    candidates.append(None)
                else:
                    candidates.append(candidate)
            if item is not None and any(
                candidate is not None
                and (
                    not np.array_equal(candidate[:item], values[:item], equal_nan=True)
                    or not np.array_equal(
                        candidate[item + 1 :], values[item + 1 :], equal_nan=True
                    )
                )
                for candidate in candidates
            ):
                # A stored representative may control several items, as with
                # a common LLTM discrimination. Its derivative is global.
                item = None
            perturbed = []
            for candidate in candidates:
                if candidate is None:
                    perturbed.append(None)
                    continue
                try:
                    worker.set_parameters(**{name: candidate})
                    if item is None:
                        shifted, _, _ = _conditional_score_moments(worker, nodes)
                        value = _score_features(shifted)
                    else:
                        value = _conditional_item_mean(worker, nodes, item)
                except ValueError:
                    value = None
                finally:
                    current = worker.parameters
                    restore = {
                        parameter: original
                        for parameter, original in parameters.items()
                        if not np.array_equal(
                            current[parameter], original, equal_nan=True
                        )
                    }
                    if restore:
                        worker.set_parameters(**restore)
                perturbed.append(value)
            upper, lower = perturbed
            center_value = original if item is None else means[:, item]
            if upper is not None and lower is not None:
                derivative = (upper - lower) / (2 * step)
            elif upper is not None:
                derivative = (upper - center_value) / step
            elif lower is not None:
                derivative = (center_value - lower) / step
            else:
                raise ValueError(f"cannot differentiate model moment parameter {name}")
            if item is None:
                if present is not None:
                    derivative *= present
                jacobian[:, column] += weights @ derivative
            else:
                pair_indices = np.flatnonzero((left == item) | (right == item))
                others = np.where(
                    left[pair_indices] == item, right[pair_indices], left[pair_indices]
                )
                indices = np.concatenate([[item], model.n_items + pair_indices])
                factors = np.column_stack([np.ones(len(nodes)), means[:, others]])
                if present is not None:
                    factors *= present[:, indices]
                jacobian[indices, column] += (weights * derivative) @ factors
    return jacobian if theta is None else _safe_divide(jacobian, design.counts[:, None])


def _projected_chi_square(
    residual: NDArray[np.float64],
    covariance: NDArray[np.float64],
    jacobian: NDArray[np.float64],
) -> tuple[float, int]:
    """Whiten sampling residuals and remove the nuisance tangent subspace.

    This is algebraically C = Xi^-1 - Xi^-1 D (D' Xi^-1 D)^+ D' Xi^-1,
    the M2/M2* weight of Maydeu-Olivares & Joe and Cai & Hansen. Covariance
    here is already that of the sample means, so an extra N is unnecessary.
    """
    covariance = (covariance + covariance.T) * 0.5
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("model moment covariance and derivatives must be finite")
    # Standardize before rank decisions so ordinal score units do not determine
    # which dimensions survive. Exactly deterministic dimensions carry no test.
    scales = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    active = scales > np.finfo(float).eps
    if not np.any(active):
        return np.nan, 0
    standardized = covariance[np.ix_(active, active)] / np.outer(
        scales[active], scales[active]
    )
    values, vectors = np.linalg.eigh(standardized)
    threshold = max(float(np.max(values)), 1.0) * 1e-10
    if np.any(values < -threshold):
        raise ValueError("model moment covariance is not positive semidefinite")
    positive = values > threshold
    whitening = (vectors[:, positive] / np.sqrt(values[positive])).T
    whitened = whitening @ (residual[active] / scales[active])
    tangent = whitening @ (jacobian[active] / scales[active, None])
    norms = np.linalg.norm(tangent, axis=0)
    nonzero = norms > 1e-9
    if np.any(nonzero):
        basis, singular, _ = np.linalg.svd(
            tangent[:, nonzero] / norms[nonzero], full_matrices=False
        )
        rank = int(np.count_nonzero(singular > singular[0] * 1e-6))
        whitened -= basis[:, :rank] @ (basis[:, :rank].T @ whitened)
    else:
        rank = 0
    degrees = int(np.count_nonzero(positive)) - rank
    if degrees <= 0:
        return np.nan, 0
    # A residual outside covariance support is impossible under the null;
    # pseudoinverse weighting must not silently erase this evidence of misfit.
    null_residual = vectors[:, ~positive].T @ (residual[active] / scales[active])
    impossible = np.linalg.norm(null_residual) > 1e-6 or np.any(
        np.abs(residual[~active]) > 1e-10
    )
    statistic = np.inf if impossible else float(whitened @ whitened)
    return statistic, degrees


def _m2_from_moments(
    model: BaseItemModel,
    moments: _FitMoments,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64] | None,
    n_quadpts: int,
    design: _MomentDesign | None = None,
) -> dict[str, float]:
    """Compute a genuinely covariance-weighted limited-information test."""
    if design is None:
        design = _moment_design(responses)
    selected = design.counts > 0
    observed = _flatten_score_moments(moments.observed_uni, moments.observed_bi)
    expected = _flatten_score_moments(moments.expected_uni, moments.expected_bi)
    covariance = _model_sample_covariance(model, responses, theta, n_quadpts, design)
    jacobian = _model_moment_jacobian(model, responses, theta, n_quadpts, design)
    statistic, degrees = _projected_chi_square(
        (observed - expected)[selected],
        covariance[np.ix_(selected, selected)],
        jacobian[selected],
    )
    return {
        "M2": statistic,
        "df": degrees,
        "p_value": float(stats.chi2.sf(statistic, degrees)) if degrees else np.nan,
        "M2_df_ratio": statistic / degrees if degrees else np.nan,
    }


def _srmsr_from_moments(moments: _FitMoments) -> float:
    """Compute SRMSR from finite pairwise score correlations."""
    upper = np.triu_indices_from(moments.observed_corr, k=1)
    residuals = moments.observed_corr[upper] - moments.expected_corr[upper]
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(residuals**2)))


def _compute_rmsea(chi2: float, df: int, n: int) -> float:
    """Compute RMSEA."""
    if df <= 0:
        return np.nan

    rmsea_sq = max((chi2 / df - 1) / (n - 1), 0)
    return float(np.sqrt(rmsea_sq))


def _compute_rmsea_ci(
    chi2: float,
    df: int,
    n: int,
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Compute confidence interval for RMSEA."""
    if df <= 0:
        return (np.nan, np.nan)

    def rmsea_from_ncp(ncp: float) -> float:
        return np.sqrt(max(ncp / (df * (n - 1)), 0))

    from scipy.optimize import brentq

    try:
        central_survival = float(stats.chi2.sf(chi2, df))

        def solve_ncp(target_survival: float) -> float:
            if central_survival >= target_survival:
                return 0.0
            upper = max(float(chi2), float(df), 1.0)
            while stats.ncx2.sf(chi2, df, upper) < target_survival:
                upper *= 2.0
                if upper > 1e8:
                    raise RuntimeError("could not bracket RMSEA noncentrality")
            return float(
                brentq(
                    lambda ncp: stats.ncx2.sf(chi2, df, ncp) - target_survival,
                    0.0,
                    upper,
                )
            )

        ncp_lower = solve_ncp(alpha / 2)
        lower = rmsea_from_ncp(ncp_lower)
        ncp_upper = solve_ncp(1 - alpha / 2)
        upper = rmsea_from_ncp(ncp_upper)

    except (ValueError, RuntimeError):
        se = np.sqrt(2 / (n - 1))
        rmsea = _compute_rmsea(chi2, df, n)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = max(rmsea - z * se, 0)
        upper = rmsea + z * se

    return (float(lower), float(upper))


def _compute_cfi(chi2: float, df: int, chi2_0: float, df_0: int) -> float:
    """Compute Comparative Fit Index."""
    if df <= 0 or df_0 <= 0 or np.isnan(chi2) or np.isnan(chi2_0):
        return np.nan
    if np.isinf(chi2):
        return 0.0 if np.isfinite(chi2_0) else np.nan

    numerator = max(chi2 - df, 0)
    denominator = max(chi2_0 - df_0, chi2 - df, 0)

    if denominator <= 0:
        return 1.0

    cfi = 1 - numerator / denominator
    return float(np.clip(cfi, 0, 1))


def _compute_tli(chi2: float, df: int, chi2_0: float, df_0: int) -> float:
    """Compute Tucker-Lewis Index (NNFI)."""
    if df_0 <= 0 or df <= 0:
        return np.nan

    ratio_0 = chi2_0 / df_0
    ratio = chi2 / df

    if ratio_0 <= 1:
        return 1.0

    tli = (ratio_0 - ratio) / (ratio_0 - 1)
    return float(tli)


def _compute_srmsr(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_quadpts: int,
    theta: NDArray[np.float64] | None = None,
) -> float:
    """Compute Standardized Root Mean Square Residual."""
    response_values, max_observed = _validate_diagnostic_inputs(model, responses)
    moments = _prepare_fit_moments(
        model,
        response_values,
        max_observed,
        theta,
        n_quadpts,
    )
    return _srmsr_from_moments(moments)


def model_fit_summary(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
) -> str:
    """Generate a formatted summary of model fit statistics.

    Parameters
    ----------
    model : BaseItemModel
        Fitted IRT model
    responses : NDArray
        Response matrix
    theta : NDArray, optional
        Ability estimates

    Returns
    -------
    str
        Formatted summary string
    """
    fit = compute_fit_indices(model, responses, theta)

    lines = [
        "Model Fit Summary",
        "=" * 50,
        "",
        f"M2 statistic:     {fit['M2']:.3f}",
        f"Degrees of freedom: {fit['M2_df']}",
        f"P-value:          {fit['M2_p']:.4f}",
        "",
        f"RMSEA:            {fit['RMSEA']:.4f}",
        f"  90% CI:         [{fit['RMSEA_CI_lower']:.4f}, {fit['RMSEA_CI_upper']:.4f}]",
        f"CFI:              {fit['CFI']:.4f}",
        f"TLI:              {fit['TLI']:.4f}",
        f"SRMSR:            {fit['SRMSR']:.4f}",
        "",
        "Interpretation guidelines:",
        "  RMSEA < 0.05: Good fit",
        "  RMSEA < 0.08: Acceptable fit",
        "  CFI > 0.95: Good fit",
        "  TLI > 0.95: Good fit",
        "  SRMSR < 0.08: Good fit",
    ]

    return "\n".join(lines)
