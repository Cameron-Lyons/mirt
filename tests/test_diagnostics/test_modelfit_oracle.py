"""M2/M2* checked against exhaustive multinomial response distributions.

The reference enumerates every response pattern, differentiates category
probabilities analytically, and uses an unwhitened null-space quadratic form.
It does not call the diagnostic's moment, covariance, derivative, or projection
helpers. This catches wrong repeated-item powers and spurious chi-square df.
"""

from itertools import product

import numpy as np
import pytest
from scipy import linalg, stats
from scipy.special import expit

from mirt import compute_fit_indices, compute_m2, fit_mirt
from mirt.diagnostics import modelfit
from mirt.models import (
    FourParameterLogistic,
    GradedResponseModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.mixture import MixtureIRT


def _nodes(n_points, n_factors):
    locations, weights = np.polynomial.hermite.hermgauss(n_points)
    locations *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    indices = np.array(list(product(range(n_points), repeat=n_factors)))
    return locations[indices], np.prod(weights[indices], axis=1)


def _category_probability_derivatives(model, nodes):
    """Independent logistic and graded category probabilities/derivatives."""
    parameters = model.parameters
    a = parameters["discrimination"]
    b = parameters["thresholds"] if model.is_polytomous else parameters["difficulty"]
    free = model.free_parameter_masks
    coordinates = [
        (name, index)
        for name, mask in free.items()
        for index in np.ndindex(mask.shape)
        if mask[index]
    ]
    counts = model.n_categories if model.is_polytomous else [2] * model.n_items
    probabilities = np.zeros((len(nodes), model.n_items, max(counts)))
    derivatives = np.zeros((*probabilities.shape, len(coordinates)))
    for item, categories in enumerate(counts):
        slope = np.atleast_1d(a[item])
        locations = np.atleast_1d(b[item])[: categories - 1]
        # Both builtin parameterizations reduce to a dot product minus the
        # sum of loadings times each location when n_factors > 1.
        logits = nodes @ slope[:, None] - slope.sum() * locations
        cumulative = expit(logits)
        probabilities[:, item, :categories] = (
            np.diff(
                np.column_stack(
                    [np.ones(len(nodes)), cumulative, np.zeros(len(nodes))]
                ),
                axis=1,
            )
            * -1
        )
        for column, (name, index) in enumerate(coordinates):
            if index[0] != item:
                continue
            dc = np.zeros_like(cumulative)
            if name == "discrimination":
                factor = index[1] if len(index) > 1 else 0
                dc = (
                    cumulative * (1 - cumulative) * (nodes[:, factor, None] - locations)
                )
            else:
                threshold = index[1] if len(index) > 1 else 0
                dc[:, threshold] = (
                    -slope.sum()
                    * cumulative[:, threshold]
                    * (1 - cumulative[:, threshold])
                )
            derivatives[:, item, :categories, column] = -np.diff(
                np.column_stack([np.zeros(len(nodes)), dc, np.zeros(len(nodes))]),
                axis=1,
            )
    if not model.is_polytomous:
        # The graded construction orders binary categories as (0,1) too.
        assert probabilities.shape[2] == 2
    return probabilities, derivatives


def _exhaustive_distribution(model, nodes, weights):
    counts = model.n_categories if model.is_polytomous else [2] * model.n_items
    patterns = np.array(list(product(*(range(count) for count in counts))))
    feature = np.array(
        [
            [
                *row,
                *(
                    row[j] * row[k]
                    for j in range(model.n_items)
                    for k in range(j + 1, model.n_items)
                ),
            ]
            for row in patterns
        ],
        dtype=float,
    )
    probabilities, derivatives = _category_probability_derivatives(model, nodes)
    selected = probabilities[:, np.arange(model.n_items)[None, :], patterns]
    conditional_mass = selected.prod(axis=2)
    selected_d = derivatives[:, np.arange(model.n_items)[None, :], patterns, :]
    score = np.divide(
        selected_d,
        selected[..., None],
        out=np.zeros_like(selected_d),
        where=selected[..., None] > 0,
    ).sum(axis=2)
    mass = weights @ conditional_mass
    derivative_mass = np.einsum("n,nr,nrp->rp", weights, conditional_mass, score)
    return patterns, feature, mass, feature.T @ derivative_mass


def _oracle(model, responses, theta=None, n_points=31):
    """Covariance of every available-case mean from full pattern probabilities."""
    n_items = model.n_items
    present = np.isfinite(responses) & (responses >= 0)
    mask = np.column_stack(
        [
            present,
            *(
                present[:, j] & present[:, k]
                for j in range(n_items)
                for k in range(j + 1, n_items)
            ),
        ]
    ).astype(float)
    counts = mask.sum(axis=0)
    observed_rows = np.nan_to_num(np.where(present, responses, np.nan))
    observed_features = np.column_stack(
        [
            observed_rows,
            *(
                observed_rows[:, j] * observed_rows[:, k]
                for j in range(n_items)
                for k in range(j + 1, n_items)
            ),
        ]
    )
    selected = counts > 0
    observed = (observed_features * mask).sum(axis=0)[selected] / counts[selected]
    covariance = np.zeros((len(counts), len(counts)))
    expected = np.zeros(len(counts))
    jacobian = None
    nodes, weights = _nodes(n_points, model.n_factors)
    if theta is None:
        _, features, mass, derivative = _exhaustive_distribution(model, nodes, weights)
        means = mass @ features
        centered = features - means
        population_cov = (centered * mass[:, None]).T @ centered
        covariance = population_cov * (mask.T @ mask)
        expected = means * counts
        jacobian = derivative * counts[:, None]
    else:
        nodes = np.asarray(theta).reshape(len(responses), model.n_factors)
        for row, node in enumerate(nodes):
            _, features, mass, derivative = _exhaustive_distribution(
                model, node[None, :], np.ones(1)
            )
            mean = mass @ features
            centered = (features - mean) * mask[row]
            covariance += (centered * mass[:, None]).T @ centered
            expected += mean * mask[row]
            if jacobian is None:
                jacobian = np.zeros_like(derivative)
            jacobian += derivative * mask[row, :, None]
    covariance = covariance[np.ix_(selected, selected)] / np.outer(
        counts[selected], counts[selected]
    )
    expected = expected[selected] / counts[selected]
    jacobian = jacobian[selected] / counts[selected, None]
    null = linalg.null_space(jacobian.T, rcond=1e-7)
    df = null.shape[1]
    if not df:
        return np.nan, 0
    residual = null.T @ (observed - expected)
    statistic = residual @ np.linalg.solve(null.T @ covariance @ null, residual)
    return float(statistic), df


@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("fixed_theta", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_exhaustive_pattern_covariance_and_projection(polytomous, fixed_theta, missing):
    rng = np.random.default_rng(31415)
    model = (
        GradedResponseModel(6, n_categories=[2, 3, 4, 2, 3, 2])
        if polytomous
        else TwoParameterLogistic(6)
    )
    model.set_parameters(discrimination=np.array([0.65, 1.3, 0.8, 1.65, 0.9, 1.15]))
    if not polytomous:
        model.set_parameters(difficulty=np.array([-1.2, -0.3, 0.4, 0.8, -0.7, 1.1]))
    categories = model.n_categories if polytomous else [2] * model.n_items
    responses = rng.integers(0, categories, size=(17, model.n_items)).astype(float)
    if missing:
        responses[rng.random(responses.shape) < 0.25] = np.nan
        responses[0] = -1
    theta = rng.normal(size=len(responses)) if fixed_theta else None
    expected, df = _oracle(model, responses, theta)
    actual = compute_m2(model, responses, theta=theta, n_quadpts=31)
    assert actual["df"] == df
    assert actual["M2"] == pytest.approx(expected, rel=2e-7, abs=2e-7)
    assert actual["p_value"] == pytest.approx(stats.chi2.sf(expected, df), abs=1e-9)


def test_multidimensional_rotation_redundancy_uses_tangent_rank():
    model = TwoParameterLogistic(7, n_factors=2)
    model.set_parameters(
        discrimination=np.array(
            [
                [0.8, 0.2],
                [1.2, 0.4],
                [0.3, 1.1],
                [0.2, 0.9],
                [1.1, 0.6],
                [0.7, 1.0],
                [1.4, 0.1],
            ]
        ),
        difficulty=np.linspace(-1.2, 1.2, 7),
    )
    responses = np.random.default_rng(225).integers(0, 2, size=(71, 7))
    expected, df = _oracle(model, responses, n_points=17)
    actual = compute_m2(model, responses, n_quadpts=17)
    # A rotation of the two loading columns leaves the response distribution
    # unchanged. The 21 stored free parameters span only 20 moment directions.
    assert df == 8
    assert actual["df"] == df
    assert actual["M2"] == pytest.approx(expected, rel=2e-6)


def test_underidentified_collapsed_test_does_not_invent_degrees_of_freedom():
    model = GradedResponseModel(3, n_categories=[2, 3, 4])
    responses = np.random.default_rng(125).integers(0, [2, 3, 4], size=(100, 3))
    result = compute_fit_indices(model, responses)
    assert result["M2_df"] == 0
    for name in (
        "M2",
        "M2_p",
        "RMSEA",
        "RMSEA_CI_lower",
        "RMSEA_CI_upper",
        "CFI",
        "TLI",
    ):
        assert np.isnan(result[name])
    assert np.isfinite(result["SRMSR"])


def test_heterogeneous_items_reject_categories_within_global_maximum():
    model = GradedResponseModel(6, n_categories=[2, 3, 4, 2, 3, 4])
    responses = np.zeros((10, 6), dtype=int)
    responses[8, 0] = 2
    with pytest.raises(ValueError, match="each item's category range"):
        compute_m2(model, responses)


def test_latent_class_marginals_do_not_masquerade_as_joint_model_moments():
    model = MixtureIRT(7)
    responses = np.random.default_rng(545).integers(0, 2, size=(71, 7))
    with pytest.raises(ValueError, match="class-marginal item probabilities"):
        compute_m2(model, responses)


@pytest.mark.parametrize("missing", [False, True])
def test_generating_model_chi_square_calibration_and_local_dependence_power(missing):
    model = TwoParameterLogistic(7).set_parameters(
        discrimination=np.array([0.65, 1.3, 0.8, 1.65, 0.9, 1.15, 1.4]),
        difficulty=np.linspace(-1.2, 1.2, 7),
    )
    rng = np.random.default_rng(28319)
    statistics, p_values = [], []
    for _ in range(96):
        responses = (
            rng.random((2500, 7)) < model.probability(rng.normal(size=(2500, 1)))
        ).astype(float)
        if missing:
            responses[rng.random(responses.shape) < 0.2] = np.nan
        result = compute_m2(model, responses, n_quadpts=31)
        assert result["df"] == 14
        statistics.append(result["M2"])
        p_values.append(result["p_value"])
    # Broad, predeclared tolerances: test actual chi-square scale and tail
    # calibration over independent IRT samples, rather than p-value ranges.
    assert 11.5 < np.mean(statistics) < 16.5
    assert 0.0 < np.mean(np.asarray(p_values) < 0.05) < 0.13
    responses = (
        rng.random((2500, 7)) < model.probability(rng.normal(size=(2500, 1)))
    ).astype(float)
    responses[:, 1] = responses[:, 0]
    if missing:
        responses[rng.random(responses.shape) < 0.2] = -1
    misfit = compute_m2(model, responses, n_quadpts=31)
    assert misfit["M2"] > 200
    assert misfit["p_value"] < 1e-20


def test_fitted_irt_null_and_fitted_local_dependence_alternative():
    rng = np.random.default_rng(9163)
    generating = TwoParameterLogistic(7).set_parameters(
        discrimination=np.array([0.8, 1.2, 1.1, 0.9, 1.4, 1.0, 1.3]),
        difficulty=np.linspace(-1.1, 1.1, 7),
    )
    responses = (
        rng.random((3500, 7)) < generating.probability(rng.normal(size=(3500, 1)))
    ).astype(int)
    fitted = fit_mirt(responses, model="2PL", n_quadpts=31, max_iter=150, tol=1e-5)
    reference, df = _oracle(fitted.model, responses)
    null_fit = compute_m2(fitted.model, responses, n_quadpts=31)
    assert null_fit["M2"] == pytest.approx(reference, rel=2e-6)
    assert null_fit["df"] == df == 14
    assert null_fit["p_value"] > 0.05
    misfit_responses = responses.copy()
    # A substantial shared response disturbance in one pair cannot be
    # absorbed by recalibrating independent one-factor item curves.
    copied = rng.random(len(responses)) < 0.7
    misfit_responses[copied, 1] = misfit_responses[copied, 0]
    refitted = fit_mirt(
        misfit_responses, model="2PL", n_quadpts=31, max_iter=150, tol=1e-5
    )
    alternative = compute_m2(refitted.model, misfit_responses, n_quadpts=31)
    assert alternative["p_value"] < 1e-8
    assert alternative["M2"] > null_fit["M2"] * 5


def test_ordinal_m2_star_chi_square_calibration():
    model = GradedResponseModel(
        8, n_categories=[2, 3, 4, 2, 3, 4, 2, 3]
    ).set_parameters(
        discrimination=np.array([0.65, 1.3, 0.8, 1.65, 0.9, 1.15, 1.4, 1.0]),
        thresholds=np.array(
            [
                [-1.1, 0, 0],
                [-0.7, 0.8, 0],
                [-1.8, -0.2, 0.9],
                [-0.4, 0, 0],
                [-1.3, 0.6, 0],
                [-1.1, 0.1, 1.4],
                [0.3, 0, 0],
                [-0.6, 1.3, 0],
            ]
        ),
    )
    rng = np.random.default_rng(6193)
    statistics, p_values = [], []
    for _ in range(96):
        probabilities = model.probability(rng.normal(size=(2500, 1)))
        responses = np.sum(
            rng.random((2500, 8, 1)) > np.cumsum(probabilities, axis=-1), axis=-1
        )
        result = compute_m2(model, responses, n_quadpts=31)
        assert result["df"] == 13
        statistics.append(result["M2"])
        p_values.append(result["p_value"])
    assert 10.5 < np.mean(statistics) < 15.5
    assert 0.0 < np.mean(np.asarray(p_values) < 0.05) < 0.13
    responses[:, 5] = responses[:, 2]
    misfit = compute_m2(model, responses, n_quadpts=31)
    assert misfit["M2"] > 200
    assert misfit["p_value"] < 1e-20


class _KnownConditionalBinary(TwoParameterLogistic):
    """Known local-independent binary distribution with a deterministic item."""

    @property
    def free_parameter_masks(self):
        return {
            name: np.zeros_like(values, dtype=bool)
            for name, values in self.parameters.items()
        }

    def probability(self, theta, item_idx=None):
        values = np.array([0.0, 0.3, 0.8, 0.45])
        if item_idx is not None:
            return np.full(len(theta), values[item_idx])
        return np.broadcast_to(values, (len(theta), self.n_items))


def test_singular_covariance_uses_support_rank_and_detects_impossible_scores():
    model = _KnownConditionalBinary(4)
    rng = np.random.default_rng(471)
    responses = (rng.random((1000, 4)) < [0.0, 0.3, 0.8, 0.45]).astype(int)
    result = compute_m2(model, responses)
    # The first item and its three pair products have exactly zero variance.
    assert result["df"] == 6
    assert np.isfinite(result["M2"])
    assert 0 < result["p_value"] < 1
    responses[0, 0] = 1
    impossible = compute_m2(model, responses)
    assert impossible["df"] == 6
    assert np.isinf(impossible["M2"])
    assert impossible["p_value"] == 0


def test_independence_baseline_matches_exact_binary_pearson_table():
    counts = np.array([[30, 10], [20, 60]])
    responses = np.repeat(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), counts.ravel(), axis=0
    )
    model = TwoParameterLogistic(2)
    values, maximum = modelfit._validate_diagnostic_inputs(model, responses)
    moments = modelfit._prepare_fit_moments(model, values, maximum, None, 31)
    actual, df = modelfit._baseline_m2(moments, values)
    independent = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.sum()
    expected = np.sum((counts - independent) ** 2 / independent)
    assert df == 1
    assert actual == pytest.approx(expected, rel=1e-12)


class _AnchoredLogistic(TwoParameterLogistic):
    @property
    def free_parameter_masks(self):
        masks = super().free_parameter_masks
        for mask in masks.values():
            mask[:2] = False
        return masks


@pytest.mark.parametrize("explicit_masks", [False, True])
def test_fixed_anchor_masks_change_the_projected_tangent(explicit_masks):
    family = TwoParameterLogistic if explicit_masks else _AnchoredLogistic
    model = family(6).set_parameters(
        discrimination=np.array([0.65, 1.3, 0.8, 1.65, 0.9, 1.15]),
        difficulty=np.linspace(-1.2, 1.2, 6),
    )
    if explicit_masks:
        masks = model.free_parameter_masks
        for mask in masks.values():
            mask[:2] = False
        model.set_free_parameter_masks(masks)
    responses = np.random.default_rng(742).integers(0, 2, size=(313, 6))
    expected, df = _oracle(model, responses)
    actual = compute_m2(model, responses, n_quadpts=31)
    # Four fixed anchor coordinates remain testable instead of being removed
    # as nuisance directions just because they are present in parameter storage.
    assert actual["df"] == df == 13
    assert actual["M2"] == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize("family", [ThreeParameterLogistic, FourParameterLogistic])
def test_probability_boundary_derivatives_preserve_model_and_remain_finite(family):
    model = family(9).set_parameters(
        discrimination=np.linspace(0.65, 1.7, 9),
        difficulty=np.linspace(-1.2, 1.2, 9),
        guessing=np.zeros(9),
    )
    original = model.parameters
    responses = np.random.default_rng(781).integers(0, 2, size=(203, 9))
    result = compute_m2(model, responses, n_quadpts=31)
    assert result["df"] > 0
    assert np.isfinite(result["M2"])
    assert 0 <= result["p_value"] <= 1
    for name, values in original.items():
        np.testing.assert_array_equal(model.parameters[name], values)
