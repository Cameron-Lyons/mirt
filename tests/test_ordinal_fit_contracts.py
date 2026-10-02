"""Real ordinal calibration with heterogeneous and sparse category support."""

import numpy as np
import pytest

from mirt import fit_mirt
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models import GradedResponseModel, OneParameterLogistic, TwoParameterLogistic


@pytest.mark.parametrize("family", ["GRM", "GPCM", "PCM", "NRM"])
@pytest.mark.parametrize("declared", [None, [2, 3, 4], (2, 3, 4), np.array([2, 3, 4])])
def test_public_fit_preserves_per_item_category_counts(family, declared):
    responses = np.array([[0, 0, 0], [1, 1, 1], [0, 2, 2], [1, 1, 3], [-1, -1, 2]])
    result = fit_mirt(
        responses,
        model=family,
        n_categories=declared,
        n_quadpts=5,
        max_iter=2,
        use_rust=False,
        compute_standard_errors=False,
    )
    assert result.model.n_categories == [2, 3, 4]
    assert np.isfinite(result.log_likelihood)


def test_explicit_scalar_preserves_unobserved_response_categories():
    responses = np.array([[0, 0, 0], [1, 1, 1], [0, 2, 2], [1, 1, 3]])
    result = fit_mirt(
        responses,
        model="GRM",
        n_categories=np.int64(4),
        n_quadpts=5,
        max_iter=2,
        use_rust=False,
        compute_standard_errors=False,
    )
    assert result.model.n_categories == [4, 4, 4]


@pytest.mark.parametrize(
    "counts,message",
    [
        ([2, 3], "shape"),
        ([[2, 3, 4]], "shape"),
        (True, "integer category counts"),
        ([2, True, 4], "at least 2"),
        (3.0, "integer category counts"),
        ([2, 3.5, 4], "integer category counts"),
        ([2, 1, 4], "at least 2"),
        (0, "at least 2"),
    ],
)
def test_declared_category_validation_is_explicit_and_per_item(counts, message):
    responses = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(MirtValidationError, match=message):
        fit_mirt(responses, model="GRM", n_categories=counts, use_rust=False)


def test_response_code_must_fit_its_own_item_even_below_global_category_maximum():
    with pytest.raises(MirtDataError, match="for each item"):
        fit_mirt(np.array([[0, 0, 0], [2, 1, 3]]), model="GRM", n_categories=[2, 3, 4])


def test_unobserved_item_requires_declared_count_and_constant_zero_item_infers_binary():
    responses = np.array([[0, -1, 0], [0, -1, 1], [0, -1, 2]])
    with pytest.raises(MirtValidationError, match="items with no observed responses"):
        fit_mirt(responses, model="GRM")
    result = fit_mirt(
        responses,
        model="GRM",
        n_categories=[2, 4, 3],
        n_quadpts=5,
        max_iter=2,
        use_rust=False,
        compute_standard_errors=False,
    )
    assert result.model.n_categories == [2, 4, 3]
    observed_only = fit_mirt(
        responses[:, [0, 2]],
        model="GRM",
        n_quadpts=5,
        max_iter=2,
        use_rust=False,
        compute_standard_errors=False,
    )
    assert observed_only.model.n_categories == [2, 3]


@pytest.mark.parametrize("declared", [None, 4])
def test_sparse_graded_calibration_keeps_real_category_probabilities_valid(declared):
    # This previously inferred four categories globally. For the binary first
    # item EM returned thresholds near [-1.8, 6, 2] and negative probabilities
    # as large as -0.83, hidden by clipping in the likelihood.
    rng = np.random.default_rng(33091)
    generating = GradedResponseModel(8, n_categories=[2, 3, 4, 3, 2, 4, 3, 4])
    probabilities = generating.probability(rng.normal(size=(1200, 1)))
    responses = (
        rng.random(probabilities.shape[:2])[:, :, None] > probabilities.cumsum(axis=2)
    ).sum(axis=2)
    result = fit_mirt(
        responses,
        model="GRM",
        n_categories=declared,
        n_quadpts=21,
        max_iter=100,
        tol=1e-5,
        use_rust=False,
        compute_standard_errors=False,
    )
    assert result.converged
    expected_counts = generating.n_categories if declared is None else [4] * 8
    assert result.model.n_categories == expected_counts
    for item, count in enumerate(expected_counts):
        thresholds = result.model.thresholds[item, : count - 1]
        assert np.all(np.diff(thresholds) >= 0)
    grid = GaussHermiteQuadrature(n_points=21)
    fitted_probability = result.model.probability(grid.nodes)
    assert np.all(fitted_probability >= 0)
    np.testing.assert_allclose(fitted_probability.sum(axis=2), 1, atol=1e-14)
    # Verify the reported objective against the actual joint probability
    # distribution; clipping negative category curves cannot fake this check.
    selected = fitted_probability[:, np.arange(8)[None, :], responses]
    pattern_mass = grid.weights @ selected.prod(axis=2)
    assert result.log_likelihood == pytest.approx(np.log(pattern_mass).sum(), abs=1e-6)


class _FixedOuterThresholds(GradedResponseModel):
    @property
    def free_parameter_masks(self):
        masks = super().free_parameter_masks
        masks["thresholds"][:, [0, 2]] = False
        return masks


@pytest.mark.parametrize("explicit_masks", [False, True])
def test_graded_optimizer_orders_free_thresholds_around_fixed_neighbors(explicit_masks):
    rng = np.random.default_rng(643)
    family = GradedResponseModel if explicit_masks else _FixedOuterThresholds
    model = family(2, n_categories=4)
    if explicit_masks:
        masks = model.free_parameter_masks
        masks["thresholds"][:, [0, 2]] = False
        model.set_free_parameter_masks(masks)
    model.set_parameters(thresholds=np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]))
    responses = rng.integers(0, 4, size=(300, 2))
    estimator = EMEstimator(
        n_quadpts=11,
        max_iter=15,
        n_jobs=2,
        use_rust=False,
        compute_standard_errors=False,
    )
    result = estimator.fit(model, responses)
    np.testing.assert_array_equal(
        result.model.thresholds[:, [0, 2]], [[-1, 1], [-1, 1]]
    )
    assert np.all(result.model.thresholds[:, 1] > -1)
    assert np.all(result.model.thresholds[:, 1] < 1)
    probabilities = result.model.probability(np.linspace(-4, 4, 41))
    assert np.all(probabilities >= 0)
    assert np.isfinite(result.log_likelihood)


@pytest.mark.parametrize("free_slot", [0, 1, 2])
def test_one_free_graded_threshold_cannot_cross_fixed_neighbors(free_slot):
    model = GradedResponseModel(1, n_categories=4).set_parameters(
        thresholds=np.array([[-1.0, 0.0, 1.0]])
    )
    masks = {
        name: np.zeros_like(values, dtype=bool)
        for name, values in model.parameters.items()
    }
    masks["thresholds"][0, free_slot] = True
    model.set_free_parameter_masks(masks)
    responses = np.array([0] * 90 + [1] * 5 + [3] * 5)[:, None]
    posterior = np.full((100, 9), 1 / 9)
    estimator = EMEstimator(n_quadpts=9, use_rust=False)
    estimator._quadrature = GaussHermiteQuadrature(9)
    before = model.thresholds.copy()
    estimator._m_step(model, responses, posterior)
    frozen = ~masks["thresholds"]
    np.testing.assert_array_equal(model.thresholds[frozen], before[frozen])
    assert np.all(np.diff(model.thresholds[0]) >= 0)
    assert np.all(model.probability(np.linspace(-5, 5, 51)) >= 0)


@pytest.mark.parametrize("family", [OneParameterLogistic, TwoParameterLogistic])
def test_unfitted_em_initialization_preserves_frozen_item_values(family):
    model = family(2)
    values = {"difficulty": np.array([2.0, 0.0])}
    if family is TwoParameterLogistic:
        values["discrimination"] = np.array([1.7, 1.0])
    model.set_parameters(**values)
    masks = model.free_parameter_masks
    for mask in masks.values():
        mask[0] = False
    model.set_free_parameter_masks(masks)
    before = model.parameters
    responses = np.random.default_rng(386).integers(0, 2, size=(150, 2))
    result = EMEstimator(
        n_quadpts=11, max_iter=5, use_rust=False, compute_standard_errors=False
    ).fit(model, responses)
    for name, original in before.items():
        np.testing.assert_array_equal(
            result.model.parameters[name][~masks[name]], original[~masks[name]]
        )
    assert np.isfinite(result.log_likelihood)
