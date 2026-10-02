"""MGPCM checks against linear adjacent logits, rather than model round trips.

Cui, Wang, and Xu (2024), equation 2.2, uses ``a @ theta - beta`` for
each adjacent-category logit: https://arxiv.org/html/2401.13090v1#S2
Here ``beta = sum(a) * steps`` follows mirt's centered-threshold convention.
"""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, softmax

from mirt.cat.mcat_selection import _compute_item_information_matrix
from mirt.estimation._polytomous_information import polytomous_item_curvature
from mirt.estimation._polytomous_objective import prepare_polytomous_objective
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    PartialCreditModel,
    RatingScaleModel,
)
from mirt.utils.simulation import simdata


def _reference_probabilities(theta, slopes, steps):
    category = np.arange(len(steps) + 1)
    intercepts = -np.sum(slopes) * np.r_[0.0, np.cumsum(steps)]
    return softmax((theta @ slopes)[:, None] * category + intercepts, axis=1)


@pytest.mark.parametrize("factors", [1, 2, 4])
def test_gpcm_curves_match_adjacent_logit_reference_for_every_public_path(factors):
    rng = np.random.default_rng(2719)
    model = GeneralizedPartialCredit(3, [2, 4, 3], n_factors=factors)
    slopes = rng.uniform(0.25, 1.8, (3, factors))
    steps = rng.normal(size=(3, 3))
    model.set_parameters(
        discrimination=slopes[:, 0] if factors == 1 else slopes,
        steps=steps,
    )
    theta = rng.normal(size=(19, factors))
    expected = np.zeros((19, 3, 4))
    for item, categories in enumerate(model.n_categories):
        expected[:, item, :categories] = _reference_probabilities(
            theta, slopes[item], steps[item, : categories - 1]
        )
        np.testing.assert_allclose(
            model.probability(theta, item),
            expected[:, item, :categories],
            rtol=1e-13,
            atol=1e-14,
        )
        for category in range(categories):
            np.testing.assert_allclose(
                model.category_probability(theta, item, category),
                expected[:, item, category],
                rtol=1e-13,
                atol=1e-14,
            )
    np.testing.assert_allclose(model.probability(theta), expected, rtol=1e-13)
    indices = np.arange(19) % 3
    np.testing.assert_allclose(
        model.probability_pairs(theta, indices),
        expected[np.arange(19), indices],
        rtol=1e-13,
    )
    responses = np.array([[0, 2, 1], [1, -1, 2]])
    expected_likelihood = np.zeros((2, 19))
    for person, row in enumerate(responses):
        for item, category in enumerate(row):
            if category >= 0:
                expected_likelihood[person] += np.log(expected[:, item, category])
    np.testing.assert_allclose(
        model.log_likelihood_batch(responses, theta),
        expected_likelihood,
        rtol=1e-13,
    )


def test_zero_loading_dimensions_preserve_unidimensional_gpcm():
    one = GeneralizedPartialCredit(2, [3, 4])
    three = GeneralizedPartialCredit(2, [3, 4], n_factors=3)
    slopes = np.array([1.7, 0.65])
    steps = np.array([[-0.8, 1.1, 0.0], [-1.2, -0.1, 0.9]])
    one.set_parameters(discrimination=slopes, steps=steps)
    three.set_parameters(
        discrimination=np.column_stack([slopes, np.zeros((2, 2))]),
        steps=steps,
    )
    theta = np.linspace(-2.0, 2.0, 17)
    points = np.column_stack([theta, np.full(17, 91.0), np.full(17, -23.0)])
    np.testing.assert_allclose(three.probability(points), one.probability(theta))
    np.testing.assert_allclose(three.information(points), one.information(theta))


@pytest.mark.parametrize("slopes", [[1.8, 0.4], [1.2, -0.4], [0.0, 0.0]])
def test_dichotomous_gpcm_reduces_to_centered_multidimensional_2pl(slopes):
    slopes = np.asarray(slopes)
    model = GeneralizedPartialCredit(1, 2, n_factors=2)
    model.set_parameters(discrimination=slopes[None], steps=np.array([[0.7]]))
    theta = np.array([[-1.3, 0.5], [0.0, 0.0], [0.7, 0.7], [1.2, -0.3]])
    expected = expit((theta - 0.7) @ slopes)
    np.testing.assert_allclose(model.probability(theta, 0)[:, 1], expected)
    np.testing.assert_allclose(
        model.information(theta, 0), np.dot(slopes, slopes) * expected * (1 - expected)
    )


def test_fisher_matrices_match_independent_probability_gradients_and_mcat():
    model = GeneralizedPartialCredit(2, [3, 4], n_factors=2)
    slopes = np.array([[1.8, 0.4], [0.6, 1.3]])
    steps = np.array([[-0.8, 0.9, 0.0], [-1.2, 0.2, 1.1]])
    model.set_parameters(discrimination=slopes, steps=steps)
    theta = np.array([[-1.3, 0.5], [0.0, 0.0], [0.7, 0.7], [1.2, -0.3]])
    total = np.zeros((4, 2, 2))
    for item, categories in enumerate(model.n_categories):
        probability = _reference_probabilities(
            theta, slopes[item], steps[item, : categories - 1]
        )
        gradients = []
        for factor in range(2):
            shift = np.zeros_like(theta)
            shift[:, factor] = 1e-5
            gradients.append(
                (
                    _reference_probabilities(
                        theta + shift, slopes[item], steps[item, : categories - 1]
                    )
                    - _reference_probabilities(
                        theta - shift, slopes[item], steps[item, : categories - 1]
                    )
                )
                / 2e-5
            )
        gradients = np.stack(gradients, axis=-1)
        expected = np.einsum("ncf,ncg,nc->nfg", gradients, gradients, 1 / probability)
        actual_gradients = []
        for factor in range(2):
            shift = np.zeros_like(theta)
            shift[:, factor] = 1e-5
            actual_gradients.append(
                (
                    model.probability(theta + shift, item)
                    - model.probability(theta - shift, item)
                )
                / 2e-5
            )
        actual_gradients = np.stack(actual_gradients, axis=-1)
        numerical_information = np.einsum(
            "ncf,ncg,nc->nfg",
            actual_gradients,
            actual_gradients,
            1 / model.probability(theta, item),
        )
        actual = model.item_information_matrix(theta, item)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-10)
        np.testing.assert_allclose(actual, numerical_information, rtol=2e-9, atol=1e-10)
        assert np.linalg.eigvalsh(actual).min() > -1e-12
        np.testing.assert_allclose(
            model.information(theta, item), np.trace(actual, axis1=1, axis2=2)
        )
        np.testing.assert_allclose(
            _compute_item_information_matrix(model, theta[0], item),
            expected[0],
            rtol=2e-9,
            atol=1e-10,
        )
        total += expected
    np.testing.assert_allclose(model.test_information_matrix(theta), total, rtol=2e-9)
    np.testing.assert_allclose(
        model.information(theta), np.trace(total, axis1=1, axis2=2), rtol=2e-9
    )
    with pytest.raises(IndexError):
        model.item_information_matrix(theta, -1)


@pytest.mark.parametrize(
    "model",
    [GeneralizedPartialCredit(1, 2), PartialCreditModel(1, 2), RatingScaleModel(1, 2)],
)
def test_score_information_retains_saturated_success_tail(model):
    if isinstance(model, PartialCreditModel):
        model.set_parameters(steps=np.array([[0.0]]))
        slope = 1.0
    elif isinstance(model, GeneralizedPartialCredit):
        model.set_parameters(discrimination=np.array([2.0]), steps=np.array([[0.0]]))
        slope = 2.0
    else:
        model.set_parameters(difficulty=np.zeros(1), thresholds=np.zeros(1))
        slope = 1.0
    theta = np.array([40.0 / slope])
    expected = slope**2 * expit(-40.0) * expit(40.0)
    np.testing.assert_allclose(
        model.information(theta, 0), expected, rtol=1e-14, atol=0.0
    )


@pytest.mark.parametrize(
    "model",
    [
        GeneralizedPartialCredit(2, [2, 4]),
        PartialCreditModel(2, [2, 4]),
        GeneralizedPartialCredit(2, [2, 4], n_factors=3),
    ],
)
def test_fisher_matrix_shapes_and_totals_include_pcm_and_empty_batches(model):
    theta = np.linspace(-2.0, 2.0, 11 * model.n_factors).reshape(11, model.n_factors)
    per_item = [model.item_information_matrix(theta, item) for item in range(2)]
    for item, matrix in enumerate(per_item):
        assert matrix.shape == (11, model.n_factors, model.n_factors)
        np.testing.assert_allclose(
            np.trace(matrix, axis1=1, axis2=2), model.information(theta, item)
        )
    np.testing.assert_allclose(model.test_information_matrix(theta), sum(per_item))
    empty = theta[:0]
    assert model.item_information_matrix(empty, 0).shape == (
        0,
        model.n_factors,
        model.n_factors,
    )
    assert model.test_information_matrix(empty).shape == (
        0,
        model.n_factors,
        model.n_factors,
    )


@pytest.mark.parametrize("slopes", [[1.7, 0.65], [1.3, -1.3], [0.0, 0.0]])
def test_estimation_gradient_and_curvature_match_independent_likelihood(slopes):
    rng = np.random.default_rng(289)
    slopes = np.asarray(slopes)
    steps = np.array([-0.9, 0.1, 1.2])
    model = GeneralizedPartialCredit(1, 4, n_factors=2)
    model.set_parameters(discrimination=slopes[None], steps=steps[None])
    theta = rng.normal(size=(37, 2))
    counts = rng.uniform(0.1, 10, size=(37, 4))
    objective = prepare_polytomous_objective(model, 0, theta, counts, 1e-12)
    assert objective is not None
    params = np.r_[slopes, steps]

    def reference(trial):
        return -np.sum(
            counts * np.log(_reference_probabilities(theta, trial[:2], trial[2:]))
        )

    value, gradient = objective(params)
    assert value == pytest.approx(reference(params), rel=1e-13)
    expected_gradient = np.zeros(5)
    expected_curvature = np.zeros(5)
    for index in range(5):
        shift = np.zeros(5)
        shift[index] = 1e-4
        before, after = reference(params - shift), reference(params + shift)
        expected_gradient[index] = (after - before) / 2e-4
        expected_curvature[index] = (after - 2 * value + before) / 1e-8
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-6)
    curvature = polytomous_item_curvature(model, 0, theta, counts, 1e-12)
    np.testing.assert_allclose(
        np.r_[curvature["discrimination"], curvature["steps"]],
        expected_curvature,
        rtol=2e-6,
        atol=1e-4,
    )


def test_estimation_recovers_parameters_from_reference_expected_counts():
    rng = np.random.default_rng(942)
    theta = rng.normal(size=(120, 3))
    slopes = np.array([1.7, 0.4, 0.8])
    steps = np.array([-1.1, 0.2, 1.3])
    counts = 500 * _reference_probabilities(theta, slopes, steps)
    model = GeneralizedPartialCredit(1, 4, n_factors=3)
    objective = prepare_polytomous_objective(model, 0, theta, counts, 1e-12)
    result = minimize(
        objective,
        np.r_[np.ones(3), [-0.7, 0.0, 0.7]],
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.05, 3.0)] * 3 + [(-3.0, 3.0)] * 3,
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-7},
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, np.r_[slopes, steps], rtol=1e-5, atol=2e-5)


def test_multidimensional_simulation_frequencies_agree_with_calibrated_model():
    points = np.array([[-0.8, 0.5], [0.3, -0.6], [0.8, 0.7]])
    slopes = np.array([[1.7, 0.6], [0.4, 1.3]])
    steps = np.array([[-0.8, 0.0, 0.9], [-1.1, 0.2, 1.2]])
    persons_per_point = 15_000
    responses = simdata(
        model="GPCM",
        theta=np.repeat(points, persons_per_point, axis=0),
        n_items=2,
        n_categories=4,
        n_factors=2,
        discrimination=slopes,
        steps=steps,
        seed=628,
    )
    model = GeneralizedPartialCredit(2, 4, n_factors=2)
    model.set_parameters(discrimination=slopes, steps=steps)
    expected = model.probability(points)
    for row in range(3):
        block = responses[row * persons_per_point : (row + 1) * persons_per_point]
        for item in range(2):
            observed = np.bincount(block[:, item], minlength=4) / persons_per_point
            # A deterministic six-standard-error envelope includes rare cells.
            tolerance = (
                6
                * np.sqrt(
                    expected[row, item] * (1 - expected[row, item]) / persons_per_point
                )
                + 1 / persons_per_point
            )
            assert np.all(np.abs(observed - expected[row, item]) <= tolerance)
