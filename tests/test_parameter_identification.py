"""Regression coverage for fixed and padded item parameters."""

import numpy as np
import pytest

from mirt.estimation.bl import BLEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.standard_errors import (
    _flatten_parameters as flatten_se_parameters,
)
from mirt.estimation.standard_errors import (
    _set_flat_parameters as set_flat_se_parameters,
)
from mirt.estimation.standard_errors import (
    _unflatten_se,
    compute_observed_information,
)
from mirt.models.dichotomous import OneParameterLogistic
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.utils.extraction import estfun


@pytest.mark.parametrize(
    ("model", "expected_n_parameters"),
    [
        (OneParameterLogistic(3), 3),
        (GradedResponseModel(3, [2, 4, 3], n_factors=2), 12),
        (GeneralizedPartialCredit(3, [2, 4, 3], n_factors=2), 12),
        (PartialCreditModel(3, [2, 4, 3]), 6),
        (NominalResponseModel(3, [2, 4, 3], n_factors=2), 18),
    ],
)
def test_parameter_count_includes_only_free_values(model, expected_n_parameters):
    assert model.n_parameters == expected_n_parameters


def test_item_optimizer_excludes_fixed_and_padded_pcm_values():
    model = PartialCreditModel(3, [2, 4, 3])
    estimator = BLEstimator(n_quadpts=5, max_iter=1)

    parameters, bounds = estimator._get_item_params_and_bounds(model, item_idx=1)

    assert parameters.shape == (3,)
    assert bounds == [(-6.0, 6.0)] * 3

    estimator._set_item_params(model, item_idx=1, params=np.array([-0.8, 0.1, 1.2]))

    np.testing.assert_allclose(model.discrimination, 1.0)
    np.testing.assert_allclose(model.steps[1, :3], [-0.8, 0.1, 1.2])
    np.testing.assert_allclose(model.steps[1, 3:], 0.0)


def test_nrm_canonicalization_preserves_probabilities():
    model = NominalResponseModel(2, [2, 4], n_factors=2)
    model.set_parameters(
        slopes=np.array(
            [
                [[0.4, -0.3], [1.1, 0.2], [8.0, 8.0], [9.0, 9.0]],
                [[-0.2, 0.5], [0.3, -0.4], [1.2, 0.8], [-0.7, 1.4]],
            ]
        ),
        intercepts=np.array(
            [
                [0.3, -0.4, 8.0, 9.0],
                [-0.2, 0.6, -0.7, 1.1],
            ]
        ),
    )
    theta = np.array([[-1.0, 0.2], [0.0, 0.0], [0.8, -0.5]])
    expected = [model.probability(theta, item) for item in range(model.n_items)]
    estimator = BLEstimator(n_quadpts=5, max_iter=1)

    flattened, bounds, structure = estimator._flatten_parameters(model)
    estimator._unflatten_parameters(model, flattened, structure)

    assert bounds[:8] == [(-5.0, 5.0)] * 8
    for item, probabilities in enumerate(expected):
        np.testing.assert_allclose(model.probability(theta, item), probabilities)
    np.testing.assert_allclose(model.slopes[:, 0], 0.0)
    np.testing.assert_allclose(model.intercepts[:, 0], 0.0)
    np.testing.assert_allclose(model.slopes[0, 2:], 0.0)
    np.testing.assert_allclose(model.intercepts[0, 2:], 0.0)


def test_standard_error_layout_reconstructs_fixed_values_as_zero():
    model = PartialCreditModel(3, [2, 4, 3])

    flattened, layouts = flatten_se_parameters(model)
    estimates = np.linspace(-0.5, 0.5, model.n_parameters)
    set_flat_se_parameters(model, estimates, layouts)
    standard_errors = _unflatten_se(np.ones_like(flattened), layouts, model)

    assert flattened.shape == (model.n_parameters,)
    np.testing.assert_allclose(model.discrimination, 1.0)
    np.testing.assert_allclose(standard_errors["discrimination"], 0.0)
    for item, n_categories in enumerate(model.n_categories):
        np.testing.assert_allclose(model.steps[item, n_categories - 1 :], 0.0)
        np.testing.assert_allclose(
            standard_errors["steps"][item, n_categories - 1 :], 0.0
        )


def test_information_and_itemwise_se_use_free_parameter_layout():
    quadrature = GaussHermiteQuadrature(n_points=5)
    responses = np.array([[0, 0], [1, 1], [0, 1], [1, 2]])
    posterior = np.full((responses.shape[0], quadrature.n_points), 0.2)

    rasch = OneParameterLogistic(2)
    information = compute_observed_information(
        rasch, responses.clip(max=1), posterior, quadrature
    )
    assert information.shape == (rasch.n_parameters, rasch.n_parameters)
    np.testing.assert_allclose(rasch.discrimination, 1.0)

    pcm = PartialCreditModel(2, [2, 3])
    standard_errors = compute_se(
        pcm, responses, quadrature, posterior, method="central"
    )
    np.testing.assert_allclose(standard_errors["discrimination"], 0.0)
    np.testing.assert_allclose(standard_errors["steps"][0, 1:], 0.0)


def test_bl_pcm_fit_keeps_fixed_values_out_of_fit_statistics(rng):
    n_persons = 50
    n_categories = [2, 4, 3]
    responses = np.column_stack(
        [rng.integers(0, count, size=n_persons) for count in n_categories]
    )
    model = PartialCreditModel(3, n_categories)

    result = BLEstimator(n_quadpts=5, max_iter=2).fit(model, responses)

    assert result.n_parameters == 6
    assert result.aic == pytest.approx(-2 * result.log_likelihood + 12)
    assert result.bic == pytest.approx(
        -2 * result.log_likelihood + np.log(n_persons) * 6
    )
    np.testing.assert_allclose(model.discrimination, 1.0)
    np.testing.assert_allclose(result.standard_errors["discrimination"], 0.0)
    for item, count in enumerate(n_categories):
        np.testing.assert_allclose(model.steps[item, count - 1 :], 0.0)
        np.testing.assert_allclose(
            result.standard_errors["steps"][item, count - 1 :], 0.0
        )


def test_estfun_uses_free_nrm_parameter_layout():
    model = NominalResponseModel(2, [2, 4], n_factors=2)
    responses = np.array([[0, 0], [1, 3], [-1, -1]])
    theta = np.array([[-0.5, 0.2], [0.7, -0.3], [0.0, 0.0]])

    scores = estfun(model, responses, theta)

    assert scores.shape == (3, model.n_parameters)
    assert np.all(np.isfinite(scores))
    assert np.any(scores[:2] != 0.0)
    np.testing.assert_allclose(scores[2], 0.0)


def test_estfun_excludes_fixed_rasch_discriminations():
    model = OneParameterLogistic(2)
    responses = np.array([[0, 1], [1, 0], [-1, -1]])

    scores = estfun(model, responses, np.array([-0.5, 0.7, 0.0]))

    assert scores.shape == (3, model.n_parameters)
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores[2], 0.0)
