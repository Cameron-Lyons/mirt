"""Regression coverage for finite-difference standard-error methods."""

import numpy as np
import pytest

from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.standard_errors import (
    _posterior_from_model,
    compute_observed_information,
    compute_sandwich_se,
)
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic


def _balanced_problem(
    n_repeats: int = 50,
) -> tuple[
    OneParameterLogistic,
    np.ndarray,
    GaussHermiteQuadrature,
    np.ndarray,
]:
    model = OneParameterLogistic(n_items=1)
    responses = np.array([[0], [1]] * n_repeats, dtype=np.int_)
    quadrature = GaussHermiteQuadrature(n_points=21)
    posterior = _posterior_from_model(model, responses, quadrature)
    return model, responses, quadrature, posterior


def _marginal_log_likelihoods(
    difficulty: float,
    responses: np.ndarray,
    quadrature: GaussHermiteQuadrature,
    prior_mass: np.ndarray | None = None,
) -> np.ndarray:
    probability = 1.0 / (1.0 + np.exp(-(quadrature.nodes.ravel() - difficulty)))
    likelihood = np.where(
        responses[:, 0, None] == 1,
        probability[None, :],
        1.0 - probability[None, :],
    )
    mass = quadrature.weights if prior_mass is None else prior_mass
    return np.log(likelihood @ mass)


def test_forward_curvature_is_distinct_from_central() -> None:
    model, responses, quadrature, posterior = _balanced_problem(n_repeats=8)
    h = 0.35

    forward = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method="forward",
        step_size=h,
    )
    expected_correct = (responses[:, 0, None] * posterior).sum(axis=0)
    expected_total = posterior.sum(axis=0)

    def expected_complete_log_likelihood(difficulty: float) -> float:
        probability = 1.0 / (1.0 + np.exp(-(quadrature.nodes.ravel() - difficulty)))
        return float(
            expected_correct @ np.log(probability)
            + (expected_total - expected_correct) @ np.log1p(-probability)
        )

    center = expected_complete_log_likelihood(0.0)
    plus = expected_complete_log_likelihood(h)
    plus_two = expected_complete_log_likelihood(2.0 * h)
    expected = np.sqrt(-(h**2) / (plus_two - 2.0 * plus + center))

    np.testing.assert_allclose(forward["difficulty"][0], expected, rtol=1e-13)
    assert forward["discrimination"][0] == 0.0
    central = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method="central",
        step_size=h,
    )
    assert not np.isclose(forward["difficulty"][0], central["difficulty"][0])


def test_parallel_item_curvature_preserves_model_state() -> None:
    model = TwoParameterLogistic(n_items=4).set_parameters(
        discrimination=np.array([0.7, 1.1, 1.4, 0.9]),
        difficulty=np.array([-0.8, -0.2, 0.4, 1.0]),
    )
    responses = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int_,
    )
    quadrature = GaussHermiteQuadrature(n_points=15)
    posterior = _posterior_from_model(model, responses, quadrature)
    original = model.parameters

    serial = compute_se(
        model, responses, quadrature, posterior, method="central", n_jobs=1
    )
    parallel = compute_se(
        model, responses, quadrature, posterior, method="central", n_jobs=3
    )

    for name, values in original.items():
        np.testing.assert_array_equal(parallel[name], serial[name])
        np.testing.assert_array_equal(model.parameters[name], values)


def test_crossprod_uses_outer_product_of_marginal_scores() -> None:
    model, responses, quadrature, posterior = _balanced_problem()
    h = 1e-4
    result = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method="crossprod",
        step_size=h,
    )
    scores = (
        _marginal_log_likelihoods(h, responses, quadrature)
        - _marginal_log_likelihoods(-h, responses, quadrature)
    ) / (2.0 * h)

    np.testing.assert_allclose(
        result["difficulty"][0],
        1.0 / np.sqrt(scores @ scores),
        rtol=1e-10,
    )
    assert result["discrimination"][0] == 0.0


def test_sandwich_uses_weighted_bread_and_person_scores() -> None:
    model, responses, quadrature, posterior = _balanced_problem()
    h = 1e-4
    weights = np.linspace(0.5, 1.5, responses.shape[0])
    result = compute_sandwich_se(
        model,
        responses,
        posterior,
        quadrature,
        survey_weights=weights,
        h=h,
    )

    center = _marginal_log_likelihoods(0.0, responses, quadrature)
    plus = _marginal_log_likelihoods(h, responses, quadrature)
    minus = _marginal_log_likelihoods(-h, responses, quadrature)
    bread = -float(weights @ (plus - 2.0 * center + minus)) / h**2
    weighted_scores = weights * (plus - minus) / (2.0 * h)
    expected = np.sqrt((weighted_scores @ weighted_scores) / bread**2)

    np.testing.assert_allclose(result["difficulty"][0], expected, rtol=5e-7)


@pytest.mark.parametrize("method", ["louis", "oakes", "sem"])
def test_observed_methods_infer_shifted_prior_mass(method: str) -> None:
    model = OneParameterLogistic(n_items=1)
    model.difficulty[:] = 1.2
    responses = np.array([[0], [1]] * 40, dtype=np.int_)
    quadrature = GaussHermiteQuadrature(n_points=21)
    density = GaussianDensity(mean=np.array([1.2]), cov=np.array([[0.7**2]]))
    log_mass = density.log_quadrature_mass(quadrature.nodes, quadrature.weights)
    mass = np.exp(log_mass)
    posterior = _posterior_from_model(model, responses, quadrature, log_mass)
    h = 1e-4

    inferred = compute_observed_information(
        model, responses, posterior, quadrature, h=h
    )
    explicit = compute_observed_information(
        model,
        responses,
        posterior,
        quadrature,
        h=h,
        prior_mass=9.0 * mass,
    )
    result = compute_se(
        model,
        responses,
        quadrature,
        posterior,
        method=method,
        step_size=h,
    )

    np.testing.assert_allclose(inferred, explicit, rtol=5e-7)
    np.testing.assert_allclose(
        result["difficulty"][0], np.sqrt(1.0 / inferred[0, 0]), rtol=2e-5
    )
    assert result["discrimination"][0] == 0.0


def test_zero_prior_cells_require_explicit_mass() -> None:
    model, responses, quadrature, _ = _balanced_problem(n_repeats=2)
    mass = quadrature.weights
    mass[0] = 0.0
    mass /= mass.sum()
    log_mass = np.full_like(mass, -np.inf)
    positive = mass > 0.0
    log_mass[positive] = np.log(mass[positive])
    posterior = _posterior_from_model(model, responses, quadrature, log_mass)

    information = compute_observed_information(
        model,
        responses,
        posterior,
        quadrature,
        h=1e-4,
        prior_mass=mass,
    )
    assert np.isfinite(information[0, 0])
    with pytest.raises(ValueError, match="pass prior_mass explicitly"):
        compute_observed_information(model, responses, posterior, quadrature)
