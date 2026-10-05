"""Regression tests for survey-weighted EM estimation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import mirt
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.weighted import (
    WeightedEMEstimator,
    compute_design_effect,
    compute_effective_sample_size,
)
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.fixture
def responses() -> np.ndarray:
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0]])


@pytest.mark.parametrize("normalize_weights", [True, False])
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.zeros(4), "at least one positive"),
        (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
        (np.array([1.0, np.inf, 1.0, 1.0]), "finite"),
        (np.array([1.0, -1.0, 1.0, 1.0]), "non-negative"),
    ],
)
def test_fit_rejects_invalid_weights(
    responses: np.ndarray,
    normalize_weights: bool,
    weights: np.ndarray,
    message: str,
) -> None:
    estimator = WeightedEMEstimator(
        n_quadpts=5,
        max_iter=2,
        normalize_weights=normalize_weights,
    )

    with pytest.raises(ValueError, match=message):
        estimator.fit(TwoParameterLogistic(2), responses, weights=weights)


@pytest.mark.parametrize(
    "function", [compute_effective_sample_size, compute_design_effect]
)
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([]), "at least one positive"),
        (np.zeros(3), "at least one positive"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, -1.0]), "non-negative"),
    ],
)
def test_weight_summaries_reject_invalid_values(
    function: Callable[[np.ndarray], float],
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(weights)


@pytest.mark.parametrize("normalize_weights", [None, 1, "yes"])
def test_normalize_weights_requires_boolean(normalize_weights: Any) -> None:
    with pytest.raises(ValueError, match="normalize_weights must be a boolean"):
        WeightedEMEstimator(normalize_weights=normalize_weights)


def test_weight_summary_values() -> None:
    weights = np.array([1.0, 2.0, 1.0])

    assert compute_effective_sample_size(weights) == pytest.approx(8.0 / 3.0)
    assert compute_design_effect(weights) == pytest.approx(9.0 / 8.0)


def test_extreme_finite_weights_do_not_overflow(responses: np.ndarray) -> None:
    weights = np.full(4, np.finfo(np.float64).max)

    assert compute_effective_sample_size(weights) == pytest.approx(4.0)
    assert compute_design_effect(weights) == pytest.approx(1.0)

    estimator = WeightedEMEstimator(n_quadpts=5, max_iter=1)
    estimator.fit(TwoParameterLogistic(2), responses, weights=weights)

    assert np.all(np.isfinite(estimator._weights))
    assert estimator._weights.sum() == pytest.approx(4.0)


def test_exhausted_fit_statistics_describe_returned_model(
    responses: np.ndarray,
) -> None:
    survey_weights = np.array([0.5, 1.0, 1.5, 2.0])
    estimator = WeightedEMEstimator(
        n_quadpts=7,
        max_iter=1,
        normalize_weights=False,
    )
    result = estimator.fit(TwoParameterLogistic(2), responses, weights=survey_weights)

    _, log_marginal = estimator._e_step_weighted(
        result.model,
        responses,
        np.zeros(1),
        np.eye(1),
        survey_weights,
    )
    expected_ll = float(survey_weights @ log_marginal)
    effective_n = survey_weights.sum() ** 2 / np.sum(survey_weights**2)

    assert result.log_likelihood == pytest.approx(expected_ll, abs=1e-12)
    assert result.aic == pytest.approx(-2 * expected_ll + 2 * result.n_parameters)
    assert result.bic == pytest.approx(
        -2 * expected_ll + np.log(effective_n) * result.n_parameters
    )
    assert estimator.convergence_history[-1] == pytest.approx(expected_ll)


def test_convergence_on_last_allowed_iteration_is_reported(
    responses: np.ndarray,
) -> None:
    result = WeightedEMEstimator(
        n_quadpts=7,
        max_iter=2,
        tol=1e9,
    ).fit(TwoParameterLogistic(2), responses)

    assert result.n_iterations == 2
    assert result.converged is True


def _reference_e_step(model, responses, quadrature, mean, covariance):
    # Evaluate each grid point through the scalar likelihood and construct prior
    # masses independently from normal density ratios.
    points = quadrature.nodes
    log_mass = (
        np.log(quadrature.weights)
        + multivariate_normal.logpdf(points, mean=mean, cov=covariance)
        - multivariate_normal.logpdf(points, mean=np.zeros(len(mean)))
    )
    log_mass -= logsumexp(log_mass)
    log_joint = (
        np.column_stack(
            [
                model.log_likelihood(responses, np.tile(point, (len(responses), 1)))
                for point in points
            ]
        )
        + log_mass
    )
    marginal = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - marginal[:, None]), marginal


@pytest.fixture(params=["numpy", "rust"])
def backend(request):
    if request.param == "rust" and not mirt.is_rust_available():
        pytest.skip("native backend unavailable")
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    yield request.param
    mirt.set_backend(previous)


@pytest.mark.parametrize(
    "kind", ["1PL", "2PL", "3PL", "4PL", "MIRT", "GRM", "GPCM", "PCM", "NRM"]
)
def test_weighted_e_step_matches_scalar_reference(kind, backend):
    if kind == "MIRT":
        model = mirt.TwoParameterLogistic(4, n_factors=2)
    else:
        factories = {
            "1PL": mirt.OneParameterLogistic,
            "2PL": mirt.TwoParameterLogistic,
            "3PL": mirt.ThreeParameterLogistic,
            "4PL": mirt.FourParameterLogistic,
            "GRM": mirt.GradedResponseModel,
            "GPCM": mirt.GeneralizedPartialCredit,
            "PCM": mirt.PartialCreditModel,
            "NRM": mirt.NominalResponseModel,
        }
        options = (
            {"n_categories": [2, 5, 3, 4]}
            if kind in {"GRM", "GPCM", "PCM", "NRM"}
            else {}
        )
        model = factories[kind](4, **options)
    rng = np.random.default_rng(917)
    categories = model._n_categories if model.is_polytomous else [2] * 4
    responses = np.column_stack([rng.integers(-1, k, 26) for k in categories])[::2]
    responses[0] = -99
    responses[:, -1] = -7
    responses.setflags(write=False)
    mean = np.full(model.n_factors, 0.4)
    covariance = np.eye(model.n_factors) * 0.8
    if model.n_factors == 2:
        covariance[0, 1] = covariance[1, 0] = 0.25
    weights = rng.uniform(0.0, 3.0, len(responses))
    weights[1] = 0.0
    weights.setflags(write=False)
    estimator = WeightedEMEstimator(n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(7, model.n_factors)
    expected = _reference_e_step(
        model, responses, estimator._quadrature, mean, covariance
    )
    actual = estimator._e_step_weighted(model, responses, mean, covariance, weights)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual[0].sum(axis=1), 1.0, atol=1e-14)
    # Sampling weights affect the weighted objective and M-step, not individual
    # posterior distributions.
    unweighted = estimator._e_step_weighted(
        model, responses, mean, covariance, np.ones(len(responses))
    )
    np.testing.assert_array_equal(actual[0], unweighted[0])


@pytest.mark.parametrize("max_iter", [1, 3])
def test_weighted_long_test_fit_retains_log_marginals(monkeypatch, max_iter):
    responses = np.tile([0, 1], (4, 600))
    model = TwoParameterLogistic(1200)
    weights = np.array([0.0, 0.5, 1.5, 2.0])
    estimator = WeightedEMEstimator(
        n_quadpts=5, max_iter=max_iter, normalize_weights=False
    )
    # Isolate fit reporting and convergence from optimization and uncertainty.
    monkeypatch.setattr(estimator, "_m_step_weighted", lambda *args: None)
    monkeypatch.setattr(
        estimator, "_compute_weighted_standard_errors", lambda *args: {}
    )
    result = estimator.fit(model, responses, weights)
    posterior, log_marginal = _reference_e_step(
        model, responses, estimator._quadrature, np.zeros(1), np.eye(1)
    )
    expected = float(weights @ log_marginal)
    assert np.all(log_marginal < -800)
    assert np.all(np.isfinite(posterior))
    assert result.log_likelihood == pytest.approx(expected, abs=1e-8)
    assert result.aic == pytest.approx(-2 * expected + 2 * model.n_parameters)
    assert result.bic == pytest.approx(
        -2 * expected
        + np.log(compute_effective_sample_size(weights)) * model.n_parameters
    )
    np.testing.assert_allclose(estimator.convergence_history, expected, atol=1e-8)
    assert result.converged is (max_iter > 1)


@pytest.mark.parametrize("override", ["subclass", "instance"])
def test_weighted_e_step_preserves_custom_person_likelihood(override):
    class PersonLikelihood(TwoParameterLogistic):
        def log_likelihood(self, responses, theta):
            assert theta.shape == (len(responses), self.n_factors)
            return (
                TwoParameterLogistic.log_likelihood(self, responses, theta)
                - np.arange(len(responses)) * theta[:, 0] ** 2
            )

    model = PersonLikelihood(2)
    if override == "instance":
        model = TwoParameterLogistic(2)
        model.log_likelihood = PersonLikelihood.log_likelihood.__get__(model)
    responses = np.array([[0, 1], [1, 0], [1, 1]])
    estimator = WeightedEMEstimator(n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(7)
    expected = _reference_e_step(
        model, responses, estimator._quadrature, np.zeros(1), np.eye(1)
    )
    actual = estimator._e_step_weighted(
        model, responses, np.zeros(1), np.eye(1), np.ones(3)
    )
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-14)


def test_weighted_standard_errors_share_counts_without_copying_posterior(monkeypatch):
    class CustomModel(TwoParameterLogistic):
        pass

    estimator = WeightedEMEstimator(n_quadpts=5)
    estimator._quadrature = GaussHermiteQuadrature(5)
    responses = np.array([[0, 1], [1, 0], [1, 1]])
    posterior = np.full((3, 5), 0.2)
    weights = np.array([0.0, 0.5, 2.0])
    seen = []

    def item_se(model, item, name, data, original_posterior, *, r_k, n_k_valid, r_kc):
        assert original_posterior is posterior
        weighted = posterior * weights[:, None]
        np.testing.assert_allclose(r_k, responses[:, item] @ weighted)
        np.testing.assert_allclose(n_k_valid, weighted.sum(axis=0))
        assert r_kc is None
        seen.append(r_k)
        return 0.5

    monkeypatch.setattr(estimator, "_compute_item_se", item_se)
    result = estimator._compute_weighted_standard_errors(
        CustomModel(2), responses, posterior, weights
    )
    assert len(seen) == 4
    assert np.shares_memory(seen[0], seen[2])
    assert np.shares_memory(seen[1], seen[3])
    np.testing.assert_array_equal(posterior, np.full((3, 5), 0.2))
    for value in result.values():
        np.testing.assert_array_equal(value, [0.5, 0.5])


@pytest.mark.parametrize(
    "factory",
    [
        mirt.OneParameterLogistic,
        mirt.TwoParameterLogistic,
        mirt.ThreeParameterLogistic,
        mirt.FourParameterLogistic,
        mirt.GradedResponseModel,
        mirt.GeneralizedPartialCredit,
        mirt.PartialCreditModel,
    ],
)
def test_weighted_standard_errors_match_likelihood_curvature(factory):
    poly = factory in (
        mirt.GradedResponseModel,
        mirt.GeneralizedPartialCredit,
        mirt.PartialCreditModel,
    )
    model = factory(2, n_categories=[2, 4]) if poly else factory(2)
    if factory is mirt.FourParameterLogistic:
        model.set_parameters(upper=np.full(2, 0.9))
    estimator = WeightedEMEstimator(n_quadpts=7)
    estimator._quadrature = GaussHermiteQuadrature(7)
    points = estimator._quadrature.nodes
    rng = np.random.default_rng(918)
    categories = model._n_categories if model.is_polytomous else [2] * 2
    responses = np.column_stack([rng.integers(-1, k, 40) for k in categories])
    posterior = rng.uniform(0.1, 1.0, (40, 7))
    posterior /= posterior.sum(axis=1, keepdims=True)
    weights = rng.uniform(0.5, 3.0, 40)
    weights[::7] = 0.0
    params = model.parameters
    actual = estimator._compute_weighted_standard_errors(
        model, responses, posterior, weights
    )

    def objective():
        likelihoods = np.column_stack(
            [model.log_likelihood(responses, point[None, :]) for point in points]
        )
        return float(np.sum(weights[:, None] * posterior * likelihoods))

    center = objective()
    step = 1e-3
    for name, values in params.items():
        for index in np.ndindex(values.shape):
            if not model.free_parameter_masks[name][index]:
                assert actual[name][index] == 0.0
                continue
            plus = values.copy()
            minus = values.copy()
            plus[index] += step
            minus[index] -= step
            model.set_parameters(**{name: plus})
            high = objective()
            model.set_parameters(**{name: minus})
            low = objective()
            model.set_parameters(**{name: values})
            curvature = (high - 2 * center + low) / step**2
            expected = np.sqrt(-1.0 / curvature) if curvature < 0 else np.nan
            np.testing.assert_allclose(
                actual[name][index], expected, rtol=2e-4, atol=1e-6
            )
    for name, values in params.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_custom_readonly_likelihood_buffer_is_preserved():
    cached = np.array([-1.0, -2.0, -3.0])
    cached.setflags(write=False)
    model = TwoParameterLogistic(2)
    model.log_likelihood = lambda *args: cached
    estimator = WeightedEMEstimator(n_quadpts=5)
    estimator._quadrature = GaussHermiteQuadrature(5)
    posterior, log_marginal = estimator._e_step_weighted(
        model, np.ones((3, 2), dtype=int), np.zeros(1), np.eye(1), np.ones(3)
    )
    np.testing.assert_array_equal(cached, [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(log_marginal, cached, atol=1e-14)
    np.testing.assert_allclose(
        posterior, np.tile(estimator._quadrature.weights, (3, 1)), atol=1e-14
    )


@pytest.mark.parametrize("kind", ["2PL", "GRM"])
def test_weighted_fit_matches_scalar_e_step(kind, backend):
    class ScalarEstimator(WeightedEMEstimator):
        def _e_step_weighted(self, model, responses, mean, covariance, weights):
            return _reference_e_step(
                model, responses, self._quadrature, mean, covariance
            )

    rng = np.random.default_rng(829)
    model = (
        TwoParameterLogistic(4)
        if kind == "2PL"
        else mirt.GradedResponseModel(4, n_categories=[2, 3, 4, 5])
    )
    categories = model._n_categories if model.is_polytomous else [2] * 4
    responses = np.column_stack([rng.integers(-1, k, 80) for k in categories])
    weights = rng.uniform(0.5, 2.0, 80)
    weights[::9] = 0.0
    options = dict(n_quadpts=7, max_iter=3, tol=1e-12, normalize_weights=False)
    prior = dict(prior_mean=np.array([0.3]), prior_cov=np.array([[0.8]]))
    expected = ScalarEstimator(**options).fit(model.copy(), responses, weights, **prior)
    actual = WeightedEMEstimator(**options).fit(
        model.copy(), responses, weights, **prior
    )
    assert actual.log_likelihood == pytest.approx(expected.log_likelihood, abs=1e-5)
    assert actual.n_iterations == expected.n_iterations
    for name, values in expected.model.parameters.items():
        np.testing.assert_allclose(
            actual.model.parameters[name], values, atol=1e-4, rtol=1e-4
        )
        np.testing.assert_allclose(
            actual.standard_errors[name],
            expected.standard_errors[name],
            atol=1e-4,
            rtol=1e-3,
        )


def test_weighted_fit_reports_its_standard_error_method():
    responses = mirt.simdata("2PL", n_persons=200, n_items=4, seed=5)
    weights = np.random.default_rng(5).uniform(0.5, 2.0, 200)
    result = WeightedEMEstimator(max_iter=50).fit(
        TwoParameterLogistic(4), responses, weights
    )
    assert result.se_method == "complete_data"
    assert result.vcov is None
    assert np.all(result.standard_errors["discrimination"] > 0)

    estimator = WeightedEMEstimator(
        max_iter=50, compute_standard_errors=False, item_optim_ftol=1e-9
    )
    assert estimator.item_optim_ftol == 1e-9
    plain = estimator.fit(TwoParameterLogistic(4), responses, weights)
    assert plain.standard_errors == {}
    assert plain.se_method is None
    with pytest.raises(mirt.MirtValidationError, match="compute_standard_errors"):
        WeightedEMEstimator(compute_standard_errors="yes")
