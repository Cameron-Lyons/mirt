"""Numerical and memory contracts for batched fitting and scoring paths."""

from __future__ import annotations

import weakref

import numpy as np
import pytest

import mirt
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models import (
    FourParameterLogistic,
    GeneralizedPartialCredit,
    GradedResponseModel,
    OneParameterLogistic,
    PartialCreditModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.scoring.map import MAPScorer
from mirt.scoring.ml import MLScorer

native = pytest.mark.skipif(
    not mirt.is_rust_available(), reason="native backend unavailable"
)


@pytest.fixture(autouse=True)
def restore_backend():
    previous = mirt.get_backend()
    yield
    mirt.set_backend(previous)


@native
@pytest.mark.parametrize("kind", ["2pl", "3pl", "mirt", "grm", "gpcm"])
def test_cached_likelihood_matches_independent_model(kind):
    from mirt.backends.rust import likelihood, polytomous

    rng = np.random.default_rng(819)
    points = np.linspace(-3, 3, 13)
    a = rng.uniform(0.5, 1.5, 4)
    b = rng.uniform(-1, 1, 4)
    responses = rng.integers(-1, 2, (67, 8), dtype=np.int32)[:, ::2]
    responses[0] = -19
    mirt.set_backend("rust")
    if kind == "mirt":
        points = np.column_stack((points, points[::-1]))
        model = TwoParameterLogistic(4, n_factors=2)
        a = np.column_stack((a, a[::-1]))
        model.set_parameters(discrimination=a, difficulty=b)
        result = likelihood.compute_log_likelihoods_mirt(responses, points, a, b)
    elif kind in ("2pl", "3pl"):
        if kind == "2pl":
            model = TwoParameterLogistic(4).set_parameters(
                discrimination=a, difficulty=b
            )
            result = likelihood.compute_log_likelihoods_2pl(responses, points, a, b)
        else:
            c = np.linspace(0.1, 0.3, 4)
            model = ThreeParameterLogistic(4).set_parameters(
                discrimination=a, difficulty=b, guessing=c
            )
            result = likelihood.compute_log_likelihoods_3pl(responses, points, a, b, c)
    else:
        categories = np.array([2, 5, 3, 4])
        responses = np.column_stack([rng.integers(-1, k, 67) for k in categories])
        responses[0] = -99
        factory = GradedResponseModel if kind == "grm" else GeneralizedPartialCredit
        model = factory(4, n_categories=categories.tolist())
        model.set_parameters(discrimination=a)
        if kind == "grm":
            result = polytomous.compute_log_likelihoods_grm(
                responses, points, a, model.parameters["thresholds"], categories
            )
        else:
            steps = np.column_stack((np.zeros(4), model.parameters["steps"]))
            result = polytomous.compute_log_likelihoods_gpcm(
                responses, points, a, steps, categories
            )
    # Evaluate each theta independently through the model, not the cached kernel.
    theta = np.asarray(points).reshape(len(points), -1)
    expected = np.column_stack(
        [model.log_likelihood(responses, q[None, :]) for q in theta]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@native
@pytest.mark.parametrize(
    "factory",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
    ],
)
@pytest.mark.parametrize("method", ["MAP", "ML"])
def test_native_optimizer_scores_match_scipy(factory, method):
    rng = np.random.default_rng(83)
    model = factory(9)
    model.set_parameters(difficulty=np.linspace(-1, 1, 9))
    model._is_fitted = True
    responses = rng.integers(-1, 2, (43, 9))
    responses[:3] = np.array([-1, 0, 1])[:, None]
    responses = np.concatenate([responses, responses[:4]])
    if method == "MAP":
        scorer = MAPScorer(
            prior_mean=np.array([0.6]),
            prior_cov=np.array([[1.7]]),
            theta_bounds=(-2, 3),
        )
    else:
        scorer = MLScorer(theta_bounds=(-2, 3))
    mirt.set_backend("numpy")
    expected = scorer.score(model, responses)
    mirt.set_backend("rust")
    actual = scorer.score(model, responses)
    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        actual.standard_error, expected.standard_error, atol=2e-4, rtol=2e-4
    )
    scorer.n_jobs = 2
    parallel = scorer.score(model, responses)
    np.testing.assert_array_equal(actual.theta, parallel.theta)
    np.testing.assert_array_equal(actual.standard_error, parallel.standard_error)


@native
@pytest.mark.parametrize("scorer", [MAPScorer(), MLScorer()])
def test_native_scoring_preserves_search_on_nonconcave_likelihood(scorer):
    model = FourParameterLogistic(8).set_parameters(
        discrimination=[
            0.8343,
            0.4359,
            -2.4143,
            -1.1776,
            0.1675,
            1.3614,
            2.5753,
            2.9783,
        ],
        difficulty=[
            1.6616,
            -0.7611,
            2.3583,
            -0.4814,
            -2.6107,
            -1.9353,
            2.3169,
            -2.3716,
        ],
        guessing=[0.0690, 0.3894, 0.1462, 0.3542, 0.1844, 0.0916, 0.0036, 0.3728],
        upper=[0.8099, 0.6227, 0.7565, 0.7068, 0.8050, 0.6864, 0.7678, 0.6656],
    )
    model._is_fitted = True
    responses = np.array([[0, 0, 1, 1, 1, 0, 0, 1]])
    mirt.set_backend("numpy")
    expected = scorer.score(model, responses)
    mirt.set_backend("rust")
    actual = scorer.score(model, responses)
    np.testing.assert_allclose(actual.theta, expected.theta, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        actual.standard_error, expected.standard_error, atol=2e-4, rtol=2e-4
    )


@native
def test_optimizer_batches_unique_patterns_and_preserves_custom_models(monkeypatch):
    from mirt.backends.rust import optimization_scoring

    model = TwoParameterLogistic(2)
    model._is_fitted = True
    seen = []
    original = optimization_scoring.mirt_rs.compute_optimized_scores

    def capture(responses, *args):
        seen.append(responses.copy())
        return original(responses, *args)

    monkeypatch.setattr(
        optimization_scoring.mirt_rs, "compute_optimized_scores", capture
    )
    data = np.array([[1, -1], [1, -9], [0, 1], [0, 1]])
    MAPScorer().score(model, data)
    assert seen[0].shape == (2, 2)
    assert seen[0].min() == -1
    monkeypatch.setattr(
        model, "log_likelihood", lambda responses, theta: -((theta[:, 0] - 0.4) ** 2)
    )
    result = MLScorer().score(model, np.array([[1, 0]]))
    assert result.theta[0] == pytest.approx(0.4, abs=1e-5)
    assert len(seen) == 1


@native
@pytest.mark.parametrize(
    "factory", [GradedResponseModel, GeneralizedPartialCredit, PartialCreditModel]
)
def test_native_polytomous_mstep_matches_scipy_objective(factory):
    from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step

    rng = np.random.default_rng(18)
    model = factory(3, n_categories=[2, 4, 3])
    reference = model.copy()
    points = np.linspace(-3, 3, 11)[:, None]
    responses = np.column_stack([rng.integers(-1, k, 100) for k in model.n_categories])
    responses[:, -1] = -1
    posterior = rng.random((100, 11))
    posterior /= posterior.sum(axis=1, keepdims=True)
    before = model.parameters
    mirt.set_backend("rust")
    assert try_polytomous_m_step(
        model,
        responses,
        posterior,
        points,
        max_iter=150,
        ftol=1e-11,
        epsilon=1e-10,
        n_jobs=2,
    )
    estimator = EMEstimator(
        item_optim_maxiter=150, item_optim_ftol=1e-11, use_rust=False
    )
    for j in range(2):
        optimized = estimator._optimize_item_params(
            reference, j, responses, posterior, points, posterior.sum(axis=0)
        )
        estimator._set_item_params(reference, j, optimized)
        counts = np.column_stack(
            [
                posterior[responses[:, j] == k].sum(axis=0)
                for k in range(model.n_categories[j])
            ]
        )

        def loss(fitted):
            return -np.sum(
                counts
                * np.log(np.clip(fitted.probability(points, j), 1e-10, 1 - 1e-10))
            )

        assert loss(model) <= loss(reference) + 1e-5
    for name, values in before.items():
        np.testing.assert_array_equal(model.parameters[name][-1], values[-1])
    if factory is PartialCreditModel:
        np.testing.assert_array_equal(model.parameters["discrimination"], np.ones(3))


@native
@pytest.mark.parametrize("kind", ["2PL", "3PL", "GRM", "GPCM", "PCM"])
def test_compressed_em_matches_expanded_fit(monkeypatch, kind):
    import mirt.estimation._patterns as patterns

    rng = np.random.default_rng(72)
    poly = kind in ("GRM", "GPCM", "PCM")
    pool = rng.integers(-1, 4 if poly else 2, (12, 5))
    responses = np.repeat(pool, rng.integers(22, 38, len(pool)), axis=0)
    mirt.set_backend("rust")
    kwargs = dict(model=kind, n_quadpts=11, max_iter=4, tol=1e-12)
    if poly:
        kwargs["n_categories"] = 4
    compressed = mirt.fit_mirt(responses, **kwargs)
    monkeypatch.setattr(patterns, "compress_responses", lambda values: (values, None))
    expanded = mirt.fit_mirt(responses, **kwargs)
    assert compressed.n_observations == responses.shape[0]
    assert compressed.log_likelihood == pytest.approx(expanded.log_likelihood, abs=2e-5)
    assert compressed.bic == pytest.approx(expanded.bic, abs=4e-5)
    for name, values in compressed.model.parameters.items():
        np.testing.assert_allclose(
            values, expanded.model.parameters[name], atol=3e-5, rtol=3e-5
        )
        np.testing.assert_allclose(
            compressed.standard_errors[name],
            expanded.standard_errors[name],
            atol=3e-4,
            rtol=3e-4,
        )


def test_pattern_compression_skips_distinct_and_custom_models():
    from mirt.estimation._patterns import (
        compress_responses,
        supports_pattern_compression,
    )

    values = np.random.default_rng(71).integers(0, 2, (2000, 30))
    output, counts = compress_responses(values)
    assert output is values
    assert counts is None

    class Custom(TwoParameterLogistic):
        pass

    assert not supports_pattern_compression(Custom(2))
    repeated = np.tile([[0, -1], [1, 0]], (500, 1))
    repeated[::4, 1] = -999
    compressed, counts = compress_responses(repeated)
    np.testing.assert_array_equal(compressed, [[0, -1], [1, 0]])
    np.testing.assert_array_equal(counts, [500, 500])


def test_information_retains_constant_number_of_person_arrays(monkeypatch):
    import mirt.estimation.standard_errors as se

    model = TwoParameterLogistic(5)
    original_parameters = model.parameters
    responses = np.random.default_rng(91).integers(0, 2, (100, 5))
    quadrature = GaussHermiteQuadrature(n_points=7)
    original = se._marginal_log_likelihoods
    references = []
    peak_live = 0

    def track(*args):
        nonlocal peak_live
        value = original(*args)
        references.append(weakref.ref(value))
        peak_live = max(peak_live, sum(ref() is not None for ref in references))
        return value

    monkeypatch.setattr(se, "_marginal_log_likelihoods", track)
    information, _ = se._finite_difference_information(
        model, responses, quadrature, quadrature.weights, 1e-4
    )
    assert len(references) == 201
    assert peak_live <= 7
    np.testing.assert_array_equal(information, information.T)
    for name, values in original_parameters.items():
        np.testing.assert_array_equal(model.parameters[name], values)

    def fail(*args):
        raise RuntimeError("likelihood failed")

    monkeypatch.setattr(se, "_marginal_log_likelihoods", fail)
    with pytest.raises(RuntimeError, match="likelihood failed"):
        se._finite_difference_scores(
            model, responses, quadrature, quadrature.weights, 1e-4
        )
    for name, values in original_parameters.items():
        np.testing.assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(
    "factory",
    [
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
    ],
)
def test_analytic_em_curvature_matches_finite_difference_objective(factory):
    from mirt.estimation._item_information import item_standard_errors

    rng = np.random.default_rng(901)
    poly = factory in (
        GradedResponseModel,
        GeneralizedPartialCredit,
        PartialCreditModel,
    )
    model = factory(3, n_categories=[3, 4, 5]) if poly else factory(3)
    if factory is FourParameterLogistic:
        model.set_parameters(upper=np.full(3, 0.9))
    responses = (
        np.column_stack([rng.integers(-1, k, 250) for k in model.n_categories])
        if poly
        else rng.integers(-1, 2, (250, 3))
    )
    posterior = rng.random((250, 15))
    posterior /= posterior.sum(axis=1, keepdims=True)
    estimator = EMEstimator(se_step_size=1e-3)
    estimator._quadrature = GaussHermiteQuadrature(n_points=15)
    before = model.parameters
    analytic = item_standard_errors(
        model, responses, posterior, estimator._quadrature.nodes, 1e-10
    )
    numerical = estimator._compute_standard_errors(model, responses, posterior)
    for name, values in analytic.items():
        np.testing.assert_allclose(
            values, numerical[name], atol=1e-5, rtol=1e-5, equal_nan=True
        )
        np.testing.assert_array_equal(model.parameters[name], before[name])
