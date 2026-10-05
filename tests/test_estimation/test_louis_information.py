"""Exact Louis observed information and fitted-model covariance contracts."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
import mirt.estimation._louis_information as louis
from mirt.estimation.base import _parameter_bounds
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.standard_errors import (
    _finite_difference_information,
    _finite_difference_scores,
    _flatten_parameters,
    _marginal_log_likelihoods,
    _posterior_from_model,
    compute_crossprod_se,
    compute_expected_information,
    compute_oakes_se,
    compute_observed_information,
    compute_sandwich_se,
    estimate_covariance,
)
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)

CATEGORIES = [2, 3, 4, 5, 4]


def _binary_model(factory, rng):
    model = factory(5)
    values = {"difficulty": rng.normal(0.0, 0.8, 5)}
    if factory is not OneParameterLogistic:
        values["discrimination"] = rng.uniform(0.7, 1.8, 5)
    if factory in (ThreeParameterLogistic, FourParameterLogistic):
        values["guessing"] = rng.uniform(0.08, 0.22, 5)
    if factory is FourParameterLogistic:
        values["upper"] = rng.uniform(0.86, 0.96, 5)
    return model.set_parameters(**values)


def _polytomous_model(factory, rng):
    model = factory(5, n_categories=CATEGORIES)
    values = {}
    for name, current in model.parameters.items():
        mask = model.free_parameter_masks[name]
        noise = rng.uniform(-0.25, 0.25, current.shape)
        values[name] = np.where(mask, current + noise, current)
    if factory is GradedResponseModel:
        for item, count in enumerate(CATEGORIES):
            values["thresholds"][item, : count - 1].sort()
    if factory is PartialCreditModel:
        values.pop("discrimination")
    return model.set_parameters(**values)


def _simulate(model, theta, rng):
    """Draw responses by inverting each item's category distribution."""
    columns = []
    for item in range(model.n_items):
        probability = np.asarray(model.probability(theta, item))
        if probability.ndim == 1:
            probability = np.column_stack((1.0 - probability, probability))
        cumulative = np.cumsum(probability, axis=1)
        draws = rng.random((theta.shape[0], 1))
        columns.append(
            np.minimum((draws > cumulative).sum(axis=1), probability.shape[1] - 1)
        )
    return np.column_stack(columns)


def _responses(model, rng, n_persons=240, missing=0.08):
    theta = rng.standard_normal((n_persons, model.n_factors))
    responses = _simulate(model, theta, rng)
    responses[rng.random(responses.shape) < missing] = -1
    return responses


def _model_and_data(name, rng):
    factories = {
        "1PL": OneParameterLogistic,
        "2PL": TwoParameterLogistic,
        "3PL": ThreeParameterLogistic,
        "4PL": FourParameterLogistic,
        "GRM": GradedResponseModel,
        "GPCM": GeneralizedPartialCredit,
        "PCM": PartialCreditModel,
        "NRM": NominalResponseModel,
    }
    if name == "2PL-2D":
        model = TwoParameterLogistic(5, n_factors=2).set_parameters(
            discrimination=rng.uniform(0.5, 1.5, (5, 2)),
            difficulty=rng.normal(0.0, 0.8, 5),
        )
    elif name in ("1PL", "2PL", "3PL", "4PL"):
        model = _binary_model(factories[name], rng)
    else:
        model = _polytomous_model(factories[name], rng)
    return model, _responses(model, rng)


MODELS = ["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM", "2PL-2D"]


@pytest.mark.parametrize("name", MODELS)
def test_louis_information_matches_marginal_finite_differences(name):
    rng = np.random.default_rng(MODELS.index(name) + 71)
    model, responses = _model_and_data(name, rng)
    quadrature = GaussHermiteQuadrature(
        9 if model.n_factors == 1 else 7, model.n_factors
    )
    # A shifted, narrower latent prior exercises a non-default prior mass.
    density = GaussianDensity(
        mean=np.full(model.n_factors, 0.3), cov=np.eye(model.n_factors) * 0.8
    )
    mass = np.exp(density.log_quadrature_mass(quadrature.nodes, quadrature.weights))
    mass /= mass.sum()
    weights = rng.uniform(0.5, 2.0, responses.shape[0])
    _, layouts = _flatten_parameters(model)
    before = model.parameters

    terms = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts, person_weights=weights
    )
    reference, _ = _finite_difference_information(
        model, responses, quadrature, mass, 1e-4, person_weights=weights
    )
    scores, _ = _finite_difference_scores(model, responses, quadrature, mass, 1e-5)

    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(terms.information, reference, rtol=0, atol=2e-6 * scale)
    np.testing.assert_allclose(
        terms.score_crossproduct,
        (scores * weights[:, None]).T @ scores,
        rtol=0,
        atol=1e-7 * np.max(np.abs(terms.score_crossproduct)),
    )
    assert louis.has_analytic_item_derivatives(model) == (name not in ("NRM", "2PL-2D"))
    for parameter, values in before.items():
        np.testing.assert_array_equal(model.parameters[parameter], values)


@pytest.mark.parametrize("name", ["2PL", "4PL", "2PL-2D"])
def test_factored_binary_scores_match_materialized_scores(name, monkeypatch):
    rng = np.random.default_rng(5)
    model, responses = _model_and_data(name, rng)
    quadrature = GaussHermiteQuadrature(
        9 if model.n_factors == 1 else 5, model.n_factors
    )
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)
    weights = rng.uniform(0.5, 2.0, responses.shape[0])
    factored = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts, person_weights=weights
    )
    monkeypatch.setattr(
        louis,
        "_score_accumulator",
        lambda model, terms, nodes: louis._CategoricalScores(terms),
    )
    materialized = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts, person_weights=weights
    )
    # Differenced derivatives satisfy the binary score factorization only to
    # their truncation error; closed forms satisfy it exactly.
    rtol = 1e-6 if name == "2PL-2D" else 1e-11
    np.testing.assert_allclose(
        factored.information, materialized.information, rtol=rtol, atol=1e-10
    )
    np.testing.assert_allclose(
        factored.score_crossproduct,
        materialized.score_crossproduct,
        rtol=rtol,
        atol=1e-10,
    )


@pytest.mark.parametrize("name", ["2PL", "GRM"])
def test_person_blocks_and_frequencies_do_not_change_information(name, monkeypatch):
    rng = np.random.default_rng(13)
    model, responses = _model_and_data(name, rng)
    quadrature = GaussHermiteQuadrature(11)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)
    unique, inverse, counts = np.unique(
        responses, axis=0, return_inverse=True, return_counts=True
    )
    expanded = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts
    )
    monkeypatch.setattr(louis, "_MAX_BLOCK_ENTRIES", 64)
    compressed = louis.louis_information(
        model,
        unique,
        quadrature.nodes,
        mass,
        layouts,
        person_weights=counts.astype(float),
    )
    assert inverse.size == responses.shape[0]
    np.testing.assert_allclose(compressed.information, expanded.information, rtol=1e-10)
    np.testing.assert_allclose(
        compressed.score_crossproduct, expanded.score_crossproduct, rtol=1e-10
    )


def test_public_matrix_methods_use_exact_terms():
    rng = np.random.default_rng(29)
    model, _ = _model_and_data("GRM", rng)
    responses = _responses(model, rng, 1500)
    quadrature = GaussHermiteQuadrature(11)
    posterior = _posterior_from_model(model, responses, quadrature)
    _, layouts = _flatten_parameters(model)
    mass = quadrature.weights / quadrature.weights.sum()
    weights = rng.uniform(0.5, 1.5, responses.shape[0])

    information = compute_observed_information(model, responses, posterior, quadrature)
    exact = louis.louis_information(model, responses, quadrature.nodes, mass, layouts)
    np.testing.assert_allclose(information, exact.information, rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(exact.information) > 0.0)

    oakes = compute_oakes_se(model, responses, posterior, quadrature)
    expected = np.sqrt(np.diag(np.linalg.inv(exact.information)))
    np.testing.assert_allclose(oakes["discrimination"], expected[:5], rtol=1e-10)
    crossprod = compute_crossprod_se(model, responses, posterior, quadrature)
    expected = np.sqrt(np.diag(np.linalg.inv(exact.score_crossproduct)))
    np.testing.assert_allclose(crossprod["discrimination"], expected[:5], rtol=1e-10)

    sandwich = compute_sandwich_se(
        model, responses, posterior, quadrature, survey_weights=weights
    )
    weighted = louis.louis_information(
        model,
        responses,
        quadrature.nodes,
        mass,
        layouts,
        person_weights=weights,
        meat_weights=weights**2,
    )
    bread = np.linalg.inv(weighted.information)
    covariance = bread @ weighted.score_crossproduct @ bread
    np.testing.assert_allclose(
        sandwich["discrimination"], np.sqrt(np.diag(covariance))[:5], rtol=1e-10
    )
    for method in ("louis", "oakes", "sem"):
        result = compute_se(model, responses, quadrature, posterior, method=method)
        np.testing.assert_allclose(result["thresholds"], oakes["thresholds"])


def test_exact_terms_accept_float_coded_responses():
    rng = np.random.default_rng(31)
    model, responses = _model_and_data("GRM", rng)
    quadrature = GaussHermiteQuadrature(9)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)

    integer = louis.louis_information(model, responses, quadrature.nodes, mass, layouts)
    floating = louis.louis_information(
        model, responses.astype(np.float64), quadrature.nodes, mass, layouts
    )
    np.testing.assert_array_equal(floating.information, integer.information)


def test_fisher_is_marginal_expected_information_not_theta_information():
    data = mirt.simdata(model="2PL", n_persons=2000, n_items=8, seed=3)
    result = mirt.fit_mirt(data, model="2PL", tol=1e-7)
    model = result.model
    quadrature = GaussHermiteQuadrature(21)
    posterior = _posterior_from_model(model, data, quadrature)

    fisher = compute_se(model, data, quadrature, posterior, method="fisher")
    oakes = compute_se(model, data, quadrature, posterior, method="oakes")

    # The former implementation assigned one theta-information value to every
    # parameter of an item.
    assert not np.allclose(fisher["discrimination"], fisher["difficulty"])
    for name in ("discrimination", "difficulty"):
        np.testing.assert_allclose(fisher[name], oakes[name], rtol=0.06)
        np.testing.assert_allclose(
            fisher[name], result.standard_errors[name], rtol=0.06
        )


def test_expected_information_matches_finite_difference_pattern_scores():
    rng = np.random.default_rng(41)
    model = _binary_model(ThreeParameterLogistic, rng)
    quadrature = GaussHermiteQuadrature(15)
    mass = quadrature.weights / quadrature.weights.sum()
    exact, _ = compute_expected_information(model, quadrature, mass, n_persons=300)

    patterns = louis.enumerated_patterns(model)
    scores, _ = _finite_difference_scores(model, patterns, quadrature, mass, 1e-5)
    probability = np.exp(_marginal_log_likelihoods(model, patterns, quadrature, mass))
    assert probability.sum() == pytest.approx(1.0)
    reference = 300.0 * (scores * probability[:, None]).T @ scores
    np.testing.assert_allclose(exact, reference, rtol=1e-6, atol=1e-8)


def test_fisher_rejects_unenumerable_pattern_spaces():
    model = TwoParameterLogistic(17)
    responses = np.zeros((4, 17), dtype=int)
    quadrature = GaussHermiteQuadrature(5)
    posterior = _posterior_from_model(model, responses, quadrature)
    with pytest.raises(MirtValidationError, match="method='oakes'"):
        compute_se(model, responses, quadrature, posterior, method="fisher")


def test_coordinates_on_bounds_are_held_fixed():
    rng = np.random.default_rng(3)
    model = _binary_model(TwoParameterLogistic, rng)
    discrimination = model.parameters["discrimination"]
    discrimination[1] = _parameter_bounds(model, "discrimination")[1]
    model.set_parameters(discrimination=discrimination)
    responses = _responses(model, rng, 2000)
    quadrature = GaussHermiteQuadrature(15)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)

    estimate = estimate_covariance(
        model,
        responses,
        quadrature,
        mass,
        "oakes",
        bounds=lambda name: _parameter_bounds(model, name),
    )
    information = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts
    ).information
    active = np.arange(10) != 1
    expected = np.linalg.inv(information[np.ix_(active, active)])
    assert np.all(np.linalg.eigvalsh(expected) > 0.0)

    assert np.isnan(estimate.standard_errors["discrimination"][1])
    assert np.all(np.isnan(estimate.covariance[1]))
    assert np.all(np.isnan(estimate.covariance[:, 1]))
    np.testing.assert_allclose(
        estimate.covariance[np.ix_(active, active)], expected, rtol=1e-10
    )


@pytest.mark.parametrize("use_rust", [False, True])
def test_unobserved_items_have_unknown_rather_than_zero_errors(use_rust):
    data = mirt.simdata(model="2PL", n_persons=500, n_items=5, seed=1)
    data[:, 2] = -1
    result = mirt.fit_mirt(data, use_rust=use_rust)

    for name in ("discrimination", "difficulty"):
        assert np.isnan(result.standard_errors[name][2])
        assert np.all(np.isfinite(np.delete(result.standard_errors[name], 2)))
    assert np.all(np.isnan(result.vcov[[2, 7]]))


def _fit_data(model, rng, n_persons=600):
    return _responses(model, rng, n_persons, missing=0.05)


@pytest.mark.parametrize("name", ["2PL", "GRM", "GPCM", "PCM"])
@pytest.mark.parametrize("use_rust", [False, True])
def test_em_default_reports_observed_information_and_covariance(name, use_rust):
    rng = np.random.default_rng(17)
    model, _ = _model_and_data(name, rng)
    responses = _fit_data(model, rng)
    result = mirt.fit_mirt(
        responses,
        model=name,
        n_categories=CATEGORIES if model.is_polytomous else None,
        use_rust=use_rust,
        tol=1e-7,
    )
    quadrature = GaussHermiteQuadrature(21)
    posterior = _posterior_from_model(result.model, responses, quadrature)
    expected = compute_oakes_se(result.model, responses, posterior, quadrature)

    assert result.se_method == "oakes"
    assert result.vcov is not None
    assert result.vcov.shape == (result.model.n_parameters,) * 2
    assert len(result.vcov_labels) == result.model.n_parameters
    for parameter, values in expected.items():
        np.testing.assert_allclose(result.standard_errors[parameter], values, rtol=1e-6)
    np.testing.assert_allclose(
        np.sqrt(np.diag(result.vcov)),
        np.concatenate(
            [
                result.standard_errors[parameter][mask]
                for parameter, mask in result.model.free_parameter_masks.items()
            ]
        ),
        rtol=1e-12,
    )


def test_native_and_python_two_parameter_fits_report_the_same_errors():
    data = mirt.simdata(model="2PL", n_persons=800, n_items=6, seed=11)
    native = mirt.fit_mirt(data, model="2PL", tol=1e-8, use_rust=True)
    python = mirt.fit_mirt(data, model="2PL", tol=1e-8, use_rust=False)
    for name in ("discrimination", "difficulty"):
        np.testing.assert_allclose(
            native.standard_errors[name], python.standard_errors[name], rtol=2e-3
        )
    np.testing.assert_allclose(native.vcov, python.vcov, rtol=5e-3, atol=1e-6)
    assert native.vcov_labels == python.vcov_labels


def test_em_uses_the_fitted_latent_prior_mass():
    data = mirt.simdata(model="2PL", n_persons=700, n_items=6, seed=23)
    density = GaussianDensity(mean=np.zeros(1), cov=np.eye(1), estimate_mean=True)
    estimator = EMEstimator(latent_density=density, tol=1e-7, use_rust=False)
    result = estimator.fit(TwoParameterLogistic(6), data)
    quadrature = estimator._quadrature
    mass = np.exp(density.log_quadrature_mass(quadrature.nodes, quadrature.weights))
    posterior = _posterior_from_model(
        result.model, data, quadrature, np.log(mass / mass.sum())
    )
    expected = compute_oakes_se(
        result.model, data, posterior, quadrature, prior_mass=mass
    )
    for name, values in expected.items():
        np.testing.assert_allclose(result.standard_errors[name], values, rtol=1e-9)


def test_complete_data_and_unsupported_models_keep_labelled_complete_data():
    rng = np.random.default_rng(19)
    model, _ = _model_and_data("NRM", rng)
    responses = _fit_data(model, rng, 300)
    nominal = mirt.fit_mirt(
        responses, model="NRM", n_categories=CATEGORIES, max_iter=60
    )
    assert nominal.se_method == "complete_data"
    assert nominal.vcov is None
    explicit = mirt.fit_mirt(
        responses,
        model="NRM",
        n_categories=CATEGORIES,
        max_iter=60,
        se_method="oakes",
    )
    assert explicit.se_method == "oakes"
    assert explicit.vcov.shape == (explicit.model.n_parameters,) * 2

    data = mirt.simdata(model="2PL", n_persons=400, n_items=5, seed=2)
    complete = mirt.fit_mirt(data, se_method="complete_data")
    observed = mirt.fit_mirt(data)
    assert complete.se_method == "complete_data"
    assert complete.vcov is None
    # Complete-data curvature omits the missing information.
    assert np.all(
        complete.standard_errors["difficulty"] < observed.standard_errors["difficulty"]
    )


@pytest.mark.parametrize("method", ["crossprod", "sandwich"])
def test_em_score_based_methods(method):
    data = mirt.simdata(model="GRM", n_persons=500, n_items=5, n_categories=4, seed=8)
    result = mirt.fit_mirt(data, model="GRM", se_method=method, tol=1e-7)
    quadrature = GaussHermiteQuadrature(21)
    posterior = _posterior_from_model(result.model, data, quadrature)
    if method == "crossprod":
        expected = compute_crossprod_se(result.model, data, posterior, quadrature)
    else:
        expected = compute_sandwich_se(result.model, data, posterior, quadrature)
    assert result.se_method == method
    for name, values in expected.items():
        np.testing.assert_allclose(result.standard_errors[name], values, rtol=1e-6)


def test_se_method_validation():
    with pytest.raises(MirtValidationError, match="se_method"):
        EMEstimator(se_method="louis")
    with pytest.raises(MirtValidationError, match="se_method"):
        mirt.fit_mirt(np.zeros((4, 2), dtype=int), se_method="numerical")
    with pytest.raises(MirtValidationError, match="only to EM"):
        mirt.fit_mirt(np.zeros((4, 2), dtype=int), estimation="MHRM", se_method="oakes")


def test_default_standard_errors_are_calibrated():
    """Seeded Monte Carlo check of the default 2PL standard errors.

    The former complete-data default gave an empirical-SD-to-SE ratio of 1.39
    and 82% interval coverage on this design.
    """
    a = np.array([0.8, 1.0, 1.3, 1.6, 1.1])
    b = np.array([-1.0, -0.4, 0.0, 0.5, 1.1])
    truth = np.concatenate([a, b])
    rng = np.random.default_rng(20261005)
    estimates, errors = [], []
    for _ in range(60):
        theta = rng.standard_normal(500)
        probability = 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b)))
        data = (rng.random(probability.shape) < probability).astype(int)
        result = mirt.fit_mirt(data, model="2PL", tol=1e-6)
        estimates.append(
            np.concatenate([result.model.discrimination, result.model.difficulty])
        )
        errors.append(
            np.concatenate(
                [
                    result.standard_errors["discrimination"],
                    result.standard_errors["difficulty"],
                ]
            )
        )
    estimates, errors = np.asarray(estimates), np.asarray(errors)
    ratio = estimates.std(axis=0, ddof=1) / errors.mean(axis=0)
    coverage = np.mean(np.abs(estimates - truth) <= 1.959964 * errors)
    assert 0.85 <= np.median(ratio) <= 1.15
    assert 0.91 <= coverage <= 0.99


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["2PL", "GRM"])
def test_observed_information_coverage_monte_carlo(model_name):
    rng = np.random.default_rng(404)
    n_items = 10 if model_name == "2PL" else 6
    generator = (
        TwoParameterLogistic(n_items)
        if model_name == "2PL"
        else GradedResponseModel(n_items, n_categories=4)
    )
    values = {"discrimination": rng.uniform(0.8, 1.8, n_items)}
    if model_name == "2PL":
        values["difficulty"] = rng.normal(0.0, 0.8, n_items)
    else:
        values["thresholds"] = np.array([-1.5, 0.0, 1.5]) + rng.uniform(
            -0.4, 0.4, (n_items, 3)
        )
    generator.set_parameters(**values)
    truth = np.concatenate([values[name].ravel() for name in values])
    hits, ratios = [], []
    estimates, errors = [], []
    for _ in range(80):
        data = _responses(generator, rng, 500, missing=0.0)
        result = mirt.fit_mirt(data, model=model_name, tol=1e-6)
        estimate = np.concatenate(
            [result.model.parameters[name].ravel() for name in values]
        )
        error = np.concatenate(
            [result.standard_errors[name].ravel() for name in values]
        )
        estimates.append(estimate)
        errors.append(error)
        hits.append(np.abs(estimate - truth) <= 1.959964 * error)
    estimates, errors = np.asarray(estimates), np.asarray(errors)
    ratios = estimates.std(axis=0, ddof=1) / errors.mean(axis=0)
    assert 0.92 <= np.mean(hits) <= 0.98
    assert 0.85 <= np.median(ratios) <= 1.15


@pytest.mark.performance
def test_exact_observed_information_outpaces_marginal_differences():
    import time

    data = mirt.simdata(model="2PL", n_persons=2000, n_items=15, seed=3)
    result = mirt.fit_mirt(data, model="2PL", compute_standard_errors=False)
    quadrature = GaussHermiteQuadrature(21)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(result.model)

    start = time.perf_counter()
    louis.louis_information(result.model, data, quadrature.nodes, mass, layouts)
    exact = time.perf_counter() - start
    start = time.perf_counter()
    _finite_difference_information(result.model, data, quadrature, mass, 1e-5)
    differences = time.perf_counter() - start
    assert exact * 10 < differences
