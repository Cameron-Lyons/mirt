"""Consumers of a fit's latent population (``FitResult.latent_covariance``).

A confirmatory fit with correlated factors stores the estimated factor
covariance on its ``FitResult``. Every consumer that rebuilds a posterior or
integrates over the population must use that covariance when given the fit,
exactly as if it had been passed explicitly, and must keep its standard-normal
behaviour for a bare model or an uncorrelated fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
from mirt import fit_mirt, fscores, simdata
from mirt.diagnostics import ld, modelfit, residuals
from mirt.diagnostics.comparison import vuong_test
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2
from mirt.results._common import resolve_latent_prior
from mirt.scoring import eapsum, eapsum_table, sum_score_to_theta
from mirt.utils import imputation
from mirt.utils.plausible import generate_plausible_values
from mirt.utils.reliability import conditional_rxx, marginal_rxx

_TWO_FACTORS = "F1 = 1-5\nF2 = 6-10"


@pytest.fixture(scope="module")
def correlated_data() -> np.ndarray:
    rng = np.random.default_rng(11)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=800)
    slopes = np.zeros((10, 2))
    slopes[:5, 0] = np.linspace(1.0, 2.0, 5)
    slopes[5:, 1] = np.linspace(1.2, 1.8, 5)
    intercepts = np.linspace(-1.2, 1.2, 10)
    probabilities = 1.0 / (1.0 + np.exp(-(theta @ slopes.T + intercepts)))
    return (rng.random(probabilities.shape) < probabilities).astype(int)


@pytest.fixture(scope="module")
def correlated(correlated_data):
    """A two-factor CFA with an estimated factor correlation near 0.6."""
    result = fit_mirt(correlated_data, spec=f"{_TWO_FACTORS}\nCOV = F1*F2")
    assert 0.45 < result.factor_correlation[0, 1] < 0.75
    return correlated_data, result


@pytest.fixture(scope="module")
def orthogonal(correlated_data):
    """The same structure without COV: standard-normal, uncorrelated factors."""
    result = fit_mirt(correlated_data, spec=_TWO_FACTORS)
    assert result.latent_covariance is None
    return correlated_data, result


@pytest.fixture(scope="module")
def rasch():
    """A unidimensional Rasch fit with an estimated latent variance."""
    theta = np.random.default_rng(4).normal(0.0, 1.6, 800)
    data = simdata(
        theta=theta,
        discrimination=np.ones(10),
        difficulty=np.linspace(-2.0, 2.0, 10),
        seed=4,
    )
    result = fit_mirt(data, "1PL", spec="F = 1-10\nCOV = F*F")
    assert result.latent_covariance[0, 0] > 1.8
    return data, result


def _arrays(value) -> list[np.ndarray]:
    """Numeric arrays of a diagnostic's output, in a stable order."""
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, tuple):
        return [np.asarray(item, dtype=float) for item in value]
    if isinstance(value, dict):
        return [
            np.asarray(value[key], dtype=float)
            for key in sorted(value)
            if not isinstance(value[key], list)
        ] + [
            np.asarray([sorted(map(str, value[key]))], dtype=object)
            for key in sorted(value)
            if isinstance(value[key], list)
        ]
    if isinstance(value, ld.LDResult):
        return [value.q3_matrix, value.ld_chi2_matrix, value.g2_matrix]
    if isinstance(value, residuals.ResidualAnalysisResult):
        return [value.standardized_residuals, np.asarray(value.theta_estimates)]
    raise TypeError(type(value))


def _same(first, second) -> bool:
    left, right = _arrays(first), _arrays(second)
    return len(left) == len(right) and all(
        a.shape == b.shape
        and (
            np.array_equal(a, b)
            if a.dtype == object
            else np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
        )
        for a, b in zip(left, right, strict=True)
    )


class TestResolveLatentPrior:
    def test_fit_result_supplies_its_covariance(self, correlated):
        _, result = correlated
        model, mean, cov = resolve_latent_prior(result)
        assert model is result.model
        assert mean is None
        assert_array_equal(cov, result.latent_covariance)

    def test_explicit_arguments_take_precedence(self, correlated):
        _, result = correlated
        _, mean, cov = resolve_latent_prior(result, [0.5, 0.0], np.eye(2))
        assert_array_equal(mean, [0.5, 0.0])
        assert_array_equal(cov, np.eye(2))

    def test_bare_model_and_uncorrelated_fit_keep_the_standard_normal(
        self, correlated, orthogonal
    ):
        assert resolve_latent_prior(correlated[1].model)[1:] == (None, None)
        assert resolve_latent_prior(orthogonal[1])[1:] == (None, None)


class TestItemFit:
    STATISTICS = ["S_X2", "infit", "outfit", "z_infit", "z_outfit"]

    @pytest.mark.filterwarnings("ignore:z_infit and z_outfit with EAP")
    def test_fit_result_equals_explicit_covariance(self, correlated):
        data, result = correlated
        fitted = compute_itemfit(result, data, self.STATISTICS)
        explicit = compute_itemfit(
            result.model, data, self.STATISTICS, prior_cov=result.latent_covariance
        )
        identity = compute_itemfit(result.model, data, self.STATISTICS)

        assert _same(fitted, explicit)
        assert not np.allclose(fitted["S_X2"], identity["S_X2"])
        assert not np.allclose(fitted["infit"], identity["infit"])

    def test_top_level_itemfit_uses_the_fit_population(self, correlated):
        data, result = correlated
        frame = mirt.itemfit(result, data, ["S_X2", "infit"])
        explicit = compute_itemfit(
            result.model, data, ["S_X2", "infit"], prior_cov=result.latent_covariance
        )
        assert_allclose(np.asarray(frame["S_X2"]), explicit["S_X2"])
        assert_allclose(np.asarray(frame["infit"]), explicit["infit"])
        overridden = mirt.itemfit(result, data, ["S_X2"], prior_cov=np.eye(2))
        identity = compute_itemfit(result.model, data, ["S_X2"])
        assert_allclose(np.asarray(overridden["S_X2"]), identity["S_X2"])

    def test_s_x2_wrapper_and_explicit_grid(self, correlated):
        data, result = correlated
        fitted = compute_s_x2(result, data)
        explicit = compute_s_x2(result.model, data, prior_cov=result.latent_covariance)
        assert _same(fitted, explicit)
        # An explicit grid replaces the population for S-X2.
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        grid = GaussHermiteQuadrature(41, 2)
        on_grid = compute_s_x2(
            result,
            data,
            quadrature_points=grid.nodes,
            quadrature_weights=grid.weights,
        )
        assert _same(on_grid, compute_s_x2(result.model, data))

    def test_binned_statistics_use_the_fitted_variance(self, rasch):
        data, result = rasch
        statistics = ["X2", "G2", "PV_Q1"]
        fitted = compute_itemfit(result, data, statistics, seed=3)
        explicit = compute_itemfit(
            result.model,
            data,
            statistics,
            seed=3,
            prior_cov=result.latent_covariance,
        )
        identity = compute_itemfit(result.model, data, statistics, seed=3)
        assert _same(fitted, explicit)
        assert not np.allclose(fitted["PV_Q1"], identity["PV_Q1"])

    def test_invalid_population_is_rejected(self, correlated):
        data, result = correlated
        with pytest.raises(ValueError, match="prior_cov"):
            compute_itemfit(result.model, data, ["S_X2"], prior_cov=np.eye(3))


class TestLimitedInformationFit:
    def test_fit_indices_integrate_over_the_fitted_covariance(self, correlated):
        data, result = correlated
        fitted = modelfit.compute_fit_indices(result, data)
        explicit = modelfit.compute_fit_indices(
            result.model, data, prior_cov=result.latent_covariance
        )
        identity = modelfit.compute_fit_indices(result.model, data)

        assert fitted["SRMSR"] == pytest.approx(explicit["SRMSR"], rel=1e-12)
        assert fitted["SRMSR"] < 0.5 * identity["SRMSR"]
        assert fitted["M2"] < 0.5 * identity["M2"]
        assert fitted["RMSEA"] < identity["RMSEA"]

    def test_estimated_correlation_is_a_nuisance_parameter(self, correlated):
        data, result = correlated
        fitted = modelfit.compute_m2(result, data)
        known = modelfit.compute_m2(
            result.model, data, prior_cov=result.latent_covariance
        )
        assert fitted["df"] == known["df"] - 1
        assert fitted["M2"] <= known["M2"] + 1e-9
        assert fitted == modelfit.compute_m2(result, data, n_quadpts=21)

    def test_estimated_variance_is_a_nuisance_parameter(self, rasch):
        data, result = rasch
        fitted = modelfit.compute_m2(result, data)
        known = modelfit.compute_m2(
            result.model, data, prior_cov=result.latent_covariance
        )
        identity = modelfit.compute_m2(result.model, data)
        assert fitted["df"] == known["df"] - 1
        assert fitted["M2"] < identity["M2"]

    def test_only_entries_moved_from_the_identity_count_as_estimated(self, correlated):
        from dataclasses import replace

        _, result = correlated
        variance_only = replace(
            result, latent_covariance=np.array([[1.4, 0.0], [0.0, 1.0]])
        )
        _, latent = modelfit._resolve_latent_population(variance_only, None, None)
        assert latent.estimated == ((0, 0),)
        _, latent = modelfit._resolve_latent_population(result, None, None)
        assert latent.estimated == ((0, 1),)
        # A population passed explicitly is known, even if it equals the fit's.
        _, latent = modelfit._resolve_latent_population(
            result, None, result.latent_covariance
        )
        assert latent.estimated == ()

    def test_latent_jacobian_differentiates_the_integrated_moments(self, correlated):
        _, result = correlated
        model = result.model
        correlation = result.latent_covariance[0, 1]
        latent = modelfit._LatentNormal(
            np.zeros(2), result.latent_covariance, ((0, 1),)
        )
        jacobian = modelfit._latent_moment_jacobian(model, 15, latent)

        def moments(value: float) -> np.ndarray:
            population = modelfit._LatentNormal(
                np.zeros(2), np.array([[1.0, value], [value, 1.0]])
            )
            integrated, _ = modelfit._integrate_model_moments(
                model, 15, latent=population
            )
            return modelfit._flatten_score_moments(
                integrated.univariate, integrated.bivariate
            )

        step = 1e-5
        expected = (moments(correlation + step) - moments(correlation - step)) / (
            2 * step
        )
        assert jacobian.shape == (expected.size, 1)
        assert_allclose(jacobian[:, 0], expected, rtol=1e-5, atol=1e-9)
        # The correlation moves cross-factor products, not item means (up to
        # quadrature error, since the correlated nodes differ by factor).
        means, products = np.abs(jacobian[: model.n_items, 0]), jacobian[10:, 0]
        assert means.max() < 1e-3 * np.abs(products).max()


@pytest.mark.parametrize(
    ("function", "score_options"),
    [
        (residuals.compute_residuals, {}),
        (residuals.analyze_residuals, {}),
        (residuals.compute_outfit_infit, {}),
        (residuals.identify_misfitting_patterns, {}),
        (ld.compute_q3, {}),
        # These score on their own ``n_quadpts`` grid (default 21).
        (ld.compute_ld_chi2, {"n_quadpts": 21}),
        (ld.compute_ld_statistics, {"n_quadpts": 21}),
    ],
)
def test_eap_based_diagnostics_score_under_the_fitted_covariance(
    correlated, function, score_options
):
    data, result = correlated
    theta = fscores(
        result.model, data, prior_cov=result.latent_covariance, **score_options
    ).theta
    fitted = function(result, data)
    explicit = function(result.model, data, theta)
    identity = function(result.model, data)
    assert _same(fitted, explicit)
    assert not _same(fitted, identity)


def test_personfit_scores_under_the_fitted_covariance(correlated):
    data, result = correlated
    fitted = mirt.personfit(result, data)
    explicit = mirt.personfit(
        result,
        data,
        theta=fscores(result.model, data, prior_cov=result.latent_covariance).theta,
    )
    identity = mirt.personfit(result, data, theta=fscores(result.model, data).theta)
    assert_allclose(np.asarray(fitted["Zh"]), np.asarray(explicit["Zh"]))
    assert not np.allclose(np.asarray(fitted["Zh"]), np.asarray(identity["Zh"]))


def test_vuong_marginalizes_each_fit_over_its_population(correlated, orthogonal):
    data, result = correlated
    _, uncorrelated = orthogonal
    comparison = vuong_test(result, uncorrelated, data)
    expected = (result.log_likelihood - uncorrelated.log_likelihood) / len(data)
    assert comparison["mean_log_likelihood_difference"] == pytest.approx(
        expected, abs=1e-4
    )
    assert comparison["z"] > 0


def test_plausible_values_draw_from_the_fitted_population(correlated):
    data, result = correlated
    fitted = generate_plausible_values(result, data, n_plausible=3, seed=2)
    explicit = generate_plausible_values(
        result.model, data, n_plausible=3, seed=2, prior_cov=result.latent_covariance
    )
    identity = generate_plausible_values(result.model, data, n_plausible=3, seed=2)
    assert_array_equal(fitted, explicit)
    correlation = np.corrcoef(fitted[:, 0, 0], fitted[:, 1, 0])[0, 1]
    identity_correlation = np.corrcoef(identity[:, 0, 0], identity[:, 1, 0])[0, 1]
    assert correlation > identity_correlation + 0.1


def test_simdata_draws_abilities_from_the_fitted_covariance(correlated, monkeypatch):
    import mirt._categorical

    _, result = correlated
    drawn: list[np.ndarray] = []
    original = mirt._categorical.draw_item_responses

    def spy(model, theta, rng, *args, **kwargs):
        drawn.append(np.array(theta))
        return original(model, theta, rng, *args, **kwargs)

    monkeypatch.setattr(mirt._categorical, "draw_item_responses", spy)
    simdata(result, n_persons=50, seed=8)
    simdata(result.model, n_persons=50, seed=8)
    factor = np.linalg.cholesky(result.latent_covariance)
    assert_allclose(drawn[0], drawn[1] @ factor.T)


class TestImputation:
    @pytest.fixture
    def incomplete(self, correlated):
        data, result = correlated
        values = data.copy()
        values[np.random.default_rng(0).random(values.shape) < 0.1] = -1
        return values, result

    def test_em_imputation_scores_under_the_fitted_covariance(
        self, incomplete, monkeypatch
    ):
        values, result = incomplete
        captured: list[np.ndarray] = []
        original = imputation._draw_model_responses

        def spy(imputed, missing, model, theta, rng):
            captured.append(np.array(theta))
            return original(imputed, missing, model, theta, rng)

        monkeypatch.setattr(imputation, "_draw_model_responses", spy)
        mirt.impute_responses(values, method="EM", model=result, seed=1)
        expected = fscores(
            result.model, values, prior_cov=result.latent_covariance
        ).theta
        assert_allclose(captured[0], expected)

    def test_multiple_imputation_draws_under_the_fitted_covariance(
        self, incomplete, monkeypatch
    ):
        import mirt.utils.plausible as plausible

        values, result = incomplete
        priors = []
        original = plausible._generate_pv_posterior

        def spy(*args, prior=None, **kwargs):
            priors.append(prior)
            return original(*args, prior=prior, **kwargs)

        monkeypatch.setattr(plausible, "_generate_pv_posterior", spy)
        mirt.impute_responses(
            values, method="multiple", model=result, n_imputations=2, seed=1
        )
        mirt.impute_responses(
            values, method="multiple", model=result.model, n_imputations=2, seed=1
        )
        assert_allclose(
            priors[0].cholesky, np.linalg.cholesky(result.latent_covariance)
        )
        assert priors[1] is None


class TestUnidimensionalSummaries:
    def test_eapsum_uses_the_fitted_variance(self, rasch):
        data, result = rasch
        prior = result.latent_covariance
        fitted = eapsum(result, data)
        explicit = eapsum(result.model, data, prior_cov=prior)
        identity = eapsum(result.model, data)
        assert_allclose(fitted.theta, explicit.theta)
        assert not np.allclose(fitted.theta, identity.theta)
        assert_allclose(fitted.theta, fscores(result, data, method="EAPsum").theta)

        scores = np.arange(11)
        assert_allclose(
            sum_score_to_theta(result, scores)[0],
            sum_score_to_theta(result.model, scores, prior_cov=prior)[0],
        )
        assert_allclose(
            eapsum_table(result, data).theta,
            eapsum_table(result.model, data, prior_cov=prior).theta,
        )

    def test_reliability_uses_the_fitted_variance(self, rasch):
        _, result = rasch
        variance = float(result.latent_covariance[0, 0])
        theta = np.linspace(-3.0, 3.0, 7)
        assert_allclose(
            conditional_rxx(result, theta),
            conditional_rxx(result.model, theta, latent_variance=variance),
        )
        assert_allclose(
            conditional_rxx(result, theta, latent_variance=1.0),
            conditional_rxx(result.model, theta),
        )

        fitted = marginal_rxx(result)
        explicit = marginal_rxx(
            result.model, density=lambda grid: np.exp(-0.5 * grid**2 / variance)
        )
        assert fitted == pytest.approx(explicit, rel=1e-12)
        assert fitted > marginal_rxx(result.model)


def test_reports_pass_the_fit_population(correlated, monkeypatch):
    from mirt.diagnostics import itemfit
    from mirt.reports import ItemAnalysisReport, ModelFitReport

    data, result = correlated
    received = []

    def spy(function):
        def wrapper(model, *args, **kwargs):
            received.append(model)
            return function(model, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        modelfit, "compute_fit_indices", spy(modelfit.compute_fit_indices)
    )
    monkeypatch.setattr(itemfit, "compute_itemfit", spy(itemfit.compute_itemfit))
    ModelFitReport(result, data, include_plots=False).generate()
    ItemAnalysisReport(result, data, include_plots=False).generate()
    assert received == [result, result]


@pytest.mark.parametrize(
    "consumer",
    [
        lambda fit, data: compute_itemfit(fit, data, ["S_X2", "infit"]),
        lambda fit, data: modelfit.compute_m2(fit, data),
        lambda fit, data: ld.compute_q3(fit, data),
        lambda fit, data: generate_plausible_values(fit, data, seed=0),
    ],
    ids=["itemfit", "m2", "q3", "plausible_values"],
)
def test_uncorrelated_fit_matches_the_bare_model(orthogonal, consumer):
    data, result = orthogonal
    assert _same(consumer(result, data), consumer(result.model, data))
