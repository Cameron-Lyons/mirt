"""A fit's estimated latent population: ``latent_mean`` and ``latent_covariance``.

``EMEstimator`` reports the final mean and covariance of a Gaussian latent
density on its ``FitResult``, so fixed-item calibration and a directly
estimated ``FactorCovarianceDensity`` carry their population like a
``fit_mirt(spec=...)`` fit. Consumers given the fit use both moments.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import mirt
from mirt import TwoParameterLogistic, simdata
from mirt.diagnostics.comparison import vuong_test
from mirt.diagnostics.itemfit import compute_itemfit
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import FactorCovarianceDensity
from mirt.exceptions import MirtValidationError
from mirt.results._common import resolve_latent_prior
from mirt.results.fit_result import FitResult
from mirt.utils import imputation
from mirt.utils.calibration import fixed_item_calibration
from mirt.utils.reliability import marginal_rxx


@pytest.fixture(scope="module")
def calibration():
    """Fixed-item calibration of a population N(0.8, 1.3^2)."""
    theta = np.random.default_rng(1).normal(0.8, 1.3, 800)
    a = np.linspace(0.8, 1.6, 8)
    b = np.linspace(-1.5, 1.5, 8)
    data = np.asarray(simdata(theta=theta, discrimination=a, difficulty=b, seed=1))
    anchors = TwoParameterLogistic(5).set_parameters(
        discrimination=a[:5], difficulty=b[:5]
    )
    result = fixed_item_calibration(
        data, TwoParameterLogistic(8), list(range(5)), anchors
    )
    return result, data


def test_fixed_item_calibration_records_its_population(calibration):
    result, data = calibration
    fit = result.fit_result

    np.testing.assert_array_equal(fit.latent_mean, result.latent_mean)
    np.testing.assert_array_equal(fit.latent_covariance, result.latent_cov)
    assert fit.latent_mean[0] == pytest.approx(0.8, abs=0.15)
    np.testing.assert_allclose(
        mirt.fscores(fit, data).theta,
        mirt.fscores(
            result.model,
            data,
            prior_mean=result.latent_mean,
            prior_cov=result.latent_cov,
        ).theta,
    )


def test_a_population_held_at_its_default_is_not_recorded(calibration):
    _, data = calibration
    anchors = {"discrimination": np.ones(5), "difficulty": np.linspace(-1, 1, 5)}

    mean_only = fixed_item_calibration(
        data, TwoParameterLogistic(8), list(range(5)), anchors, estimate_cov=False
    )

    assert mean_only.fit_result.latent_mean is not None
    assert mean_only.fit_result.latent_covariance is None


def test_direct_factor_covariance_fits_score_like_spec_fits():
    rng = np.random.default_rng(11)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=500)
    slopes = np.zeros((8, 2))
    slopes[:4, 0] = np.linspace(1.0, 2.0, 4)
    slopes[4:, 1] = np.linspace(1.2, 1.8, 4)
    logits = theta @ slopes.T + np.linspace(-1.0, 1.0, 8)
    data = (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    spec = mirt.mirt_model("F1 = 1-4\nF2 = 5-8\nCOV = F1*F2")

    via_spec = mirt.fit_mirt(data, spec=spec, n_quadpts=11, tol=1e-6)
    model = mirt.MultidimensionalModel(
        8,
        2,
        model_type="confirmatory",
        loading_pattern=spec.loading_pattern(8).astype(float),
    )
    density = FactorCovarianceDensity(2)
    direct = EMEstimator(n_quadpts=11, latent_density=density, tol=1e-6).fit(
        model, data
    )

    np.testing.assert_array_equal(direct.latent_covariance, density.cov)
    np.testing.assert_allclose(
        direct.latent_covariance, via_spec.latent_covariance, atol=1e-8
    )
    assert direct.latent_mean is None
    np.testing.assert_allclose(
        mirt.fscores(direct, data).theta, mirt.fscores(via_spec, data).theta, atol=1e-6
    )


def test_latent_mean_is_validated_serialized_and_summarized(calibration):
    result, _ = calibration
    fit = result.fit_result

    restored = FitResult.from_dict(fit.to_dict())

    np.testing.assert_array_equal(restored.latent_mean, fit.latent_mean)
    np.testing.assert_array_equal(restored.latent_covariance, fit.latent_covariance)
    assert "latent_mean" not in fit.to_dict(include_parameters=False)
    assert "Latent mean:" in fit.summary()
    for bad, message in (([0.0, 1.0], "shape"), ([np.inf], "finite"), ("a", "numeric")):
        with pytest.raises(MirtValidationError, match=message):
            replace(fit, latent_mean=bad)


def test_consumers_default_to_the_latent_mean(calibration, monkeypatch):
    result, data = calibration
    fit = result.fit_result

    _, mean, cov = resolve_latent_prior(fit)
    np.testing.assert_array_equal(mean, fit.latent_mean)
    np.testing.assert_array_equal(cov, fit.latent_covariance)
    _, explicit, _ = resolve_latent_prior(fit, prior_mean=[0.0])
    np.testing.assert_array_equal(explicit, [0.0])

    sd = float(np.sqrt(fit.latent_covariance[0, 0]))
    expected = marginal_rxx(
        fit.model,
        density=lambda theta: np.exp(-0.5 * ((theta - fit.latent_mean[0]) / sd) ** 2),
    )
    assert marginal_rxx(fit) == pytest.approx(expected, rel=1e-10)

    shifted = replace(fit, latent_mean=np.array([2.5]))
    assert mirt.simdata(shifted, n_persons=2000, seed=3).mean() > (
        mirt.simdata(replace(fit, latent_mean=None), n_persons=2000, seed=3).mean()
        + 0.1
    )
    # The same item model under a different population fits differently.
    comparison = vuong_test(fit, replace(fit, latent_mean=None), data)
    assert comparison["mean_log_likelihood_difference"] > 0.0


def test_item_fit_and_imputation_use_the_latent_mean(calibration, monkeypatch):
    result, data = calibration
    fit = result.fit_result
    seen = {}

    def recording_itemfit(model, responses, statistics, **kwargs):
        seen["itemfit"] = kwargs["prior_mean"]
        return compute_itemfit(model, responses, statistics, **kwargs)

    monkeypatch.setattr("mirt.diagnostics.itemfit.compute_itemfit", recording_itemfit)
    mirt.itemfit(fit, data, ["infit"])
    np.testing.assert_array_equal(seen["itemfit"], fit.latent_mean)

    def recording_draws(model, observed, n_imputations, n_quadpts, rng, **priors):
        seen["imputation"] = priors["prior_mean"]
        return np.zeros((len(observed), 1, n_imputations))

    monkeypatch.setattr(imputation, "_posterior_ability_draws", recording_draws)
    missing = data[:20].copy()
    missing[0, 0] = -1
    mirt.impute_responses(missing, method="multiple", model=fit, n_imputations=2)
    np.testing.assert_array_equal(seen["imputation"], fit.latent_mean)
