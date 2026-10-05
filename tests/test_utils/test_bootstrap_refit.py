"""Refits keep the estimator configuration of the fit they resample.

Bootstrap, jackknife, parametric bootstrap, ``bootstrap_lr`` and
``multi_start_fit`` refit copies of a model. A ``FitResult`` records how it
was fitted (``refit_recipe``), so every refit uses the same estimator class,
latent density, item priors, equality constraints and quadrature, and skips
standard errors that no resampling statistic uses.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

import mirt
from mirt import OneParameterLogistic, TwoParameterLogistic, simdata
from mirt._categorical import draw_item_responses
from mirt.backends.rust import _helpers as rust_helpers
from mirt.backends.rust import estimation as rust_estimation
from mirt.estimation._refit import RefitRecipe, em_estimator_for, recipe_for
from mirt.estimation.bifactor_em import BifactorEMEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import FactorCovarianceDensity, GaussianDensity
from mirt.estimation.mixed_format_em import MixedFormatEMEstimator
from mirt.estimation.priors import BetaPrior, LogNormalPrior
from mirt.exceptions import MirtValidationError
from mirt.models.bifactor import BifactorModel
from mirt.models.mixed_format import MixedItemModel
from mirt.results.fit_result import FitResult
from mirt.utils import bootstrap as bootstrap_module
from mirt.utils.calibration import fixed_item_calibration
from mirt.utils.starting import multi_start_fit

_SPEC = "F1 = 1-4\nF2 = 5-8"


@pytest.fixture
def replicate_fits(monkeypatch):
    """Record each refitting estimator and the result of its EM fit."""
    records: list[tuple[object, FitResult]] = []
    original = EMEstimator.fit

    def recording_fit(self, model, responses, *args, **kwargs):
        result = original(self, model, responses, *args, **kwargs)
        records.append((self, result))
        return result

    monkeypatch.setattr(EMEstimator, "fit", recording_fit)
    return records


@pytest.fixture
def standard_error_calls(monkeypatch):
    """Count the standard-error computations of EM fits."""
    calls: list[int] = []
    original = EMEstimator._compute_standard_errors

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EMEstimator, "_compute_standard_errors", counting)
    return calls


@pytest.fixture(scope="module")
def correlated_data() -> np.ndarray:
    rng = np.random.default_rng(11)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=600)
    slopes = np.zeros((8, 2))
    slopes[:4, 0] = np.linspace(1.0, 2.0, 4)
    slopes[4:, 1] = np.linspace(1.2, 1.8, 4)
    logits = theta @ slopes.T + np.linspace(-1.0, 1.0, 8)
    return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)


@pytest.fixture(scope="module")
def correlated_fit(correlated_data):
    """A confirmatory fit with an estimated factor correlation."""
    result = mirt.fit_mirt(
        correlated_data, spec=f"{_SPEC}\nCOV = F1*F2", n_quadpts=11, tol=1e-6
    )
    assert 0.45 < result.latent_covariance[0, 1] < 0.8
    return result


@pytest.fixture(scope="module")
def guessing_prior_fit():
    """A Bayes modal 3PL fit whose guessing estimates sit near 0.2."""
    data = np.asarray(simdata("3PL", n_persons=400, n_items=6, seed=3))
    result = mirt.fit_mirt(data, model="3PL", priors={"guessing": BetaPrior(5, 17)})
    return result, data


@pytest.fixture(scope="module")
def anchored_calibration():
    """Fixed-item calibration of a population N(0.8, 1.3^2)."""
    theta = np.random.default_rng(1).normal(0.8, 1.3, 1000)
    a = np.linspace(0.8, 1.6, 8)
    b = np.linspace(-1.5, 1.5, 8)
    data = np.asarray(simdata(theta=theta, discrimination=a, difficulty=b, seed=1))
    anchors = TwoParameterLogistic(5).set_parameters(
        discrimination=a[:5], difficulty=b[:5]
    )
    calibration = fixed_item_calibration(
        data, TwoParameterLogistic(8), list(range(5)), anchors
    )
    return calibration, data


@pytest.fixture(scope="module")
def bifactor_fit():
    """A bifactor fit with four specific factors (5-dimensional product grid)."""
    rng = np.random.default_rng(1)
    factors = np.repeat(np.arange(4), 3)
    theta = rng.standard_normal((200, 5))
    logits = 1.2 * theta[:, :1] + 0.8 * theta[:, 1 + factors] - 0.3
    data = (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    return mirt.bfactor(data, factors, max_iter=100), data


class TestRefitRecipe:
    def test_em_fits_record_their_estimator_settings(self, guessing_prior_fit):
        result, _ = guessing_prior_fit
        recipe = result.refit_recipe

        assert recipe.estimator is EMEstimator
        assert set(recipe.options["item_priors"]) == {"guessing"}
        assert recipe.options["n_quadpts"] == 21
        assert recipe.latent_density is None
        assert "verbose" not in recipe.options
        assert "compute_standard_errors" not in recipe.options

    def test_worker_threads_are_left_to_each_refit(self, replicate_fits):
        data = np.asarray(simdata("2PL", n_persons=200, n_items=4, seed=7))
        threaded = EMEstimator(n_jobs=2, tol=1e-3).fit(OneParameterLogistic(4), data)

        mirt.bootstrap_se(threaded, data, n_bootstrap=2, seed=1, n_jobs=1)

        # Item threads would multiply with the bootstrap's process workers.
        assert "n_jobs" not in threaded.refit_recipe.options
        assert [estimator.n_jobs for estimator, _ in replicate_fits[1:]] == [1, 1]

    def test_estimated_densities_are_copied_into_every_estimator(self, correlated_fit):
        recipe = correlated_fit.refit_recipe
        first, second = recipe.build(), recipe.build(max_iter=3)

        assert isinstance(recipe.latent_density, FactorCovarianceDensity)
        assert first._latent_density_spec is not second._latent_density_spec
        assert first._latent_density_spec is not recipe.latent_density
        np.testing.assert_array_equal(
            first._latent_density_spec.cov, correlated_fit.latent_covariance
        )
        np.testing.assert_array_equal(
            first._latent_density_spec.free, recipe.latent_density.free
        )
        assert second.max_iter == 3 and first.max_iter == recipe.options["max_iter"]

    def test_native_and_mixed_and_bifactor_fits_record_their_estimator(
        self, bifactor_fit
    ):
        data = np.asarray(simdata("2PL", n_persons=200, n_items=5, seed=2))
        native = mirt.fit_mirt(data, model="2PL", n_quadpts=15)
        mixed = mirt.fit_mirt(
            np.column_stack([data, data[:, :2] + data[:, 2:4]]),
            model=["2PL"] * 5 + ["GRM"] * 2,
            tol=1e-2,
            compute_standard_errors=False,
        )

        assert native.refit_recipe.estimator is EMEstimator
        assert native.refit_recipe.options["n_quadpts"] == 15
        assert mixed.refit_recipe.estimator is MixedFormatEMEstimator
        assert bifactor_fit[0].refit_recipe.estimator is BifactorEMEstimator

    def test_mixed_format_recipe_keeps_the_component_priors(self):
        data = np.asarray(simdata("3PL", n_persons=200, n_items=4, seed=4))
        model = MixedItemModel.from_itemtypes(["3PL", "3PL", "2PL", "2PL"])
        priors = {"3PL.guessing": BetaPrior(5, 17)}
        result = MixedFormatEMEstimator(tol=1e-2, item_priors=priors).fit(model, data)

        assert result.refit_recipe.options["item_priors"] == priors
        assert result.refit_recipe.build().item_priors == priors

    def test_estimators_without_a_known_constructor_record_no_recipe(self):
        class CustomEstimator(EMEstimator):
            def __init__(self) -> None:
                super().__init__()

        assert recipe_for(CustomEstimator()) is None
        assert recipe_for(object()) is None

    def test_overrides_must_be_settings_of_the_estimator(self, bifactor_fit):
        recipe = bifactor_fit[0].refit_recipe

        with pytest.raises(MirtValidationError, match="does not take item_priors"):
            recipe.build(item_priors={"guessing": BetaPrior(5, 17)})
        with pytest.raises(MirtValidationError, match="does not take bogus"):
            em_estimator_for(TwoParameterLogistic(2), bogus=1)

    def test_recipe_is_not_serialized(self, guessing_prior_fit):
        result, _ = guessing_prior_fit
        payload = result.to_dict()

        assert "refit_recipe" not in payload
        assert FitResult.from_dict(payload).refit_recipe is None
        assert "refit_recipe" not in repr(result)


class TestEstimatorDispatch:
    def test_bare_models_use_their_family_estimator(self):
        bifactor = BifactorModel(6, np.repeat([0, 1], 3))
        mixed = MixedItemModel.from_itemtypes(["2PL", "GRM"], n_categories=3)

        assert type(em_estimator_for(bifactor, max_iter=5)) is BifactorEMEstimator
        assert type(em_estimator_for(mixed)) is MixedFormatEMEstimator
        assert type(em_estimator_for(TwoParameterLogistic(2))) is EMEstimator

    def test_bifactor_options_it_does_not_take_fall_back_to_em(self):
        bifactor = BifactorModel(6, np.repeat([0, 1], 3))

        estimator = em_estimator_for(bifactor, use_gpu=False, n_quadpts=7)

        assert type(estimator) is EMEstimator

    def test_results_use_their_recipe_and_explicit_recipes_win(self, correlated_fit):
        estimator = em_estimator_for(correlated_fit, tol=1e-2)
        override = em_estimator_for(
            correlated_fit, recipe=RefitRecipe(EMEstimator, {"n_quadpts": 7})
        )

        assert isinstance(estimator._latent_density_spec, FactorCovarianceDensity)
        assert estimator.tol == 1e-2
        assert override._latent_density_spec is None
        assert override.n_quadpts == 7


class TestBootstrapRefits:
    def test_spec_covariance_replicates_reestimate_the_covariance(
        self, correlated_fit, correlated_data, replicate_fits, standard_error_calls
    ):
        errors = mirt.bootstrap_se(
            correlated_fit, correlated_data, n_bootstrap=3, seed=1
        )

        assert len(replicate_fits) == 3
        densities = [estimator._latent_density for estimator, _ in replicate_fits]
        assert all(
            isinstance(density, FactorCovarianceDensity) for density in densities
        )
        assert len({id(density) for density in densities}) == 3
        for _, result in replicate_fits:
            assert result.latent_covariance[0, 1] == pytest.approx(
                correlated_fit.latent_covariance[0, 1], abs=0.15
            )
            assert result.standard_errors == {}
        # Structural zeros stay fixed, so only the loading pattern varies.
        assert np.all(errors["slopes"][:4, 1] == 0.0)
        assert np.all(errors["slopes"][:4, 0] > 0.0)
        assert standard_error_calls == []

    def test_refitting_the_original_data_reproduces_a_spec_fit(
        self, correlated_fit, correlated_data
    ):
        refit = bootstrap_module._refit_log_likelihood(
            correlated_fit.model,
            correlated_data,
            True,
            {"max_iter": 500, "tol": 1e-6},
            correlated_fit.refit_recipe,
        )

        # Refits without the covariance lost about 10 log-likelihood units.
        assert refit.log_likelihood == pytest.approx(
            correlated_fit.log_likelihood, abs=1e-3
        )
        np.testing.assert_allclose(
            refit.latent_covariance, correlated_fit.latent_covariance, atol=1e-3
        )

    def test_bayes_modal_fits_are_bootstrapped_with_their_priors(
        self, guessing_prior_fit, replicate_fits, standard_error_calls
    ):
        result, data = guessing_prior_fit

        intervals = mirt.bootstrap_ci(result, data, n_bootstrap=10, seed=1)

        assert len(replicate_fits) == 10
        for estimator, replicate in replicate_fits:
            assert set(estimator.item_priors) == {"guessing"}
            assert replicate.log_posterior is not None
        # Maximum likelihood refits put most guessing estimates at zero.
        lower, upper = intervals["guessing"]
        guessing = result.model.parameters["guessing"]
        assert np.all(lower > 0.1)
        assert np.all((lower <= guessing) & (guessing <= upper))
        assert standard_error_calls == []

    def test_constrained_fits_keep_their_equality_constraints(self):
        data = np.asarray(simdata("2PL", n_persons=400, n_items=6, seed=5))
        common = mirt.fit_mirt(
            data, "2PL", constraints=[("discrimination", list(range(6)))]
        )

        errors = mirt.bootstrap_se(common, data, n_bootstrap=3, seed=1)

        assert np.all(errors["discrimination"] > 0.0)
        np.testing.assert_allclose(
            errors["discrimination"], errors["discrimination"][0], rtol=1e-12
        )
        assert not np.allclose(errors["difficulty"], errors["difficulty"][0])

    def test_fixed_item_calibration_intervals_cover_the_estimates(
        self, anchored_calibration
    ):
        calibration, data = anchored_calibration
        new = calibration.new_items

        lower, upper = mirt.bootstrap_ci(
            calibration.fit_result, data, n_bootstrap=10, seed=1
        )["difficulty"]

        estimates = calibration.model.parameters["difficulty"]
        # Refits under N(0, 1) moved every new item outside its interval.
        assert np.all((lower[new] <= estimates[new]) & (estimates[new] <= upper[new]))
        np.testing.assert_array_equal(lower[: new[0]], estimates[: new[0]])

    def test_bifactor_fits_are_refitted_by_dimension_reduction(
        self, bifactor_fit, monkeypatch
    ):
        result, data = bifactor_fit
        estimators = []
        original = BifactorEMEstimator.fit

        def recording_fit(self, model, responses, **kwargs):
            estimators.append(self)
            return original(self, model, responses, **kwargs)

        monkeypatch.setattr(BifactorEMEstimator, "fit", recording_fit)
        with warnings.catch_warnings():
            # The product grid of EMEstimator would warn about 4 million nodes.
            warnings.simplefilter("error")
            errors = mirt.bootstrap_se(result, data, n_bootstrap=2, seed=0)
            mirt.bootstrap_se(result.model, data, n_bootstrap=2, seed=0)

        assert len(estimators) == 4
        assert not any(estimator.compute_standard_errors for estimator in estimators)
        assert set(errors) == set(result.model.parameters)
        assert np.all(np.isfinite(errors["general_loadings"]))

    def test_theta_statistics_score_with_the_estimated_population(
        self, anchored_calibration, monkeypatch
    ):
        calibration, data = anchored_calibration
        scored = []
        original = mirt.scoring.fscores

        def recording_scores(model_or_result, responses, **kwargs):
            scored.append(model_or_result)
            return original(model_or_result, responses, **kwargs)

        monkeypatch.setattr(mirt.scoring, "fscores", recording_scores)
        mirt.bootstrap_ci(
            calibration.fit_result,
            data[:200],
            n_bootstrap=10,
            statistic="theta",
            seed=2,
        )

        assert all(isinstance(value, FitResult) for value in scored)
        assert all(value.latent_mean[0] > 0.4 for value in scored)

    def test_fits_without_a_recipe_warn_that_refits_drop_their_settings(
        self, guessing_prior_fit
    ):
        result, data = guessing_prior_fit
        restored = replace(result, refit_recipe=None)

        with pytest.warns(RuntimeWarning, match="records no refit settings"):
            mirt.bootstrap_se(restored, data[:100], n_bootstrap=2, seed=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mirt.bootstrap_se(
                replace(restored, log_posterior=None), data[:100], n_bootstrap=2
            )
        assert not any("refit settings" in str(item.message) for item in caught)

    def test_native_2pl_bootstrap_only_serves_plain_maximum_likelihood(
        self, monkeypatch
    ):
        data = np.asarray(simdata("2PL", n_persons=100, n_items=3, seed=8))
        plain = mirt.fit_mirt(data, model="2PL", n_quadpts=15)
        prior = mirt.fit_mirt(
            data, model="2PL", priors={"discrimination": LogNormalPrior(0, 0.5)}
        )
        calls = []

        def fake_bootstrap(responses, **kwargs):
            calls.append(kwargs)
            n_bootstrap = kwargs["n_bootstrap"]
            return np.ones((n_bootstrap, 3)), np.zeros((n_bootstrap, 3))

        monkeypatch.setattr(rust_helpers, "rust_enabled", lambda: True)
        monkeypatch.setattr(rust_estimation, "bootstrap_fit_2pl", fake_bootstrap)

        mirt.bootstrap_se(plain, data, n_bootstrap=2, seed=1)
        errors = mirt.bootstrap_se(prior, data, n_bootstrap=2, seed=1)

        assert len(calls) == 1
        assert calls[0]["n_quadpts"] == 15
        assert np.all(errors["discrimination"] > 0.0)

    def test_skipping_replicate_standard_errors_keeps_the_estimates(
        self, guessing_prior_fit, monkeypatch
    ):
        result, data = guessing_prior_fit
        options = {"n_bootstrap": 3, "seed": 5}
        fast = mirt.bootstrap_se(result, data, **options)

        def reference_estimator(model, recipe, max_iter):
            # The replicate estimator before standard errors were skipped.
            return em_estimator_for(
                model, recipe=recipe, max_iter=max_iter, tol=1e-3, verbose=False
            )

        monkeypatch.setattr(
            bootstrap_module, "_replicate_estimator", reference_estimator
        )
        reference = mirt.bootstrap_se(result, data, **options)

        assert fast.keys() == reference.keys()
        for name in fast:
            np.testing.assert_array_equal(fast[name], reference[name])

    def test_process_workers_receive_the_recipe(self, guessing_prior_fit):
        result, data = guessing_prior_fit
        options = {"n_bootstrap": 2, "seed": 3}

        serial = mirt.bootstrap_se(result, data[:150], **options)
        parallel = mirt.bootstrap_se(result, data[:150], n_jobs=2, **options)

        for name in serial:
            np.testing.assert_allclose(parallel[name], serial[name], atol=1e-12)


class TestParametricRefits:
    def test_abilities_are_drawn_from_the_fitted_population(
        self, anchored_calibration, monkeypatch, standard_error_calls
    ):
        calibration, _ = anchored_calibration
        fit = calibration.fit_result
        captured = []
        original = EMEstimator.fit

        def recording_fit(self, model, responses, *args, **kwargs):
            result = original(self, model, responses, *args, **kwargs)
            captured.append((responses.copy(), result))
            return result

        monkeypatch.setattr(EMEstimator, "fit", recording_fit)
        mirt.parametric_bootstrap(fit, n_bootstrap=2, n_persons=300, seed=9)

        rng = np.random.default_rng(9)
        scale = np.linalg.cholesky(fit.latent_covariance)
        assert len(captured) == 2
        for responses, replicate in captured:
            theta = rng.standard_normal((300, 1)) @ scale.T + fit.latent_mean
            np.testing.assert_array_equal(
                responses, draw_item_responses(fit.model, theta, rng)
            )
            # Anchors stay fixed and the population is re-estimated.
            np.testing.assert_array_equal(
                replicate.model.parameters["difficulty"][:5],
                fit.model.parameters["difficulty"][:5],
            )
            assert replicate.latent_mean[0] == pytest.approx(
                fit.latent_mean[0], abs=0.3
            )
        assert standard_error_calls == []


class TestLikelihoodRatioRefits:
    def test_constrained_fits_count_each_group_once(self):
        data = np.asarray(simdata("2PL", n_persons=300, n_items=5, seed=5))
        common = mirt.fit_mirt(
            data,
            "2PL",
            constraints=[("discrimination", list(range(5)))],
            compute_standard_errors=False,
        )
        free = mirt.fit_mirt(data, "2PL", compute_standard_errors=False)

        test = mirt.bootstrap_lr(common, free, data, n_bootstrap=2, seed=1)

        # Model parameter counts are equal, so the test used to be refused.
        assert test.df == 4
        assert test.reduced_log_likelihood == pytest.approx(
            common.log_likelihood, abs=0.05
        )
        # Without its recipe the reduced fit is refitted without constraints,
        # so it counts, and estimates, the full model's parameters.
        with pytest.raises(MirtValidationError, match="more free parameters"):
            mirt.bootstrap_lr(
                replace(common, refit_recipe=None), free, data, n_bootstrap=2
            )

    def test_factor_correlation_is_tested_against_orthogonal_factors(
        self, correlated_fit, correlated_data, monkeypatch
    ):
        orthogonal = mirt.fit_mirt(
            correlated_data, spec=_SPEC, n_quadpts=11, compute_standard_errors=False
        )
        tasks = []
        original = bootstrap_module._fit_lr_task

        def recording_task(task):
            tasks.append(task)
            return original(task)

        monkeypatch.setattr(bootstrap_module, "_fit_lr_task", recording_task)
        test = mirt.bootstrap_lr(
            orthogonal, correlated_fit, correlated_data, n_bootstrap=2, n_quadpts=11
        )

        assert test.df == 1
        assert test.full_log_likelihood == pytest.approx(
            correlated_fit.log_likelihood, abs=0.05
        )
        assert test.statistic > 20.0
        assert tasks[0].full_recipe is correlated_fit.refit_recipe
        assert tasks[0].latent_cholesky is None

    def test_anchor_tests_simulate_the_estimated_population(
        self, anchored_calibration, monkeypatch
    ):
        calibration, data = anchored_calibration
        anchors = calibration.model.parameters
        # Freeing the last anchor tests whether it drifted.
        freed = fixed_item_calibration(
            data,
            TwoParameterLogistic(8),
            list(range(4)),
            {name: values[:4] for name, values in anchors.items()},
            compute_standard_errors=False,
        )
        tasks = []
        original = bootstrap_module._fit_lr_task

        def recording_task(task):
            tasks.append(task)
            return original(task)

        monkeypatch.setattr(bootstrap_module, "_fit_lr_task", recording_task)
        test = mirt.bootstrap_lr(
            calibration.fit_result, freed.fit_result, data, n_bootstrap=2, seed=1
        )

        assert test.df == 2
        assert test.null_statistics.shape == (2,)
        np.testing.assert_allclose(
            tasks[0].latent_mean, calibration.latent_mean, atol=0.02
        )
        np.testing.assert_allclose(
            tasks[0].latent_cholesky ** 2, calibration.latent_cov, rtol=0.05
        )


class TestMultiStartRefits:
    def test_standard_errors_are_computed_once_for_the_best_start(
        self, standard_error_calls
    ):
        data = np.asarray(simdata("2PL", n_persons=200, n_items=4, seed=6))
        result = multi_start_fit(
            TwoParameterLogistic(4), data, n_starts=3, seed=2, use_gpu=False
        )

        assert standard_error_calls == [1]
        assert result.se_method == "oakes"
        assert np.all(np.isfinite(result.standard_errors["discrimination"]))
        assert result.refit_recipe.options["max_iter"] == 500
        assert result.converged

        unrequested = multi_start_fit(
            TwoParameterLogistic(4),
            data,
            n_starts=2,
            seed=2,
            compute_standard_errors=False,
        )
        assert standard_error_calls == [1]
        assert unrequested.standard_errors == {}

    def test_latent_density_instances_are_not_shared_or_mutated(
        self, correlated_data, replicate_fits
    ):
        spec = mirt.mirt_model(_SPEC)
        model = mirt.MultidimensionalModel(
            8,
            2,
            model_type="confirmatory",
            loading_pattern=spec.loading_pattern(8).astype(float),
        )
        density = FactorCovarianceDensity(2)

        result = multi_start_fit(
            model,
            correlated_data,
            n_starts=2,
            seed=1,
            n_quadpts=9,
            latent_density=density,
        )

        np.testing.assert_array_equal(density.cov, np.eye(2))
        used = [estimator._latent_density for estimator, _ in replicate_fits]
        assert len({id(value) for value in used}) == len(used) == 3
        assert result.latent_covariance[0, 1] > 0.4
        # The recipe continues from the returned estimates, after the extra
        # standard-error iteration, with the starts' iteration limit.
        np.testing.assert_array_equal(
            result.refit_recipe.latent_density.cov, result.latent_covariance
        )
        assert result.refit_recipe.options["max_iter"] == 500

    def test_bayes_modal_starts_are_ranked_by_log_posterior(self, monkeypatch):
        posteriors = iter([-10.0, -5.0])
        likelihoods = iter([-1.0, -9.0])
        original = EMEstimator.fit

        def scored_fit(self, model, responses, *args, **kwargs):
            result = original(self, model, responses, *args, **kwargs)
            return replace(
                result,
                log_likelihood=next(likelihoods),
                log_posterior=next(posteriors),
            )

        monkeypatch.setattr(EMEstimator, "fit", scored_fit)
        data = np.asarray(simdata("2PL", n_persons=100, n_items=3, seed=1))
        result = multi_start_fit(
            TwoParameterLogistic(3),
            data,
            n_starts=2,
            seed=1,
            max_iter=5,
            compute_standard_errors=False,
            item_priors={"discrimination": LogNormalPrior(0, 0.5)},
        )

        assert result.log_posterior == -5.0

    def test_standard_error_requests_are_validated(self):
        with pytest.raises(MirtValidationError, match="compute_standard_errors"):
            multi_start_fit(
                TwoParameterLogistic(2),
                np.array([[0, 1], [1, 0]]),
                compute_standard_errors="yes",
            )

    def test_bifactor_models_use_dimension_reduction(self, bifactor_fit):
        result, data = bifactor_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            best = multi_start_fit(result.model, data, n_starts=2, seed=1, max_iter=50)

        assert best.refit_recipe.estimator is BifactorEMEstimator
        assert best.se_method == "oakes"
        assert np.isfinite(best.log_likelihood)


def test_gaussian_latent_density_population_is_recorded():
    data = np.asarray(simdata("2PL", n_persons=300, n_items=6, seed=12))
    density = GaussianDensity(estimate_cov=True)
    # Equal slopes identify the latent variance.
    estimated = EMEstimator(latent_density=density).fit(OneParameterLogistic(6), data)
    fixed = EMEstimator().fit(
        TwoParameterLogistic(6), data, prior_mean=np.array([0.5]), prior_cov=[[2.0]]
    )
    default = EMEstimator().fit(TwoParameterLogistic(6), data)

    assert estimated.latent_mean is None
    np.testing.assert_array_equal(estimated.latent_covariance, density.cov)
    assert estimated.refit_recipe.latent_density.estimate_cov
    np.testing.assert_array_equal(fixed.latent_mean, [0.5])
    np.testing.assert_array_equal(fixed.latent_covariance, [[2.0]])
    assert default.latent_mean is None and default.latent_covariance is None
    assert fixed.refit_recipe.latent_density.mean[0] == 0.5
    assert default.refit_recipe.latent_density is None
