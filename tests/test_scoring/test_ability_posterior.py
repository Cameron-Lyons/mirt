"""Tests for grid-based posterior ability distributions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt import AbilityPosteriorResult, ability_posterior
from mirt.exceptions import MirtValidationError
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.results import FitResult, ScoreResult
from mirt.scoring._common import build_quadrature
from mirt.scoring.eap import EAPScorer
from mirt.utils.numeric import logsumexp_axis1


def _fitted_2pl(*, n_factors: int = 1) -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=3, n_factors=n_factors)
    discrimination = (
        np.array([0.8, 1.1, 1.4])
        if n_factors == 1
        else np.array([[1.2, 0.1], [0.2, 1.4], [0.8, 0.9]])
    )
    model.set_parameters(
        discrimination=discrimination,
        difficulty=np.array([-0.7, 0.1, 0.9]),
    )
    model._is_fitted = True
    return model


def _small_result() -> AbilityPosteriorResult:
    return AbilityPosteriorResult(
        points=np.array([-1.0, 0.0, 2.0]),
        weights=np.array([[0.2, 0.3, 0.5], [0.8, 0.2, 0.0]]),
        log_marginal_likelihood=np.array([-2.0, -3.0]),
        person_ids=["low", "high"],
    )


class TestAbilityPosteriorResult:
    def test_exposes_grid_dimensions_and_moments(self) -> None:
        result = _small_result()

        assert result.n_persons == 2
        assert result.n_points == 3
        assert result.n_factors == 1
        assert_allclose(result.mean, [0.8, -0.8])
        assert_allclose(result.standard_error, [1.2489996, 0.4])
        assert_allclose(result.map_estimate, [2.0, -1.0])
        assert_allclose(
            result.entropy,
            [
                -np.sum(np.array([0.2, 0.3, 0.5]) * np.log([0.2, 0.3, 0.5])),
                -np.sum(np.array([0.8, 0.2]) * np.log([0.8, 0.2])),
            ],
        )
        assert "n_persons=2" in repr(result)

    def test_equal_tail_intervals_use_weighted_grid_quantiles(self) -> None:
        result = _small_result()
        lower, upper = result.credible_intervals(level=0.8)

        assert_array_equal(lower, [-1.0, -1.0])
        assert_array_equal(upper, [2.0, 0.0])
        assert_array_equal(result.quantile(0.5), [0.0, -1.0])
        assert_array_equal(result.median, [0.0, -1.0])

    @pytest.mark.parametrize("level", [0.0, 1.0, np.nan, True, "0.9"])
    def test_rejects_invalid_credible_levels(self, level: object) -> None:
        with pytest.raises(MirtValidationError, match="level"):
            _small_result().credible_intervals(level=level)  # type: ignore[arg-type]

    def test_rejects_invalid_quantile_probability(self) -> None:
        with pytest.raises(MirtValidationError, match="probability"):
            _small_result().quantile(1.0)

    def test_exact_threshold_probabilities_and_decisions(self) -> None:
        result = _small_result()

        assert_allclose(result.classification_probabilities(), [0.5, 0.0])
        assert_allclose(
            result.classification_probabilities(cut_score=[1.0, -2.0]),
            [0.5, 1.0],
        )
        assert_array_equal(
            result.classify(confidence=0.75),
            ["uncertain", "below"],
        )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, np.inf, True])
    def test_rejects_invalid_classification_confidence(
        self,
        confidence: object,
    ) -> None:
        with pytest.raises(MirtValidationError, match="confidence"):
            _small_result().classify(confidence=confidence)  # type: ignore[arg-type]

    @pytest.mark.parametrize("cut_score", [[0.0, 1.0, 2.0], np.nan, ["x"]])
    def test_rejects_invalid_cut_scores(self, cut_score: object) -> None:
        with pytest.raises(MirtValidationError, match="cut_score"):
            _small_result().classification_probabilities(cut_score)

    def test_converts_to_score_result_with_person_ids(self) -> None:
        score = _small_result().to_score_result()

        assert isinstance(score, ScoreResult)
        assert_allclose(score.theta, [0.8, -0.8])
        assert_allclose(score.standard_error, [1.2489996, 0.4])
        assert score.person_ids == ["low", "high"]

    def test_multidimensional_summaries_preserve_factor_axis(self) -> None:
        result = AbilityPosteriorResult(
            points=np.array([[-1.0, -2.0], [0.0, 1.0], [2.0, 3.0]]),
            weights=np.array([[0.2, 0.3, 0.5], [0.8, 0.2, 0.0]]),
            log_marginal_likelihood=np.array([-1.0, -2.0]),
        )

        assert result.mean.shape == (2, 2)
        assert result.standard_error.shape == (2, 2)
        assert result.map_estimate.shape == (2, 2)
        lower, upper = result.credible_intervals(level=0.8)
        assert lower.shape == upper.shape == (2, 2)
        probabilities = result.classification_probabilities([0.0, 0.0])
        assert probabilities.shape == (2, 2)
        assert_allclose(probabilities, [[0.5, 0.8], [0.0, 0.2]])

    def test_copies_caller_owned_arrays(self) -> None:
        points = np.array([-1.0, 1.0])
        weights = np.array([[0.5, 0.5]])
        log_marginal = np.array([-1.0])
        result = AbilityPosteriorResult(points, weights, log_marginal)

        points[:] = 10.0
        weights[:] = 0.0
        log_marginal[:] = 0.0

        assert_allclose(result.points.ravel(), [-1.0, 1.0])
        assert_allclose(result.weights, [[0.5, 0.5]])
        assert_allclose(result.log_marginal_likelihood, [-1.0])

    @pytest.mark.parametrize(
        ("points", "weights", "log_marginal", "message"),
        [
            ([], np.empty((1, 0)), [-1.0], "points"),
            ([0.0, np.nan], [[0.5, 0.5]], [-1.0], "points"),
            ([-1.0, 1.0], [[1.0]], [-1.0], "weights"),
            ([-1.0, 1.0], [[0.5, -0.5]], [-1.0], "non-negative"),
            ([-1.0, 1.0], [[0.2, 0.2]], [-1.0], "sum to one"),
            ([-1.0, 1.0], [[0.5, 0.5]], [], "log_marginal"),
        ],
    )
    def test_validates_distribution_contract(
        self,
        points: object,
        weights: object,
        log_marginal: object,
        message: str,
    ) -> None:
        with pytest.raises(MirtValidationError, match=message):
            AbilityPosteriorResult(points, weights, log_marginal)  # type: ignore[arg-type]


class TestAbilityPosteriorScoring:
    def test_matches_direct_normalized_likelihood_reference(self) -> None:
        model = _fitted_2pl()
        responses = np.array([[0, 0, 1], [1, 1, 1], [1, -9, 0]])
        scorer = EAPScorer(n_quadpts=15)

        result = scorer.posterior(model, responses)
        points, prior_weights = build_quadrature(
            n_quadpts=15,
            n_factors=1,
            prior_mean=None,
            prior_cov=None,
        )
        normalized_responses = np.where(responses >= 0, responses, -1)
        log_joint = model.log_likelihood_batch(normalized_responses, points)
        log_joint += np.log(prior_weights + 1e-300)[None, :]
        expected_log_marginal = logsumexp_axis1(log_joint)
        expected_weights = np.exp(log_joint - expected_log_marginal[:, None])

        assert_allclose(result.points, points)
        assert_allclose(result.weights, expected_weights, rtol=1e-13, atol=1e-15)
        assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-15)
        assert_allclose(
            result.log_marginal_likelihood,
            expected_log_marginal,
            rtol=1e-13,
            atol=1e-15,
        )

    def test_moments_match_regular_eap_scores(self) -> None:
        model = _fitted_2pl()
        responses = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
        scorer = EAPScorer(n_quadpts=21)

        posterior = scorer.posterior(model, responses)
        scores = scorer.score(model, responses)

        assert_allclose(posterior.mean, scores.theta, rtol=1e-13, atol=1e-14)
        assert_allclose(
            posterior.standard_error,
            scores.standard_error,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_custom_prior_is_recovered_for_all_missing_responses(self) -> None:
        model = _fitted_2pl()
        scorer = EAPScorer(
            n_quadpts=21,
            prior_mean=np.array([1.5]),
            prior_cov=np.array([[2.25]]),
        )

        posterior = scorer.posterior(model, np.array([[-1, -5, -99]]))

        assert_allclose(posterior.mean, [1.5], atol=1e-14)
        assert_allclose(posterior.standard_error, [1.5], atol=1e-14)
        assert_allclose(posterior.log_marginal_likelihood, [0.0], atol=1e-14)

    def test_batching_bounds_likelihood_calls(self, monkeypatch) -> None:
        model = _fitted_2pl()
        responses = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0], [1, -1, 0]])
        expected = EAPScorer(n_quadpts=11).posterior(model, responses)
        original = model.log_likelihood_batch
        call_sizes: list[int] = []

        def capture(response_batch, theta):
            call_sizes.append(len(response_batch))
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)
        actual = EAPScorer(n_quadpts=11, batch_size=2).posterior(model, responses)

        assert call_sizes == [2, 2, 1]
        assert_allclose(actual.weights, expected.weights, rtol=0.0, atol=0.0)
        assert_allclose(
            actual.log_marginal_likelihood,
            expected.log_marginal_likelihood,
            rtol=0.0,
            atol=0.0,
        )

    def test_supports_empty_response_batches(self) -> None:
        posterior = EAPScorer(n_quadpts=9).posterior(
            _fitted_2pl(),
            np.empty((0, 3), dtype=int),
            person_ids=[],
        )

        assert posterior.weights.shape == (0, 9)
        assert posterior.log_marginal_likelihood.shape == (0,)
        assert posterior.mean.shape == (0,)
        assert posterior.standard_error.shape == (0,)
        assert posterior.map_estimate.shape == (0,)
        lower, upper = posterior.credible_intervals()
        assert lower.shape == upper.shape == (0,)

    def test_supports_multidimensional_models(self) -> None:
        model = _fitted_2pl(n_factors=2)
        responses = np.array([[0, 0, 1], [1, 1, 0], [1, -1, 1]])

        posterior = EAPScorer(n_quadpts=7).posterior(model, responses)
        scores = EAPScorer(n_quadpts=7).score(model, responses)

        assert posterior.points.shape == (49, 2)
        assert posterior.weights.shape == (3, 49)
        assert posterior.mean.shape == (3, 2)
        assert_allclose(posterior.mean, scores.theta, rtol=1e-13, atol=1e-14)
        assert_allclose(
            posterior.standard_error,
            scores.standard_error,
            rtol=1e-13,
            atol=1e-14,
        )

    def test_supports_polytomous_models_and_missing_values(self) -> None:
        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model.set_parameters(
            discrimination=np.array([0.8, 1.1, 1.4]),
            thresholds=np.array([[-1.0, 1.0, 0.0], [-1.5, 0.0, 1.5], [-0.8, 0.7, 0.0]]),
        )
        model._is_fitted = True
        responses = np.array([[0, 1, 2], [2, 3, 1], [1, -4, 0]])

        posterior = ability_posterior(model, responses, n_quadpts=11)

        assert posterior.weights.shape == (3, 11)
        assert_allclose(posterior.weights.sum(axis=1), 1.0, atol=1e-15)
        assert np.all(np.isfinite(posterior.log_marginal_likelihood))

    def test_public_function_accepts_fit_results_and_person_ids(self) -> None:
        model = _fitted_2pl()
        fit = FitResult(
            model=model,
            log_likelihood=-10.0,
            n_iterations=1,
            converged=True,
            standard_errors={},
            aic=20.0,
            bic=21.0,
        )
        responses = np.array([[0, 1, 0], [1, 0, 1]])

        posterior = ability_posterior(
            fit,
            responses,
            n_quadpts=9,
            person_ids=np.array([101, 102]),
        )

        assert posterior.person_ids == [101, 102]
        assert isinstance(posterior.to_score_result(), ScoreResult)

    def test_rejects_unfitted_models(self) -> None:
        model = TwoParameterLogistic(n_items=3)

        with pytest.raises(ValueError, match="fitted"):
            ability_posterior(model, np.array([[0, 1, 0]]))
