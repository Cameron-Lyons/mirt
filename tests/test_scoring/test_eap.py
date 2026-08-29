"""Tests for EAP scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt import fscores
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.scoring._common import build_quadrature, validate_scoring_responses
from mirt.scoring.eap import EAPScorer, _eap_response_patterns
from mirt.utils.numeric import logsumexp_axis1


class TestEAPScorerInitialization:
    """Tests for EAPScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = EAPScorer()

        assert scorer.n_quadpts == 49
        assert scorer.prior_mean is None
        assert scorer.prior_cov is None
        assert scorer.batch_size is None

    def test_custom_n_quadpts(self):
        """Test initialization with custom quadrature points."""
        scorer = EAPScorer(n_quadpts=31)

        assert scorer.n_quadpts == 31

    def test_invalid_n_quadpts(self):
        """Test that invalid n_quadpts raises error."""
        with pytest.raises(ValueError, match="at least 5"):
            EAPScorer(n_quadpts=3)

    @pytest.mark.parametrize("batch_size", [0, -1, True, 2.5, "10"])
    def test_rejects_invalid_batch_size(self, batch_size):
        """Reject non-positive and ambiguous respondent batch sizes."""
        with pytest.raises(ValueError, match="batch_size"):
            EAPScorer(batch_size=batch_size)

    @pytest.mark.parametrize("n_quadpts", [True, 5.5, "7"])
    def test_rejects_non_integer_n_quadpts(self, n_quadpts):
        """Reject ambiguous quadrature-size inputs during construction."""
        with pytest.raises(ValueError, match="at least 5"):
            EAPScorer(n_quadpts=n_quadpts)

    def test_custom_prior_mean(self):
        """Test initialization with custom prior mean."""
        prior_mean = np.array([0.5])
        scorer = EAPScorer(prior_mean=prior_mean)

        assert_allclose(scorer.prior_mean, prior_mean)

    def test_custom_prior_cov(self):
        """Test initialization with custom prior covariance."""
        prior_cov = np.array([[2.0]])
        scorer = EAPScorer(prior_cov=prior_cov)

        assert_allclose(scorer.prior_cov, prior_cov)

    def test_prior_inputs_are_detached(self):
        """Later caller mutations cannot reconfigure an existing scorer."""
        prior_mean = np.array([0.5])
        prior_cov = np.array([[2.0]])

        scorer = EAPScorer(prior_mean=prior_mean, prior_cov=prior_cov)
        prior_mean[0] = 9.0
        prior_cov[0, 0] = 9.0

        assert_allclose(scorer.prior_mean, [0.5])
        assert_allclose(scorer.prior_cov, [[2.0]])

    def test_repr(self):
        """Test __repr__ method."""
        scorer = EAPScorer(n_quadpts=21)
        repr_str = repr(scorer)

        assert "EAPScorer" in repr_str
        assert "21" in repr_str
        assert "batch_size" not in repr_str
        assert "batch_size=10" in repr(EAPScorer(batch_size=10))


class TestEAPScorerScoring:
    """Tests for EAPScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic EAP scoring."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "EAP"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_reasonable_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are in reasonable range."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta > -6)
        assert np.all(result.theta < 6)

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = EAPScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])

    @pytest.mark.parametrize(
        ("responses", "message"),
        [
            ([0, 1], "2D"),
            ([[0, 1, 0]], "items, expected"),
            ([[0.0, 0.5]], "integer-valued"),
            ([[0.0, np.nan]], "finite"),
            ([[0, 2]], "only 0, 1"),
            ([["yes", "no"]], "numeric"),
        ],
    )
    def test_validates_response_contracts(self, responses, message):
        """Reject malformed dichotomous response matrices before scoring."""
        model = TwoParameterLogistic(n_items=2)
        model._is_fitted = True

        with pytest.raises(ValueError, match=message):
            EAPScorer().score(model, responses)

    def test_validates_polytomous_categories(self):
        """Reject categories outside an item's configured range."""
        model = GradedResponseModel(n_items=2, n_categories=[3, 4])
        model._is_fitted = True

        with pytest.raises(ValueError, match="category range"):
            EAPScorer().score(model, np.array([[3, 0]]))

    def test_normalizes_missing_codes_before_likelihood(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Equivalent negative missing codes share the canonical representation."""
        model = TwoParameterLogistic(n_items=2)
        model._is_fitted = True
        original = model.log_likelihood_batch
        received = []

        def capture(responses, theta):
            received.append(responses.copy())
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)
        result = EAPScorer(n_quadpts=9).score(
            model,
            np.array([[-1, 0], [-999, 0]]),
        )

        assert_allclose(received[0], [[-1, 0]])
        assert_allclose(result.theta[0], result.theta[1])
        assert_allclose(result.standard_error[0], result.standard_error[1])

    def test_explicit_batches_preserve_scores_and_call_bounds(self, monkeypatch):
        """Explicit batching bounds likelihood calls without changing results."""
        model = TwoParameterLogistic(n_items=4)
        model.set_parameters(
            discrimination=np.array([0.8, 1.1, 1.4, 1.7]),
            difficulty=np.array([-1.0, -0.2, 0.4, 1.2]),
        )
        model._is_fitted = True
        responses = np.array(
            [
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [1, 1, -9, 0],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
            ]
        )
        expected = EAPScorer(n_quadpts=21, batch_size=100).score(model, responses)
        original = model.log_likelihood_batch
        call_sizes = []

        def capture(response_batch, theta):
            call_sizes.append(len(response_batch))
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)
        actual = EAPScorer(n_quadpts=21, batch_size=2).score(model, responses)

        assert call_sizes == [2, 2, 1]
        assert_allclose(actual.theta, expected.theta, rtol=0.0, atol=1e-14)
        assert_allclose(
            actual.standard_error,
            expected.standard_error,
            rtol=0.0,
            atol=1e-14,
        )

    def test_explicit_batches_preserve_polytomous_scores(self):
        """Respondent batching also preserves category-response likelihoods."""
        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model._is_fitted = True
        responses = np.array(
            [
                [0, 1, 2],
                [2, 3, 1],
                [1, -4, 0],
                [2, 2, 2],
                [0, 0, 0],
            ]
        )

        expected = EAPScorer(n_quadpts=21, batch_size=100).score(model, responses)
        actual = EAPScorer(n_quadpts=21, batch_size=2).score(model, responses)

        assert_allclose(actual.theta, expected.theta, rtol=0.0, atol=1e-14)
        assert_allclose(
            actual.standard_error,
            expected.standard_error,
            rtol=0.0,
            atol=1e-14,
        )

    def test_explicit_batches_preserve_multidimensional_scores(self):
        """Batch boundaries do not alter multidimensional posterior moments."""
        model = TwoParameterLogistic(n_items=4, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[1.2, 0.1], [0.2, 1.3], [0.8, 0.7], [1.0, 0.5]]),
            difficulty=np.array([-0.8, -0.1, 0.5, 1.0]),
        )
        model._is_fitted = True
        responses = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, -1, 0], [0, 0, 0, 1]])

        expected = EAPScorer(n_quadpts=5, batch_size=100).score(model, responses)
        actual = EAPScorer(n_quadpts=5, batch_size=2).score(model, responses)

        assert_allclose(actual.theta, expected.theta, rtol=0.0, atol=1e-14)
        assert_allclose(
            actual.standard_error,
            expected.standard_error,
            rtol=0.0,
            atol=1e-14,
        )

    def test_automatic_batching_bounds_large_working_sets(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The default splits work once its estimated temporary size is large."""
        model = TwoParameterLogistic(n_items=3)
        model._is_fitted = True
        responses = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
            ]
        )
        original = model.log_likelihood_batch
        call_sizes = []

        def capture(response_batch, theta):
            call_sizes.append(len(response_batch))
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)
        monkeypatch.setattr("mirt.scoring.eap._TARGET_WORKING_BYTES", 1)

        EAPScorer(n_quadpts=9).score(model, responses)

        assert call_sizes == [1, 1, 1, 1, 1]

    def test_duplicate_patterns_share_one_likelihood_evaluation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Repeated rows reuse posterior moments and retain respondent order."""
        model = TwoParameterLogistic(n_items=4)
        model._is_fitted = True
        first = np.array([0, 1, 0, 1])
        second = np.array([1, 0, 1, 0])
        responses = np.vstack([first, second, first, first, second])
        original = model.log_likelihood_batch
        call_sizes = []

        def capture(response_batch, theta):
            call_sizes.append(len(response_batch))
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)

        result = EAPScorer(n_quadpts=9).score(model, responses)

        assert sum(call_sizes) == 2
        assert_allclose(result.theta[[0, 2, 3]], result.theta[0])
        assert_allclose(result.standard_error[[0, 2, 3]], result.standard_error[0])
        assert_allclose(result.theta[[1, 4]], result.theta[1])

    def test_near_unique_large_input_skips_full_pattern_sort(self):
        """Adaptive compression preserves the fast vectorized path."""
        row_ids = np.arange(2_048, dtype=np.int_)[:, None]
        bit_positions = np.arange(12, dtype=np.int_)[None, :]
        responses = (row_ids >> bit_positions) & 1

        patterns, inverse = _eap_response_patterns(responses)

        assert np.shares_memory(patterns, responses)
        np.testing.assert_array_equal(inverse, np.arange(responses.shape[0]))

    def test_repetition_heavy_large_input_uses_sampled_compression(self):
        """A bounded sample triggers full compression when reuse is substantial."""
        row_ids = np.arange(16, dtype=np.int_)[:, None]
        bit_positions = np.arange(12, dtype=np.int_)[None, :]
        source_patterns = (row_ids >> bit_positions) & 1
        responses = source_patterns[np.arange(4_096) % source_patterns.shape[0]]

        patterns, inverse = _eap_response_patterns(responses)

        assert patterns.shape == (16, 12)
        np.testing.assert_array_equal(patterns[inverse], responses)

    def test_fscores_forwards_eap_batch_size(self, monkeypatch):
        """The public scoring entry point exposes respondent batching."""
        model = TwoParameterLogistic(n_items=2)
        model._is_fitted = True
        responses = np.array([[0, 1], [1, 0], [1, 1]])
        original = model.log_likelihood_batch
        call_sizes = []

        def capture(response_batch, theta):
            call_sizes.append(len(response_batch))
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)

        fscores(model, responses, method="EAP", n_quadpts=9, batch_size=1)

        assert call_sizes == [1, 1, 1]

    def test_reuses_canonical_integer_response_storage(self):
        """Validation does not copy an already normalized native integer matrix."""
        model = TwoParameterLogistic(n_items=2)
        responses = np.array([[0, 1], [-1, 0]], dtype=np.int_)

        normalized = validate_scoring_responses(model, responses)

        assert np.shares_memory(normalized, responses)

    def test_supports_empty_batches(self):
        """Return empty score arrays for an empty, correctly shaped matrix."""
        model = TwoParameterLogistic(n_items=2)
        model._is_fitted = True

        result = EAPScorer().score(model, np.empty((0, 2), dtype=int))

        assert result.theta.shape == (0,)
        assert result.standard_error.shape == (0,)

    def test_centered_moments_match_reference_calculation(self):
        """The allocation-light moment kernel preserves EAP estimates."""
        model = TwoParameterLogistic(n_items=3, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[1.2, 0.2], [0.1, 1.4], [0.8, 0.9]]),
            difficulty=np.array([-0.5, 0.2, 0.8]),
        )
        model._is_fitted = True
        responses = np.array([[1, 0, 1], [0, 1, -7], [1, 1, 0], [1, 0, 1]])
        prior_mean = np.array([100_000.0, -100_000.0])
        prior_cov = np.array([[1.5, 0.2], [0.2, 0.8]])
        scorer = EAPScorer(
            n_quadpts=5,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )

        result = scorer.score(model, responses)
        normalized = validate_scoring_responses(model, responses)
        points, weights = build_quadrature(
            n_quadpts=5,
            n_factors=2,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
        log_posterior = model.log_likelihood_batch(normalized, points)
        log_posterior += np.log(weights + 1e-300)[None, :]
        log_posterior -= logsumexp_axis1(log_posterior)[:, None]
        posterior = np.exp(log_posterior)
        expected_theta = posterior @ points
        deviation = points[None, :, :] - expected_theta[:, None, :]
        expected_se = np.sqrt(np.sum(posterior[:, :, None] * deviation**2, axis=1))

        assert_allclose(result.theta, expected_theta, rtol=1e-12, atol=1e-9)
        assert_allclose(result.standard_error, expected_se, rtol=1e-10, atol=1e-10)


class TestEAPScorerCustomPrior:
    """Tests for EAP with custom prior parameters."""

    def test_shifted_prior_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test that shifted prior mean affects estimates."""
        model = fitted_2pl_model.model
        scorer_default = EAPScorer()
        scorer_shifted = EAPScorer(prior_mean=np.array([1.0]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_shifted = scorer_shifted.score(model, dichotomous_responses["responses"])

        mean_diff = result_shifted.theta.mean() - result_default.theta.mean()
        assert mean_diff > 0

    def test_larger_prior_variance(self, fitted_2pl_model, dichotomous_responses):
        """Test effect of larger prior variance."""
        model = fitted_2pl_model.model
        scorer_default = EAPScorer()
        scorer_large_var = EAPScorer(prior_cov=np.array([[4.0]]))

        result_default = scorer_default.score(model, dichotomous_responses["responses"])
        result_large_var = scorer_large_var.score(
            model, dichotomous_responses["responses"]
        )

        var_default = np.var(result_default.theta)
        var_large = np.var(result_large_var.theta)
        assert var_large >= var_default * 0.9


class TestEAPScorerMultidimensional:
    """Tests for EAP with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional EAP scoring."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)
        n_persons = 100
        n_items = 12

        theta = rng.standard_normal((n_persons, 2))
        loading = np.zeros((n_items, 2))
        loading[:6, 0] = rng.uniform(0.5, 1.5, 6)
        loading[6:, 1] = rng.uniform(0.5, 1.5, 6)
        diff = rng.normal(0, 1, n_items)

        logit = theta @ loading.T - diff
        prob = 1 / (1 + np.exp(-logit))
        responses = (rng.random((n_persons, n_items)) < prob).astype(int)

        result = fit_mirt(responses, model="2PL", n_factors=2, max_iter=20, n_quadpts=7)

        scorer = EAPScorer(n_quadpts=7)
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)

    def test_multidimensional_custom_prior(self):
        """Test multidimensional EAP with custom prior."""
        from mirt import fit_mirt

        rng = np.random.default_rng(42)
        n_persons = 50
        n_items = 8

        theta = rng.standard_normal((n_persons, 2))
        loading = rng.uniform(0.5, 1.5, (n_items, 2))
        diff = rng.normal(0, 1, n_items)

        logit = theta @ loading.T - diff
        prob = 1 / (1 + np.exp(-logit))
        responses = (rng.random((n_persons, n_items)) < prob).astype(int)

        result = fit_mirt(responses, model="2PL", n_factors=2, max_iter=15, n_quadpts=5)

        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.eye(2) * 2.0

        scorer = EAPScorer(n_quadpts=5, prior_mean=prior_mean, prior_cov=prior_cov)
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)


class TestEAPScorerConsistency:
    """Tests for EAP scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)
        assert_allclose(result1.standard_error, result2.standard_error)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        scorer = EAPScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        sum_scores = dichotomous_responses["responses"].sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7


class TestEAPScorerPolytomous:
    """Tests for EAP with polytomous models."""

    def test_response_patterns_receive_distinct_scores(self):
        """EAP evaluates each respondent's complete response pattern."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 5])
        model.set_parameters(
            discrimination=np.array([0.8, 1.1, 1.4]),
            thresholds=np.array(
                [
                    [-1.0, 1.0, 0.0, 0.0],
                    [-1.5, 0.0, 1.5, 0.0],
                    [-2.0, -0.6, 0.6, 2.0],
                ]
            ),
        )
        model._is_fitted = True
        responses = np.array(
            [
                [0, 0, 0],
                [2, 3, 4],
                [-1, -1, -1],
            ]
        )

        scores = EAPScorer(n_quadpts=21).score(model, responses)

        assert scores.theta[0] < -0.5
        assert scores.theta[1] > 0.5
        assert scores.theta[2] == pytest.approx(0.0, abs=1e-12)

    def test_polytomous_scoring(self, polytomous_responses):
        """Test EAP scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = EAPScorer(n_quadpts=15)
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)
        assert np.all(scores.standard_error > 0)
