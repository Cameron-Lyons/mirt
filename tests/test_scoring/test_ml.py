"""Tests for ML scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.scoring.ml import MLScorer


class TestMLScorerInitialization:
    """Tests for MLScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = MLScorer()

        assert scorer.theta_bounds == (-6.0, 6.0)
        assert scorer.n_jobs == 1

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        scorer = MLScorer(theta_bounds=(-4.0, 4.0))

        assert scorer.theta_bounds == (-4.0, 4.0)

    def test_parallel_n_jobs(self):
        """Test initialization with parallel processing."""
        scorer = MLScorer(n_jobs=2)

        assert scorer.n_jobs == 2

    def test_repr(self):
        """Test __repr__ method."""
        scorer = MLScorer(theta_bounds=(-5.0, 5.0))
        repr_str = repr(scorer)

        assert "MLScorer" in repr_str
        assert "-5" in repr_str


class TestMLScorerScoring:
    """Tests for MLScorer scoring."""

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic ML scoring."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "ML"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive_or_inf(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive or infinite."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_within_bounds(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are within bounds."""
        model = fitted_2pl_model.model
        bounds = (-4.0, 4.0)
        scorer = MLScorer(theta_bounds=bounds)

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta >= bounds[0])
        assert np.all(result.theta <= bounds[1])

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = MLScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestMLScorerPerfectZeroScores:
    """Tests for ML with perfect and zero scores."""

    def test_perfect_score_at_upper_bound(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that perfect score returns upper bound."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 1

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == scorer.theta_bounds[1]
        assert np.isinf(result.standard_error[0])

    def test_zero_score_at_lower_bound(self, fitted_2pl_model, dichotomous_responses):
        """Test that zero score returns lower bound."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 0

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == scorer.theta_bounds[0]
        assert np.isinf(result.standard_error[0])


class TestMLScorerMultidimensional:
    """Tests for ML with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional ML scoring."""
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

        scorer = MLScorer()
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)


class TestMLScorerConsistency:
    """Tests for ML scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = MLScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)

        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])
        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        scorer = MLScorer()
        result = scorer.score(model, responses)

        correlation = np.corrcoef(result.theta[not_extreme], sum_scores[not_extreme])[
            0, 1
        ]

        assert correlation > 0.7


class TestMLScorerPolytomous:
    """Tests for ML with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test ML scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = MLScorer()
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)

    def test_middle_categories_are_optimized_instead_of_treated_as_perfect(self):
        """Category one is not a perfect score for a three-category item."""
        from mirt.models import GradedResponseModel

        model = GradedResponseModel(n_items=5, n_categories=3)
        model._is_fitted = True
        responses = np.ones((1, model.n_items), dtype=np.int_)

        result = MLScorer().score(model, responses)

        assert result.theta[0] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(result.standard_error[0])


class TestMLVsEAP:
    """Tests comparing ML and EAP estimates."""

    def test_ml_eap_correlation(self, fitted_2pl_model, dichotomous_responses):
        """Test that ML and EAP estimates are highly correlated."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])

        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        ml_result = ml_scorer.score(model, responses)
        eap_result = eap_scorer.score(model, responses)

        correlation = np.corrcoef(
            ml_result.theta[not_extreme], eap_result.theta[not_extreme]
        )[0, 1]

        assert correlation > 0.85

    def test_ml_more_extreme_than_eap(self, fitted_2pl_model, dichotomous_responses):
        """Test that ML estimates are more extreme than EAP."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        ml_result = ml_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        ml_var = np.var(ml_result.theta)
        eap_var = np.var(eap_result.theta)

        assert ml_var >= eap_var * 0.9


class TestMLScorerEmptyResponses:
    """Tests for ML with empty/missing responses."""

    def test_standard_error_uses_only_observed_items(self):
        """Test that missing items do not contribute score information."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=4)
        model.set_parameters(
            discrimination=np.array([1.0, 1.0, 50.0, 50.0]),
            difficulty=np.zeros(4),
        )
        model._is_fitted = True

        result = MLScorer().score(model, np.array([[1, 0, -1, -1]]))

        assert result.theta[0] == pytest.approx(0.0, abs=1e-6)
        assert result.standard_error[0] == pytest.approx(np.sqrt(2.0))

    def test_all_missing_returns_zero(self, fitted_2pl_model):
        """Test that all missing returns zero theta."""
        model = fitted_2pl_model.model
        responses = np.full((1, model.n_items), -1, dtype=int)

        scorer = MLScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == 0.0
        assert np.isinf(result.standard_error[0])


# Two-factor simple structure, rounded from the parity fit, with R mirt 1.48
# references from fscores(method="ML", max_theta=6) on the same parameters.
_PARITY_SLOPES = np.array(
    [
        [0.85, 0.0],
        [1.45, 0.0],
        [1.19, 0.0],
        [0.83, 0.0],
        [2.30, 0.0],
        [0.0, 1.54],
        [0.0, 2.09],
        [0.0, 1.93],
        [0.0, 1.21],
        [0.0, 1.29],
    ]
)
_PARITY_INTERCEPTS = np.array(
    [0.39, -1.20, 0.35, -0.97, 0.11, 0.55, 0.20, -1.19, 0.78, 0.67]
)
_PARITY_PATTERNS = np.array(
    [
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [-1, -1, -1, 0, 0, 0, 0, 0, 1, 1],
        [1] * 10,
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    ]
)


def _parity_model():
    from mirt.models import MultidimensionalModel

    model = MultidimensionalModel(10, n_factors=2)
    model.set_parameters(slopes=_PARITY_SLOPES, intercepts=_PARITY_INTERCEPTS)
    model._is_fitted = True
    return model


def _extreme_cases():
    from mirt.models import (
        BifactorModel,
        GradedResponseModel,
        MixedItemModel,
        MultidimensionalModel,
        NoncompensatoryModel,
        TwoParameterLogistic,
    )

    rng = np.random.default_rng(12)
    two_pl = TwoParameterLogistic(4)
    two_pl.set_parameters(
        discrimination=rng.uniform(0.7, 1.8, 4), difficulty=rng.normal(size=4)
    )
    graded = GradedResponseModel(4, n_categories=[3, 4, 3, 5])
    efa = TwoParameterLogistic(6, n_factors=2)
    efa.set_parameters(
        discrimination=rng.uniform(0.4, 1.6, (6, 2)), difficulty=rng.normal(size=6)
    )
    mirt3 = MultidimensionalModel(6, n_factors=3)
    mirt3.set_parameters(slopes=rng.uniform(0.3, 1.5, (6, 3)))
    bifactor = BifactorModel(6, specific_factors=[0, 0, 0, 1, 1, 1])
    graded2 = GradedResponseModel(5, n_categories=3, n_factors=2)
    noncompensatory = NoncompensatoryModel(4, n_factors=2)
    cases = [two_pl, graded, efa, mirt3, bifactor, graded2, noncompensatory]
    for model in cases:
        model._is_fitted = True
    mixed = MixedItemModel([(two_pl, [0, 2, 4, 6]), (graded, [1, 3, 5, 7])])
    mixed._is_fitted = True
    return [*cases, mixed]


@pytest.mark.parametrize(
    "model",
    _extreme_cases(),
    ids=["2PL", "GRM", "2PL-2f", "MIRT-3f", "bifactor", "GRM-2f", "noncomp", "mixed"],
)
def test_ml_extreme_patterns_follow_one_convention_for_every_family(model):
    """All-lowest and all-highest patterns sit on the bounds with infinite SE."""
    top = np.asarray(model.n_categories) - 1 if model.is_polytomous else 1
    lowest = np.zeros((1, model.n_items), dtype=int)
    highest = np.broadcast_to(top, (1, model.n_items)).astype(int).copy()
    lowest[0, 1] = -1
    highest[0, 0] = -1
    bounds = (-4.0, 3.0)

    result = MLScorer(theta_bounds=bounds).score(model, np.vstack([lowest, highest]))

    theta = result.theta.reshape(2, -1)
    standard_error = result.standard_error.reshape(2, -1)
    assert np.all(theta[0] == bounds[0])
    assert np.all(theta[1] == bounds[1])
    assert np.all(np.isposinf(standard_error))


def test_multidimensional_ml_matches_r_and_flags_unbounded_factors():
    """Finite estimates match R; factors without a finite ML get the bound."""
    result = MLScorer().score(_parity_model(), _PARITY_PATTERNS)

    # R mirt 1.48 references for the finite estimates.
    assert_allclose(result.theta[0], [0.2184021, 0.1112702], atol=2e-5)
    assert_allclose(result.standard_error[0], [0.6624992, 0.5814070], atol=5e-4)
    assert_allclose(result.theta[4], [1.1641610, 0.1964072], atol=2e-5)
    assert_allclose(result.standard_error[4], [0.8932882, 0.5861332], atol=5e-4)
    # Every observed F1 item of rows 1 and 3 is wrong, and every F2 item of
    # row 3 right: those coordinates have no finite ML. R reports Inf/-Inf or
    # stops unconverged at its search bound. With simple structure, F2 of row
    # 1 is then the unidimensional ML of its own items.
    assert result.theta[1, 0] == -6.0 and np.isposinf(result.standard_error[1, 0])
    assert_allclose(result.theta[1, 1], -0.6279840, atol=2e-5)
    assert_allclose(result.standard_error[1, 1], 0.6395667, atol=5e-4)
    assert_array_equal(result.theta[2], [6.0, 6.0])
    assert_array_equal(result.theta[3], [-6.0, 6.0])
    assert np.all(np.isposinf(result.standard_error[2:4]))


def test_batched_multidimensional_ml_reaches_the_per_pattern_optimum(monkeypatch):
    """Built-in models share one Newton search and lose no likelihood."""
    from mirt import simdata
    from mirt.models import MultidimensionalModel

    rng = np.random.default_rng(3)
    model = MultidimensionalModel(12, n_factors=2)
    model.set_parameters(
        slopes=rng.uniform(0.4, 1.8, (12, 2)), intercepts=rng.normal(size=12)
    )
    model._is_fitted = True
    responses = np.asarray(simdata(model=model, n_persons=150, seed=4))
    responses[rng.random(responses.shape) < 0.1] = -1

    class PerPattern(MLScorer):
        pass

    reference = PerPattern().score(model, responses)
    calls = 0
    original = MLScorer._score_multidimensional

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MLScorer, "_score_multidimensional", counted)
    batched = MLScorer().score(model, responses)

    assert calls == 0
    batched_ll = model.log_likelihood(responses, batched.theta)
    reference_ll = model.log_likelihood(responses, reference.theta)
    assert np.all(batched_ll >= reference_ll - 1e-7)
    assert_allclose(batched.theta, reference.theta, atol=2e-3)
    assert_array_equal(
        np.isposinf(batched.standard_error), np.isposinf(reference.standard_error)
    )
