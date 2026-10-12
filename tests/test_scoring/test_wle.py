"""Tests for WLE scoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.scoring.wle import WLEScorer


class TestWLEScorerInitialization:
    """Tests for WLEScorer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        scorer = WLEScorer()

        assert scorer.bounds == (-6.0, 6.0)
        assert scorer.tol == 1e-6

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        scorer = WLEScorer(bounds=(-4.0, 4.0))

        assert scorer.bounds == (-4.0, 4.0)

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        scorer = WLEScorer(tol=1e-8)

        assert scorer.tol == 1e-8

    def test_repr(self):
        """Test __repr__ method."""
        scorer = WLEScorer(bounds=(-5.0, 5.0), tol=1e-5)
        repr_str = repr(scorer)

        assert "WLEScorer" in repr_str
        assert "-5" in repr_str


class TestWLEScorerScoring:
    """Tests for WLEScorer scoring."""

    def test_information_uses_vectorized_dichotomous_path(
        self, fitted_2pl_model, monkeypatch
    ):
        """Test that dichotomous information is computed in one bulk call."""
        model = fitted_2pl_model.model
        original_information = model.information
        item_indices = []

        def record_information(theta, item_idx=None):
            item_indices.append(item_idx)
            return original_information(theta, item_idx)

        monkeypatch.setattr(model, "information", record_information)

        mask = np.zeros(model.n_items, dtype=bool)
        mask[:2] = True
        WLEScorer()._test_information(model, np.array([[0.0]]), mask)

        assert item_indices == [None]

    def test_information_uses_observed_polytomous_items(self):
        """Test itemwise fallback for sparse polytomous responses."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=3, n_categories=4)
        theta = np.array([[-0.5], [0.0], [0.5]])
        observed_mask = np.array([True, False, True])

        actual = WLEScorer()._test_information(model, theta, observed_mask)
        expected = model.information(theta, 0) + model.information(theta, 2)

        assert_allclose(actual, expected)

    def test_basic_scoring(self, fitted_2pl_model, dichotomous_responses):
        """Test basic WLE scoring."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert result.method == "WLE"
        assert result.theta.shape == (dichotomous_responses["n_persons"],)
        assert result.standard_error.shape == (dichotomous_responses["n_persons"],)

    def test_se_positive_or_inf(self, fitted_2pl_model, dichotomous_responses):
        """Test that standard errors are positive or infinite."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.standard_error > 0)

    def test_theta_within_bounds(self, fitted_2pl_model, dichotomous_responses):
        """Test that theta estimates are within bounds."""
        model = fitted_2pl_model.model
        bounds = (-4.0, 4.0)
        scorer = WLEScorer(bounds=bounds)

        result = scorer.score(model, dichotomous_responses["responses"])

        assert np.all(result.theta >= bounds[0])
        assert np.all(result.theta <= bounds[1])

    def test_unfitted_model_raises_error(self, dichotomous_responses):
        """Test that unfitted model raises error."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=dichotomous_responses["n_items"])
        scorer = WLEScorer()

        with pytest.raises(ValueError, match="fitted"):
            scorer.score(model, dichotomous_responses["responses"])


class TestWLEBiasReduction:
    """Tests for WLE bias reduction properties."""

    def test_wle_less_biased_than_ml_at_extremes(self, fitted_2pl_model):
        """Test that WLE is less biased than ML at extreme abilities."""
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        rng = np.random.default_rng(42)
        true_theta = 2.5
        n_replications = 50

        wle_estimates = []
        ml_estimates = []

        for _ in range(n_replications):
            n_items = model.n_items
            probs = 1 / (1 + np.exp(-1.0 * (true_theta - np.zeros(n_items))))
            responses = (rng.random(n_items) < probs).astype(int)

            wle_scorer = WLEScorer()
            ml_scorer = MLScorer()

            wle_result = wle_scorer.score(model, responses[None, :])
            ml_result = ml_scorer.score(model, responses[None, :])

            wle_estimates.append(wle_result.theta[0])
            ml_estimates.append(ml_result.theta[0])

        wle_bias = abs(np.mean(wle_estimates) - true_theta)
        ml_bias = abs(np.mean(ml_estimates) - true_theta)

        assert wle_bias <= ml_bias + 0.3

    def test_wle_ml_similar_for_moderate_theta(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that WLE and ML are similar for moderate theta."""
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        n_items = responses.shape[1]
        moderate = (sum_scores > n_items * 0.3) & (sum_scores < n_items * 0.7)

        if np.sum(moderate) < 5:
            pytest.skip("Not enough moderate scores")

        wle_scorer = WLEScorer()
        ml_scorer = MLScorer()

        wle_result = wle_scorer.score(model, responses)
        ml_result = ml_scorer.score(model, responses)

        correlation = np.corrcoef(
            wle_result.theta[moderate], ml_result.theta[moderate]
        )[0, 1]

        assert correlation > 0.9


class TestWLEScorerMultidimensional:
    """Tests for WLE with multidimensional models."""

    def test_multidimensional_scoring(self):
        """Test multidimensional WLE scoring."""
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

        scorer = WLEScorer()
        scores = scorer.score(result.model, responses)

        assert scores.theta.shape == (n_persons, 2)
        assert scores.standard_error.shape == (n_persons, 2)


def _simple_structure_model():
    """Two-factor simple structure with R mirt 1.48 WLE references."""
    from mirt.models import MultidimensionalModel

    slopes = np.zeros((10, 2))
    slopes[:5, 0] = [0.85, 1.45, 1.19, 0.83, 2.30]
    slopes[5:, 1] = [1.54, 2.09, 1.93, 1.21, 1.29]
    model = MultidimensionalModel(10, n_factors=2)
    model.set_parameters(
        slopes=slopes,
        intercepts=np.array(
            [0.39, -1.2, 0.35, -0.97, 0.11, 0.55, 0.2, -1.19, 0.78, 0.67]
        ),
    )
    model._is_fitted = True
    return model


_SIMPLE_STRUCTURE_PATTERNS = np.array(
    [
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [-1, -1, -1, 0, 0, 0, 0, 0, 1, 1],
        [1] * 10,
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    ]
)


class TestWLEMultidimensionalWeighting:
    """Multidimensional WLE weights by the information determinant, as R."""

    def test_simple_structure_matches_r(self):
        """Extreme patterns stay finite and every estimate matches R mirt."""
        result = WLEScorer().score(
            _simple_structure_model(), _SIMPLE_STRUCTURE_PATTERNS
        )

        # fscores(method="WLE", max_theta=6) in R mirt 1.48.
        expected = np.array(
            [
                [0.1690841, 0.09072772],
                [-0.6305900, -0.5222381],
                [2.4408900, 1.4488850],
                [-1.7997700, 1.4488840],
                [0.8988125, 0.1623090],
            ]
        )
        assert_allclose(result.theta, expected, atol=1e-5)

    def test_standard_errors_invert_the_information_matrix(self):
        from mirt.scoring._common import observed_information_matrix

        model = _simple_structure_model()
        result = WLEScorer().score(model, _SIMPLE_STRUCTURE_PATTERNS)

        information = observed_information_matrix(
            model, result.theta, _SIMPLE_STRUCTURE_PATTERNS >= 0
        )
        expected = np.sqrt(np.diagonal(np.linalg.inv(information), axis1=1, axis2=2))
        assert_allclose(result.standard_error, expected, rtol=1e-10)

    def test_bifactor_matches_r(self):
        from mirt.models import BifactorModel

        model = BifactorModel(8, specific_factors=[0, 0, 0, 0, 1, 1, 1, 1])
        model.set_parameters(
            general_loadings=np.array([1.2, 0.9, 1.5, 1.1, 1.3, 0.8, 1.6, 1.0]),
            specific_loadings=np.array([0.7, 0.9, 0.6, 1.0, 0.8, 1.1, 0.5, 0.9]),
            intercepts=np.array([0.3, -0.4, 0.8, -0.2, 0.1, 0.5, -0.6, 0.2]),
        )
        model._is_fitted = True
        patterns = np.array(
            [[1] * 8, [0] * 8, [1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1, 0, 1]]
        )

        result = WLEScorer().score(model, patterns)

        expected = np.array(
            [
                [1.1779020, 0.9209101, 0.3146946],
                [-1.0242400, -0.9784816, -1.0277430],
                [-0.5652444, 1.5377060, -0.08083153],
                [-0.8976820, 1.0729490, 2.5818720],
            ]
        )
        assert_allclose(result.theta, expected, atol=1e-5)

    def test_batched_search_matches_per_pattern_optimizer(self, monkeypatch):
        from mirt import simdata

        model = _simple_structure_model()
        responses = np.asarray(simdata(model=model, n_persons=120, seed=8))
        responses[::7, :5] = -1

        class PerPattern(WLEScorer):
            pass

        reference = PerPattern().score(model, responses)
        calls = 0
        original = WLEScorer._estimate_multidimensional_person

        def counted(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(WLEScorer, "_estimate_multidimensional_person", counted)
        batched = WLEScorer().score(model, responses)

        assert calls == 0

        from mirt.scoring._common import observed_information_matrix

        def weighted_log_likelihood(theta):
            information = observed_information_matrix(model, theta, responses >= 0)
            return (
                model.log_likelihood(responses, theta)
                + 0.5 * np.linalg.slogdet(information)[1]
            )

        # L-BFGS-B stops on a looser tolerance than the Newton search.
        assert np.all(
            weighted_log_likelihood(batched.theta)
            >= weighted_log_likelihood(reference.theta) - 1e-9
        )
        assert_allclose(batched.theta, reference.theta, atol=2e-3)
        assert_allclose(batched.standard_error, reference.standard_error, atol=2e-3)

    def test_unmeasured_factors_and_singular_information(self):
        """Unmeasured factors are zero with infinite SE; singular rows are NaN."""
        model = _simple_structure_model()
        slopes = model.parameters["slopes"].copy()
        slopes[0] = [0.85, 1.7]  # Item 0 loads on both factors.
        model.set_parameters(slopes=slopes)
        patterns = np.array(
            [
                [1, 0, 1, 1, 0, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )

        for scorer in (WLEScorer(), type("PerPattern", (WLEScorer,), {})()):
            result = scorer.score(model, patterns)
            # Row 0 measures F1 through items 0-4 and F2 only through item 0.
            assert np.all(np.isfinite(result.theta[0]))
            # One item cannot identify two factors: no WLE exists.
            assert_allclose(result.theta[1], 0.0)
            assert np.all(np.isnan(result.standard_error[1]))

        partial = WLEScorer().score(_simple_structure_model(), patterns)
        assert partial.theta[0, 1] == 0.0
        assert np.isposinf(partial.standard_error[0, 1])
        assert np.isfinite(partial.standard_error[0, 0])


class TestWLEScorerConsistency:
    """Tests for WLE scoring consistency."""

    def test_reproducibility(self, fitted_2pl_model, dichotomous_responses):
        """Test that scoring is reproducible."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result1 = scorer.score(model, dichotomous_responses["responses"])
        result2 = scorer.score(model, dichotomous_responses["responses"])

        assert_allclose(result1.theta, result2.theta)

    def test_correlation_with_sum_score(self, fitted_2pl_model, dichotomous_responses):
        """Test correlation with sum score."""
        model = fitted_2pl_model.model
        scorer = WLEScorer()

        result = scorer.score(model, dichotomous_responses["responses"])

        sum_scores = dichotomous_responses["responses"].sum(axis=1)
        correlation = np.corrcoef(result.theta, sum_scores)[0, 1]

        assert correlation > 0.7


class TestWLEScorerPolytomous:
    """Tests for WLE with polytomous models."""

    def test_polytomous_scoring(self, polytomous_responses):
        """Test WLE scoring with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scorer = WLEScorer()
        scores = scorer.score(result.model, polytomous_responses["responses"])

        assert scores.theta.shape == (polytomous_responses["n_persons"],)


class TestWLEVsOtherMethods:
    """Tests comparing WLE with other scoring methods."""

    def test_wle_eap_correlation(self, fitted_2pl_model, dichotomous_responses):
        """Test that WLE and EAP estimates are highly correlated."""
        from mirt.scoring.eap import EAPScorer

        model = fitted_2pl_model.model
        wle_scorer = WLEScorer()
        eap_scorer = EAPScorer()

        wle_result = wle_scorer.score(model, dichotomous_responses["responses"])
        eap_result = eap_scorer.score(model, dichotomous_responses["responses"])

        correlation = np.corrcoef(wle_result.theta, eap_result.theta)[0, 1]

        assert correlation > 0.8

    def test_wle_between_ml_and_eap_variance(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that WLE variance is between ML and EAP."""
        from mirt.scoring.eap import EAPScorer
        from mirt.scoring.ml import MLScorer

        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        sum_scores = responses.sum(axis=1)
        not_extreme = (sum_scores > 0) & (sum_scores < responses.shape[1])

        if np.sum(not_extreme) < 10:
            pytest.skip("Not enough non-extreme scores")

        wle_scorer = WLEScorer()
        ml_scorer = MLScorer()
        eap_scorer = EAPScorer()

        wle_result = wle_scorer.score(model, responses)
        ml_result = ml_scorer.score(model, responses)
        eap_result = eap_scorer.score(model, responses)

        wle_var = np.var(wle_result.theta[not_extreme])
        ml_var = np.var(ml_result.theta[not_extreme])
        eap_var = np.var(eap_result.theta[not_extreme])

        assert min(eap_var, ml_var) * 0.7 <= wle_var <= max(eap_var, ml_var) * 1.3


class TestWLEScorerEmptyResponses:
    """Tests for WLE with empty/missing responses."""

    def test_all_missing_returns_zero(self, fitted_2pl_model):
        """Test that all missing returns zero theta."""
        model = fitted_2pl_model.model
        responses = np.full((1, model.n_items), -1, dtype=int)

        scorer = WLEScorer()
        result = scorer.score(model, responses)

        assert result.theta[0] == 0.0
        assert np.isinf(result.standard_error[0])


def _mixed_discrimination_bank():
    """A 2PL bank of weak and strong items whose WL can be bimodal."""
    from mirt.models.dichotomous import TwoParameterLogistic

    rng = np.random.default_rng(3)
    discrimination = np.r_[rng.uniform(0.2, 0.6, 10), rng.uniform(3, 6, 10)]
    difficulty = np.r_[rng.normal(0, 2, 10), rng.uniform(-4, 4, 10)]
    model = TwoParameterLogistic(20)
    model.set_parameters(discrimination=discrimination, difficulty=difficulty)
    model._is_fitted = True
    responses = model.simulate(rng.normal(0, 1.5, size=(300, 1)), seed=3)
    responses = responses.astype(np.int64)
    responses[rng.random(responses.shape) < 0.3] = -1
    return model, responses


def _weighted_log_likelihood(model, responses, theta):
    """Return log L + 0.5 log I of each row at each column of ``theta``."""
    a = model.parameters["discrimination"]
    b = model.parameters["difficulty"]
    logits = a * (np.asarray(theta)[..., None] - b)
    observed = (responses >= 0)[:, None, :]
    correct = (responses == 1)[:, None, :]
    log_likelihood = np.sum(
        np.where(
            correct,
            -np.logaddexp(0, -logits),
            np.where(observed, -np.logaddexp(0, logits), 0.0),
        ),
        axis=-1,
    )
    p = 1.0 / (1.0 + np.exp(-logits))
    information = np.sum(observed * a**2 * p * (1 - p), axis=-1)
    return log_likelihood + 0.5 * np.log(information)


@pytest.fixture(scope="module")
def mixed_bank():
    model, responses = _mixed_discrimination_bank()
    observed = responses >= 0
    correct = (responses == 1).sum(axis=1)
    informative = (correct > 0) & (correct < observed.sum(axis=1))
    grid = np.linspace(-6, 6, 4801)
    best = np.max(_weighted_log_likelihood(model, responses, grid[None, :]), axis=1)
    return model, responses, informative, best


class TestWLEGlobalMaximum:
    """The unidimensional WLE is the global maximum of the weighted likelihood."""

    @staticmethod
    def _assert_global(bank, theta):
        model, responses, informative, best = bank
        attained = _weighted_log_likelihood(model, responses, theta[:, None])[:, 0]
        gap = (best - attained)[informative]
        assert np.max(gap) < 1e-6

    def test_batched_2pl_search_finds_global_maximum(self, mixed_bank):
        # Regression: 19% of these patterns stopped on a lower local maximum.
        theta = WLEScorer().score(mixed_bank[0], mixed_bank[1]).theta.ravel()
        self._assert_global(mixed_bank, theta)

    def test_generic_batched_search_finds_global_maximum(self, mixed_bank):
        from mirt.scoring._common import unique_response_patterns

        model, responses = mixed_bank[0], mixed_bank[1]
        patterns, inverse = unique_response_patterns(responses)
        theta, _ = WLEScorer()._estimate_unidimensional_batch(model, patterns)
        self._assert_global(mixed_bank, theta[inverse])

    def test_per_pattern_search_finds_global_maximum(self, mixed_bank):
        import copy

        model = copy.deepcopy(mixed_bank[0])
        model.probability = model.probability  # forces the per-pattern path
        theta = WLEScorer().score(model, mixed_bank[1]).theta.ravel()
        self._assert_global(mixed_bank, theta)

    def test_bimodal_pattern_matches_r(self):
        # R mirt 1.48 fscores(method="WLE", theta_lim=c(-6, 6)) on the same
        # parameters; the local search used to return -2.912.
        model, _ = _mixed_discrimination_bank()
        pattern = np.array(
            [[1, -1, 0, 1, 1, -1, -1, -1, -1, 1, -1, 1, 0, 0, -1, 0, -1, 0, 0, -1]]
        )
        theta = WLEScorer().score(model, pattern).theta.ravel()
        assert_allclose(theta, [0.825314647724048], atol=1e-4)


def test_2pl_wle_honours_instance_hook_overrides():
    # Regression: an exact 2PL went to a native kernel even when its
    # probability curves were overridden on the instance.
    import types

    from mirt.models.dichotomous import TwoParameterLogistic

    model = TwoParameterLogistic(10)
    model.set_parameters(discrimination=np.ones(10), difficulty=np.zeros(10))
    model._is_fitted = True
    pattern = np.array([[1, 1, 1, 0, 0, 0, 1, 0, 1, 0]])
    base = WLEScorer().score(model, pattern).theta.ravel()[0]

    original = model.probability

    def shifted(self, theta, item_idx=None):
        return original(np.asarray(theta) - 2.0, item_idx)

    model.probability = types.MethodType(shifted, model)
    moved = WLEScorer().score(model, pattern).theta.ravel()[0]

    assert abs(base) < 1e-4
    assert moved == pytest.approx(base + 2.0, abs=0.2)


def test_2pl_batch_matches_the_per_pattern_search_with_missing_codes():
    # The batched 2PL search replaced a native kernel; it must agree with the
    # per-pattern search for every missing code and an all-missing pattern.
    from mirt.models.dichotomous import TwoParameterLogistic

    responses = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, -1, 1, 0],
            [1, -8, -1, 0, 1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    model = TwoParameterLogistic(n_items=5)
    model.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.6, 0.9, 1.3]),
        difficulty=np.array([-1.2, -0.4, 0.1, 0.8, 1.5]),
    )
    model._is_fitted = True
    scorer = WLEScorer(bounds=(-4.5, 5.0), tol=1e-8)
    expected = np.array(
        [
            scorer._estimate_person(model, row)
            for row in np.where(responses >= 0, responses, -1)
        ]
    )

    result = scorer.score(model, responses)

    assert_allclose(result.theta.ravel(), expected[:, 0], atol=1e-7)
    assert_allclose(result.standard_error.ravel(), expected[:, 1], atol=1e-7)
    assert result.theta.ravel()[-1] == 0.0
    assert np.isinf(result.standard_error.ravel()[-1])
