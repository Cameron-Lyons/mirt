"""Tests for person fit statistics."""

from typing import get_args

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri

from mirt.constants import PROB_EPSILON
from mirt.diagnostics.personfit import (
    compute_personfit,
    compute_personfit_significance,
    flag_aberrant_persons,
)
from mirt.typing import PersonFitStatistic


class FixedProbabilityModel:
    """Minimal model returning fixed all-item probabilities."""

    def __init__(self, probabilities, n_categories=None):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.n_items = self.probabilities.shape[1]
        self._n_categories = n_categories
        self.probability_calls = 0

    @property
    def is_polytomous(self):
        return self._n_categories is not None

    @property
    def n_categories(self):
        if self._n_categories is None:
            raise AttributeError("dichotomous models do not have categories")
        return list(self._n_categories)

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        if len(theta) != len(self.probabilities):
            raise ValueError("theta length must match fixed probabilities")
        if item_idx is None:
            return self.probabilities.copy()
        if self.is_polytomous:
            return self.probabilities[
                :,
                item_idx,
                : self._n_categories[item_idx],
            ].copy()
        return self.probabilities[:, item_idx].copy()


def reference_zh(responses, probabilities, n_categories=None):
    """Scalar reference matching the person-fit log-likelihood definition."""
    n_persons, n_items = responses.shape
    valid = responses >= 0
    log_likelihood = np.zeros(n_persons)
    expected_log_likelihood = np.zeros(n_persons)
    variance_log_likelihood = np.zeros(n_persons)

    for item_idx in range(n_items):
        item_valid = valid[:, item_idx]
        item_responses = responses[:, item_idx]

        if probabilities.ndim == 2:
            probs = np.clip(
                probabilities[:, item_idx],
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            log_p = np.log(probs)
            log_q = np.log(1.0 - probs)
            observed = np.where(item_responses == 1, log_p, log_q)
            expected = probs * log_p + (1.0 - probs) * log_q
            variance = probs * (1.0 - probs) * (log_p - log_q) ** 2
        else:
            category_count = n_categories[item_idx]
            probs = np.clip(
                probabilities[:, item_idx, :category_count],
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            log_probs = np.log(probs)
            safe_responses = np.clip(
                np.where(item_valid, item_responses, 0),
                0,
                category_count - 1,
            )
            observed = log_probs[np.arange(n_persons), safe_responses]
            expected = np.sum(probs * log_probs, axis=1)
            variance = np.sum(probs * log_probs**2, axis=1) - expected**2

        log_likelihood += np.where(item_valid, observed, 0.0)
        expected_log_likelihood += np.where(item_valid, expected, 0.0)
        variance_log_likelihood += np.where(item_valid, variance, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (valid.sum(axis=1) >= 2) & (variance_log_likelihood > PROB_EPSILON),
            (log_likelihood - expected_log_likelihood)
            / np.sqrt(variance_log_likelihood),
            np.nan,
        )


class TestComputePersonfit:
    """Tests for compute_personfit function."""

    def test_basic_personfit(self, fitted_2pl_model, dichotomous_responses):
        """Test basic person fit computation."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in result
        assert "outfit" in result
        assert "Zh" in result
        assert len(result["infit"]) == dichotomous_responses["n_persons"]

    def test_personfit_values_positive_for_fit_stats(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that infit/outfit are positive."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert np.all(result["infit"] > 0)
        assert np.all(result["outfit"] > 0)

    def test_personfit_zh_distribution(self, fitted_2pl_model, dichotomous_responses):
        """Test that Zh follows approximately normal distribution."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        valid_zh = result["Zh"][~np.isnan(result["Zh"])]
        assert abs(np.mean(valid_zh)) < 1.5
        assert abs(np.std(valid_zh) - 1.0) < 1.0

    def test_personfit_statistics_subset(self, fitted_2pl_model, dichotomous_responses):
        """Test computing only subset of statistics."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
            statistics=["infit", "outfit"],
        )

        assert "infit" in result
        assert "outfit" in result
        assert "Zh" not in result

    def test_personfit_lz_statistic(self, fitted_2pl_model, dichotomous_responses):
        """Test lz statistic (same as Zh)."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
            statistics=["lz"],
        )

        assert "lz" in result
        valid_lz = result["lz"][~np.isnan(result["lz"])]
        assert len(valid_lz) > 0

    def test_personfit_default_statistics(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test default statistics are infit, outfit, Zh."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in result
        assert "outfit" in result
        assert "Zh" in result

    def test_reuses_one_probability_matrix_for_all_statistics(self):
        """Test mean-square and likelihood statistics share probabilities."""
        responses = np.tile(
            np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]),
            (3, 1),
        )
        responses[1, 1] = -1
        probabilities = np.column_stack(
            (
                np.linspace(0.2, 0.8, len(responses)),
                np.linspace(0.7, 0.3, len(responses)),
                np.linspace(0.4, 0.6, len(responses)),
            )
        )
        model = FixedProbabilityModel(probabilities)

        result = compute_personfit(model, responses, np.zeros(len(responses)))

        assert model.probability_calls == 1
        assert_allclose(
            result["Zh"],
            reference_zh(responses, probabilities),
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )

    def test_zh_only_uses_one_probability_matrix(self):
        """Test requesting only Zh avoids repeated item evaluations."""
        responses = np.tile(np.array([[0, 1], [1, 0]]), (5, 1))
        probabilities = np.tile(np.array([[0.3, 0.7]]), (len(responses), 1))
        model = FixedProbabilityModel(probabilities)

        result = compute_personfit(
            model,
            responses,
            np.zeros(len(responses)),
            statistics=["Zh"],
        )

        assert set(result) == {"Zh"}
        assert model.probability_calls == 1

    def test_lz_is_in_public_personfit_statistic_type(self):
        """Test runtime typing matches the documented lz option."""
        assert "lz" in get_args(PersonFitStatistic)

    def test_significance_reuses_hidden_zh_computation(self):
        """Test significance can accompany a mean-square-only request."""
        responses = np.tile(np.array([[0, 1], [1, 0]]), (5, 1))
        probabilities = np.tile(np.array([[0.3, 0.7]]), (len(responses), 1))
        model = FixedProbabilityModel(probabilities)

        result = compute_personfit(
            model,
            responses,
            np.zeros(len(responses)),
            statistics=["infit"],
            p_adjust="holm",
        )

        assert set(result) == {
            "infit",
            "p_value",
            "p_value_adjusted",
            "aberrant",
        }
        assert model.probability_calls == 1
        assert result["aberrant"].dtype == bool


class TestPersonfitSignificance:
    """Tests for calibrated person-fit decisions."""

    @pytest.mark.parametrize(
        ("alternative", "expected"),
        [
            ("lower", ndtr(np.array([-2.0, 0.0, 2.0]))),
            ("upper", ndtr(np.array([2.0, 0.0, -2.0]))),
            ("two-sided", 2.0 * ndtr(np.array([-2.0, 0.0, -2.0]))),
        ],
    )
    def test_tail_alternatives(self, alternative, expected):
        """Test one- and two-sided normal probabilities."""
        result = compute_personfit_significance(
            np.array([-2.0, 0.0, 2.0]),
            alternative=alternative,
        )

        assert_allclose(result["p_value"], expected)
        assert_allclose(result["p_value_adjusted"], expected)

    def test_holm_adjustment_excludes_missing_scores(self):
        """Test undefined scores do not increase the correction family."""
        raw = np.array([0.01, 0.03, 0.04, np.nan])

        result = compute_personfit_significance(
            ndtri(raw),
            alpha=0.05,
            p_adjust="holm",
        )

        assert_allclose(result["p_value"], raw, equal_nan=True)
        assert_allclose(
            result["p_value_adjusted"],
            np.array([0.03, 0.06, 0.06, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            result["aberrant"],
            np.array([True, False, False, False]),
        )

    def test_extreme_scores_use_stable_tail_probabilities(self):
        """Test large finite scores retain meaningful small probabilities."""
        result = compute_personfit_significance(
            np.array([-10.0, 10.0]),
            alternative="two-sided",
        )

        assert np.all(result["p_value"] > 0.0)
        assert_allclose(result["p_value"][0], result["p_value"][1])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": True}, "alpha"),
            ({"alternative": "middle"}, "alternative"),
            ({"p_adjust": "unknown"}, "p_adjust"),
        ],
    )
    def test_invalid_options_are_rejected(self, kwargs, message):
        """Test significance options fail with focused messages."""
        with pytest.raises(ValueError, match=message):
            compute_personfit_significance(np.array([0.0]), **kwargs)

    @pytest.mark.parametrize("zh", [1.0, [[0.0]], [1.0 + 2.0j]])
    def test_invalid_scores_are_rejected(self, zh):
        """Test scores must be a real one-dimensional vector."""
        with pytest.raises(ValueError, match="zh"):
            compute_personfit_significance(zh)


class TestFlagAberrantPersons:
    """Tests for flag_aberrant_persons function."""

    def test_basic_flagging(self, fitted_2pl_model, dichotomous_responses):
        """Test basic aberrant person flagging."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        fit_stats = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        flags = flag_aberrant_persons(fit_stats)

        assert len(flags) == dichotomous_responses["n_persons"]
        assert flags.dtype == bool

    def test_flagging_with_custom_criteria(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test flagging with custom criteria."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        fit_stats = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        strict_flags = flag_aberrant_persons(
            fit_stats,
            criteria={
                "infit": (0.8, 1.2),
                "outfit": (0.8, 1.2),
            },
        )

        lenient_flags = flag_aberrant_persons(
            fit_stats,
            criteria={
                "infit": (0.3, 2.0),
                "outfit": (0.3, 2.0),
            },
        )

        assert np.sum(strict_flags) >= np.sum(lenient_flags)

    def test_flagging_no_flags_with_wide_criteria(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that very wide criteria flag no one."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        fit_stats = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        flags = flag_aberrant_persons(
            fit_stats,
            criteria={
                "infit": (-100, 100),
                "outfit": (-100, 100),
                "Zh": (-100, 100),
            },
        )

        assert np.sum(flags) == 0

    def test_flagging_default_criteria(self, fitted_2pl_model, dichotomous_responses):
        """Test that default criteria are reasonable."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        fit_stats = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        flags = flag_aberrant_persons(fit_stats)

        flag_rate = np.mean(flags)
        assert flag_rate < 0.5

    def test_default_criteria_support_lz_alias(self):
        """Test lz-only results use the documented likelihood thresholds."""
        flags = flag_aberrant_persons({"lz": np.array([-2.5, 0.0, 2.5])})

        np.testing.assert_array_equal(flags, np.array([True, False, True]))

    def test_empty_statistics_are_rejected(self):
        """Test empty inputs fail with a clear message."""
        with pytest.raises(ValueError, match="at least one"):
            flag_aberrant_persons({})


class TestPersonfitEdgeCases:
    """Tests for edge cases in person fit computation."""

    def test_perfect_score(self, fitted_2pl_model, dichotomous_responses):
        """Test person fit for perfect score."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 1

        from mirt.scoring import fscores

        scores = fscores(model, responses, method="EAP")

        result = compute_personfit(model, responses, theta=scores.theta)

        assert np.isfinite(result["infit"][0]) or np.isnan(result["infit"][0])

    def test_zero_score(self, fitted_2pl_model, dichotomous_responses):
        """Test person fit for zero score."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"].copy()
        responses[0, :] = 0

        from mirt.scoring import fscores

        scores = fscores(model, responses, method="EAP")

        result = compute_personfit(model, responses, theta=scores.theta)

        assert np.isfinite(result["infit"][0]) or np.isnan(result["infit"][0])

    def test_theta_shape_handling(self, fitted_2pl_model, dichotomous_responses):
        """Test that 1D theta is handled correctly."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        theta_1d = scores.theta.ravel()

        result = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=theta_1d,
        )

        assert len(result["infit"]) == dichotomous_responses["n_persons"]

    def test_consistency(self, fitted_2pl_model, dichotomous_responses):
        """Test that results are consistent across calls."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result1 = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )
        result2 = compute_personfit(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert_allclose(result1["infit"], result2["infit"])
        assert_allclose(result1["outfit"], result2["outfit"])


class TestPersonfitPolytomous:
    """Tests for person fit with polytomous models."""

    def test_personfit_polytomous(self, polytomous_responses):
        """Test person fit with polytomous model."""
        from mirt import fit_mirt
        from mirt.scoring import fscores

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        scores = fscores(result.model, polytomous_responses["responses"], method="EAP")

        fit_result = compute_personfit(
            result.model,
            polytomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in fit_result
        assert len(fit_result["infit"]) == polytomous_responses["n_persons"]

    def test_polytomous_zh_matches_scalar_reference(self):
        """Test chunked polytomous Zh with different category counts."""
        responses = np.tile(
            np.array([[0, 0], [1, 1], [2, 1], [1, 0]]),
            (3, 1),
        )
        item_probabilities = np.array(
            [
                [[0.6, 0.3, 0.1], [0.7, 0.3, 0.0]],
            ]
        )
        probabilities = np.tile(item_probabilities, (len(responses), 1, 1))
        model = FixedProbabilityModel(probabilities, n_categories=[3, 2])

        result = compute_personfit(
            model,
            responses,
            np.zeros(len(responses)),
            statistics=["Zh", "lz"],
        )
        expected = reference_zh(responses, probabilities, [3, 2])

        assert model.probability_calls == 1
        assert_allclose(
            result["Zh"],
            expected,
            rtol=1e-13,
            atol=1e-13,
            equal_nan=True,
        )
        assert_allclose(result["lz"], result["Zh"], equal_nan=True)
