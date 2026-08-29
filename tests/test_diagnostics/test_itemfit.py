"""Tests for item fit statistics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt.diagnostics.itemfit as itemfit_module
from mirt.constants import PROB_CLIP_MAX, PROB_CLIP_MIN
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2


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
        return self.probabilities[:, item_idx].copy()


def reference_s_x2(responses, expected, score_scales, n_groups):
    """Scalar reference for grouped S-X2 aggregation."""
    valid = responses >= 0
    sum_scores = np.sum(np.where(valid, responses, 0), axis=1)
    score_cuts = np.percentile(
        sum_scores,
        np.linspace(0.0, 100.0, n_groups + 1),
    )
    statistics = np.zeros(responses.shape[1])
    degrees = np.zeros(responses.shape[1])

    for item_idx in range(responses.shape[1]):
        for group_idx in range(n_groups):
            if group_idx < n_groups - 1:
                in_group = (sum_scores >= score_cuts[group_idx]) & (
                    sum_scores < score_cuts[group_idx + 1]
                )
            else:
                in_group = sum_scores >= score_cuts[group_idx]

            group_valid = in_group & valid[:, item_idx]
            count = np.count_nonzero(group_valid)
            if count < 5:
                continue

            observed_mean = np.mean(
                responses[group_valid, item_idx] / score_scales[item_idx]
            )
            expected_mean = np.clip(
                np.mean(expected[group_valid, item_idx] / score_scales[item_idx]),
                PROB_CLIP_MIN,
                PROB_CLIP_MAX,
            )
            statistics[item_idx] += (
                count
                * (observed_mean - expected_mean) ** 2
                / (expected_mean * (1.0 - expected_mean))
            )
            degrees[item_idx] += 1

    return statistics, np.maximum(degrees - 1, 1)


class TestComputeItemfit:
    """Tests for compute_itemfit function."""

    def test_basic_itemfit(self, fitted_2pl_model, dichotomous_responses):
        """Test basic item fit computation."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result
        assert len(result["infit"]) == dichotomous_responses["n_items"]
        assert len(result["outfit"]) == dichotomous_responses["n_items"]

    def test_itemfit_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that fit statistics are positive."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] > 0)
        assert np.all(result["outfit"] > 0)

    def test_itemfit_values_reasonable_range(
        self, fitted_2pl_model, dichotomous_responses
    ):
        """Test that fit statistics are in reasonable range."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert np.all(result["infit"] < 3.0)
        assert np.all(result["outfit"] < 3.0)

    def test_itemfit_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test item fit with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "infit" in result
        assert "outfit" in result

    def test_itemfit_statistics_subset(self, fitted_2pl_model, dichotomous_responses):
        """Test computing only subset of statistics."""
        model = fitted_2pl_model.model
        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            statistics=["infit"],
        )

        assert "infit" in result
        assert "outfit" not in result

    def test_itemfit_no_responses_raises_error(self, fitted_2pl_model):
        """Test that missing responses raises error."""
        model = fitted_2pl_model.model
        with pytest.raises(ValueError, match="responses required"):
            compute_itemfit(model, responses=None)

    def test_itemfit_default_statistics(self, fitted_2pl_model, dichotomous_responses):
        """Test default statistics are infit and outfit."""
        model = fitted_2pl_model.model
        result = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert "infit" in result
        assert "outfit" in result

    def test_itemfit_supports_s_x2(self, fitted_2pl_model, dichotomous_responses):
        """Test the documented S-X2 statistic through the item-fit API."""
        model = fitted_2pl_model.model

        result = compute_itemfit(
            model,
            responses=dichotomous_responses["responses"],
            statistics=["S_X2"],
            n_groups=5,
        )

        assert set(result) == {"S_X2", "df", "p_value"}
        assert len(result["S_X2"]) == dichotomous_responses["n_items"]

    def test_public_itemfit_forwards_s_x2_group_count(
        self,
        fitted_2pl_model,
        dichotomous_responses,
    ):
        """Test the top-level interface exposes S-X2 grouping controls."""
        from mirt import itemfit

        result = itemfit(
            fitted_2pl_model,
            dichotomous_responses["responses"],
            statistics=["S_X2"],
            n_groups=5,
        )

        assert {"S_X2", "df", "p_value"}.issubset(result.columns)

    def test_itemfit_combines_mean_square_and_s_x2_statistics(self):
        """Test all item-fit families share one probability evaluation."""
        responses = np.tile(
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            (5, 1),
        )
        probabilities = np.tile(np.array([[0.3, 0.7]]), (len(responses), 1))
        model = FixedProbabilityModel(probabilities)

        result = compute_itemfit(
            model,
            responses,
            statistics=["infit", "outfit", "S_X2"],
            theta=np.zeros(len(responses)),
            n_groups=2,
        )

        assert set(result) == {"infit", "outfit", "S_X2", "df", "p_value"}
        assert model.probability_calls == 1


class TestComputeSX2:
    """Tests for compute_s_x2 function."""

    def test_basic_s_x2(self, fitted_2pl_model, dichotomous_responses):
        """Test basic S-X2 computation."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert "S_X2" in result
        assert "df" in result
        assert "p_value" in result
        assert len(result["S_X2"]) == dichotomous_responses["n_items"]

    def test_s_x2_values_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that S-X2 values are non-negative."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["S_X2"] >= 0)

    def test_s_x2_df_positive(self, fitted_2pl_model, dichotomous_responses):
        """Test that degrees of freedom are positive."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["df"] >= 1)

    def test_s_x2_p_values_in_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that p-values are in [0, 1]."""
        model = fitted_2pl_model.model
        result = compute_s_x2(model, dichotomous_responses["responses"])

        assert np.all(result["p_value"] >= 0)
        assert np.all(result["p_value"] <= 1)

    def test_s_x2_with_theta(self, fitted_2pl_model, dichotomous_responses):
        """Test S-X2 with provided theta."""
        from mirt.scoring import fscores

        model = fitted_2pl_model.model
        scores = fscores(model, dichotomous_responses["responses"], method="EAP")

        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
            theta=scores.theta,
        )

        assert "S_X2" in result

    def test_s_x2_custom_n_groups(self, fitted_2pl_model, dichotomous_responses):
        """Test S-X2 with custom number of groups."""
        model = fitted_2pl_model.model
        result = compute_s_x2(
            model,
            dichotomous_responses["responses"],
            n_groups=5,
        )

        assert "S_X2" in result

    def test_s_x2_matches_grouped_dichotomous_reference(self, monkeypatch):
        """Test chunked grouped aggregation with missing responses."""
        monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", 3)
        responses = np.tile(
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            (5, 1),
        )
        responses[1, 0] = -1
        probabilities = np.column_stack(
            (
                np.linspace(0.2, 0.8, len(responses)),
                np.linspace(0.7, 0.3, len(responses)),
            )
        )
        model = FixedProbabilityModel(probabilities)
        expected_statistic, expected_df = reference_s_x2(
            responses,
            probabilities,
            np.ones(2),
            n_groups=2,
        )

        result = compute_s_x2(
            model,
            responses,
            theta=np.zeros(len(responses)),
            n_groups=2,
        )

        assert_allclose(result["S_X2"], expected_statistic)
        assert_allclose(result["df"], expected_df)
        assert model.probability_calls == 1

    def test_s_x2_normalizes_polytomous_observed_scores(self, monkeypatch):
        """Test chunked observed and expected scores use the same scale."""
        monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", 3)
        responses = np.tile(
            np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]]),
            (4, 1),
        )
        item_probabilities = np.array(
            [
                [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            ]
        )
        probabilities = np.tile(item_probabilities, (len(responses), 1, 1))
        model = FixedProbabilityModel(probabilities, n_categories=[3, 3])
        categories = np.arange(3)
        expected = probabilities @ categories
        expected_statistic, expected_df = reference_s_x2(
            responses,
            expected,
            np.array([2.0, 2.0]),
            n_groups=2,
        )

        result = compute_s_x2(
            model,
            responses,
            theta=np.zeros(len(responses)),
            n_groups=2,
        )

        assert_allclose(result["S_X2"], expected_statistic)
        assert_allclose(result["df"], expected_df)
        assert np.all((result["p_value"] >= 0.0) & (result["p_value"] <= 1.0))

    @pytest.mark.parametrize("n_groups", [True, 1, 0, -1, 2.5])
    def test_s_x2_rejects_invalid_group_count(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        n_groups,
    ):
        """Test score-group validation fails clearly and consistently."""
        with pytest.raises(ValueError, match="n_groups"):
            compute_s_x2(
                fitted_2pl_model.model,
                dichotomous_responses["responses"],
                theta=np.zeros(dichotomous_responses["n_persons"]),
                n_groups=n_groups,
            )


class TestItemfitWithPolytomousModel:
    """Tests for item fit with polytomous models."""

    def test_itemfit_polytomous(self, polytomous_responses):
        """Test item fit with polytomous model."""
        from mirt import fit_mirt

        result = fit_mirt(
            polytomous_responses["responses"],
            model="GRM",
            max_iter=15,
            n_quadpts=11,
        )

        fit_result = compute_itemfit(
            result.model, responses=polytomous_responses["responses"]
        )

        assert "infit" in fit_result
        assert "outfit" in fit_result
        assert len(fit_result["infit"]) == polytomous_responses["n_items"]


class TestItemfitEdgeCases:
    """Tests for edge cases in item fit computation."""

    def test_itemfit_perfect_fit(self, fitted_2pl_model):
        """Test item fit when responses match model expectations perfectly."""
        model = fitted_2pl_model.model
        n_persons = 30
        responses = np.zeros((n_persons, model.n_items), dtype=int)

        from mirt.scoring import fscores

        theta = fscores(model, responses, method="EAP").theta

        result = compute_itemfit(model, responses=responses, theta=theta)

        assert np.all(np.isfinite(result["infit"]))
        assert np.all(np.isfinite(result["outfit"]))

    def test_itemfit_consistency(self, fitted_2pl_model, dichotomous_responses):
        """Test that item fit is consistent across calls."""
        model = fitted_2pl_model.model
        result1 = compute_itemfit(model, responses=dichotomous_responses["responses"])
        result2 = compute_itemfit(model, responses=dichotomous_responses["responses"])

        assert_allclose(result1["infit"], result2["infit"])
        assert_allclose(result1["outfit"], result2["outfit"])
