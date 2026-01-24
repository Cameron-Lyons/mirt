"""Tests for person fit statistics."""

import numpy as np
from numpy.testing import assert_allclose

from mirt.diagnostics.personfit import compute_personfit, flag_aberrant_persons


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
