"""Tests for plausible value generation."""

import numpy as np
import pytest

from mirt import (
    combine_plausible_values,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)
from mirt.utils.plausible import _inverse_cdf_rows


class TestGeneratePlausibleValues:
    """Tests for plausible value generation."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_plausible": 0}, "n_plausible"),
            ({"n_plausible": 1.5}, "n_plausible"),
            ({"n_quadpts": 0}, "n_quadpts"),
            ({"method": "mcmc", "n_iter": 0}, "n_iter"),
            ({"method": "mcmc", "burn_in": -1}, "burn_in"),
            ({"method": "mcmc", "proposal_scale": 0}, "proposal_scale"),
            ({"method": "mcmc", "proposal_scale": np.inf}, "proposal_scale"),
            ({"method": "posterior", "chunk_size": 0}, "chunk_size"),
            ({"method": "mcmc", "chunk_size": 0}, "chunk_size"),
        ],
    )
    def test_generation_parameters_are_validated(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        kwargs,
        message,
    ):
        """Reject invalid sample, quadrature, and iteration counts."""
        with pytest.raises(ValueError, match=message):
            generate_plausible_values(
                fitted_2pl_model,
                dichotomous_responses["responses"],
                **kwargs,
            )

    def test_requires_fitted_model(self):
        """Plausible values require estimated item parameters."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=2)

        with pytest.raises(ValueError, match="fitted"):
            generate_plausible_values(model, np.array([[0, 1]]))

    def test_validates_response_shape_and_codes(self, fitted_2pl_model):
        """Reject malformed and non-dichotomous response matrices."""
        with pytest.raises(ValueError, match="2D"):
            generate_plausible_values(fitted_2pl_model, np.array([0, 1]))
        with pytest.raises(ValueError, match="coded as 0 or 1"):
            generate_plausible_values(
                fitted_2pl_model,
                np.full((1, fitted_2pl_model.model.n_items), 2),
            )

    def test_validates_polytomous_response_codes(self):
        """Reject category codes outside an item's configured range."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=2, n_categories=[3, 4])
        model._is_fitted = True

        with pytest.raises(ValueError, match="item 0 must be below 3"):
            generate_plausible_values(model, np.array([[3, 0]]))

    def test_generate_pv_posterior(self, fitted_2pl_model, dichotomous_responses):
        """Test posterior sampling method."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 3)

    def test_posterior_uses_single_batch_likelihood(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior sampling evaluates every respondent in one batch."""
        model = fitted_2pl_model.model
        original = model.log_likelihood_batch
        call_count = 0

        def counted_batch(responses, theta):
            nonlocal call_count
            call_count += 1
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", counted_batch)

        generate_plausible_values(
            model,
            dichotomous_responses["responses"],
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        assert call_count == 1

    def test_posterior_batches_people_without_changing_seeded_draws(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior row batching is reproducible across chunk sizes."""
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"][:17]
        expected = generate_plausible_values(
            model,
            responses,
            n_plausible=4,
            method="posterior",
            n_quadpts=15,
            chunk_size=len(responses),
            seed=42,
        )

        original = model.log_likelihood_batch
        batch_sizes = []

        def counted_batch(responses, theta):
            batch_sizes.append(len(responses))
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", counted_batch)
        actual = generate_plausible_values(
            model,
            responses,
            n_plausible=4,
            method="posterior",
            n_quadpts=15,
            chunk_size=6,
            seed=42,
        )

        np.testing.assert_array_equal(actual, expected)
        assert batch_sizes == [6, 6, 5]

    def test_posterior_rejects_malformed_batch_likelihood(
        self,
        fitted_2pl_model,
        dichotomous_responses,
        monkeypatch,
    ):
        """Posterior generation checks the model's batch likelihood contract."""
        model = fitted_2pl_model.model
        monkeypatch.setattr(
            model,
            "log_likelihood_batch",
            lambda responses, theta: np.zeros(len(responses)),
        )

        with pytest.raises(ValueError, match="quadrature node"):
            generate_plausible_values(
                model,
                dichotomous_responses["responses"][:3],
                n_plausible=2,
                method="posterior",
                seed=42,
            )

    def test_inverse_cdf_search_matches_reference_scan(self):
        """Flattened row search agrees with direct categorical scanning."""
        cumulative = np.array(
            [
                [0.1, 0.4, 1.0],
                [0.0, 0.25, 1.0],
                [0.7, 0.9, 1.0],
            ]
        )
        uniforms = np.array(
            [
                [0.0, 0.1, 0.4, 0.999],
                [0.0, 0.2, 0.25, 0.8],
                [0.0, 0.7, 0.95, 0.999],
            ]
        )
        expected = np.sum(
            uniforms[:, :, None] > cumulative[:, None, :],
            axis=2,
        )

        actual = _inverse_cdf_rows(cumulative, uniforms)

        np.testing.assert_array_equal(actual, expected)

    def test_generate_pv_mcmc(self, fitted_2pl_model_small):
        """Test MCMC sampling method."""
        responses = fitted_2pl_model_small["responses"]
        result = fitted_2pl_model_small["result"]

        pvs = generate_plausible_values(
            result,
            responses,
            n_plausible=3,
            method="mcmc",
            seed=42,
        )

        n_persons = responses.shape[0]
        assert pvs.shape == (n_persons, 1, 3)
        assert not np.array_equal(pvs[:, :, 0], pvs[:, :, -1])

    def test_mcmc_batches_people_and_reuses_current_log_density(
        self,
        fitted_2pl_model_small,
        monkeypatch,
    ):
        """MCMC uses one paired likelihood call per iteration and batch."""
        model = fitted_2pl_model_small["result"].model
        responses = fitted_2pl_model_small["responses"][:5]
        original = model.log_likelihood
        batch_sizes = []

        def counted_likelihood(responses, theta):
            batch_sizes.append(len(responses))
            return original(responses, theta)

        monkeypatch.setattr(model, "log_likelihood", counted_likelihood)

        generate_plausible_values(
            model,
            responses,
            n_plausible=3,
            method="mcmc",
            n_iter=4,
            burn_in=2,
            chunk_size=2,
            seed=42,
        )

        n_batches = 3
        assert len(batch_sizes) == n_batches * (1 + 2 + 3 * 4)
        assert batch_sizes == [2, 2, 1] * (1 + 2 + 3 * 4)

    def test_mcmc_controls_are_reproducible(
        self,
        fitted_2pl_model_small,
    ):
        """Burn-in and proposal tuning retain seeded reproducibility."""
        model = fitted_2pl_model_small["result"].model
        responses = fitted_2pl_model_small["responses"][:8]
        kwargs = {
            "n_plausible": 3,
            "method": "mcmc",
            "n_iter": 5,
            "burn_in": 4,
            "proposal_scale": 0.35,
            "chunk_size": 3,
            "seed": 42,
        }

        first = generate_plausible_values(model, responses, **kwargs)
        second = generate_plausible_values(model, responses, **kwargs)
        unchunked = generate_plausible_values(
            model,
            responses,
            **{**kwargs, "chunk_size": len(responses)},
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, unchunked)
        assert np.isfinite(first).all()
        assert not np.array_equal(first[:, :, 0], first[:, :, -1])

    def test_mcmc_supports_paired_polytomous_likelihoods(self):
        """Batched sampling supports fitted ordinal item models."""
        from mirt.models.polytomous import GradedResponseModel

        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model._is_fitted = True
        responses = np.array(
            [[0, 1, 2], [2, 3, 1], [1, -1, 0], [2, 0, 2]],
            dtype=int,
        )

        pvs = generate_plausible_values(
            model,
            responses,
            n_plausible=2,
            method="mcmc",
            n_iter=3,
            burn_in=2,
            seed=7,
        )

        assert pvs.shape == (4, 1, 2)
        assert np.isfinite(pvs).all()

    def test_mcmc_supports_multidimensional_likelihoods(self):
        """Batched sampling advances every latent dimension."""
        from mirt.models.dichotomous import TwoParameterLogistic

        model = TwoParameterLogistic(n_items=3, n_factors=2)
        model._is_fitted = True
        responses = np.array([[0, 1, 0], [1, 1, 0], [1, -1, 1], [0, 0, 1]])

        pvs = generate_plausible_values(
            model,
            responses,
            n_plausible=3,
            method="mcmc",
            n_iter=4,
            burn_in=3,
            seed=11,
        )

        assert pvs.shape == (4, 2, 3)
        assert np.isfinite(pvs).all()
        assert np.any(pvs[:, 0, :] != 0.0)
        assert np.any(pvs[:, 1, :] != 0.0)

    def test_pv_variability(self, fitted_2pl_model, dichotomous_responses):
        """Test that PVs show appropriate variability."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            method="posterior",
            seed=42,
        )

        pv_variance = np.var(pvs, axis=2)
        assert np.mean(pv_variance) > 0

    def test_pv_correlation_with_ability(self, fitted_2pl_model, dichotomous_responses):
        """Test that PVs correlate with true ability."""
        responses = dichotomous_responses["responses"]
        true_theta = dichotomous_responses["theta"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        avg_pv = pvs[:, 0, :].mean(axis=1)
        correlation = np.corrcoef(avg_pv, true_theta)[0, 1]

        assert correlation > 0.5


class TestCombinePlausibleValues:
    """Tests for Rubin's combining rules."""

    def test_combine_estimates(self):
        """Test combining point estimates."""
        estimates = [1.0, 1.1, 0.9, 1.05, 0.95]

        result = combine_plausible_values(estimates)

        assert "estimate" in result
        assert isinstance(result["estimate"], float)
        assert result["estimate"] == pytest.approx(1.0, abs=0.05)
        assert result["n_imputations"] == 5

    def test_combine_with_variances(self):
        """Test combining with within-imputation variances."""
        estimates = [1.0, 1.1, 0.9, 1.05, 0.95]
        variances = [0.01, 0.01, 0.01, 0.01, 0.01]

        result = combine_plausible_values(estimates, variances)

        assert "variance" in result
        assert "se" in result
        assert "within_var" in result
        assert "between_var" in result
        assert "df" in result

    def test_rubin_variance_formula(self):
        """Test Rubin's variance formula."""
        estimates = [1.0, 1.2, 0.8]
        variances = [0.1, 0.1, 0.1]

        result = combine_plausible_values(estimates, variances)

        m = 3
        within = 0.1
        between = np.var(estimates, ddof=1)
        expected_total = within + (1 + 1 / m) * between

        assert result["variance"] == pytest.approx(expected_total, rel=0.01)

    def test_vector_estimates_preserve_shape(self):
        """Vector inputs return one combined result per component."""
        estimates = (
            np.array([1.0, 4.0]),
            np.array([2.0, 5.0]),
            np.array([3.0, 6.0]),
        )

        result = combine_plausible_values(estimates)

        np.testing.assert_allclose(result["estimate"], [2.0, 5.0])
        np.testing.assert_allclose(result["between_var"], [1.0, 1.0])

    def test_degrees_of_freedom_are_computed_per_component(self):
        """Zero within variance in one component does not suppress all df."""
        estimates = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 1.0]),
            np.array([3.0, 3.0, 1.0]),
        ]
        variances = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ]

        with np.errstate(all="raise"):
            result = combine_plausible_values(estimates, variances)

        assert result["df"][0] > 2.0
        assert result["df"][1] == pytest.approx(2.0)
        assert np.isinf(result["df"][2])
        assert result["se"][2] == 0.0

    @pytest.mark.parametrize("estimates", [[], [1.0]])
    def test_requires_multiple_estimates(self, estimates):
        """Combining requires enough estimates for between variance."""
        with pytest.raises(ValueError, match="at least two"):
            combine_plausible_values(estimates)

    @pytest.mark.parametrize(
        ("estimates", "message"),
        [
            ([np.ones(2), np.ones(3)], "same shape"),
            ([1.0, "invalid"], "numeric"),
            ([1.0, np.nan], "finite"),
            ([1.0, np.inf], "finite"),
        ],
    )
    def test_validates_estimates(self, estimates, message):
        """Estimate errors have an explicit public contract."""
        with pytest.raises(ValueError, match=message):
            combine_plausible_values(estimates)

    @pytest.mark.parametrize(
        ("variances", "message"),
        [
            ([1.0], "number"),
            ([np.ones(2), np.ones(2)], "shape"),
            ([1.0, "invalid"], "numeric"),
            ([1.0, np.nan], "finite"),
            ([1.0, np.inf], "finite"),
            ([1.0, -1.0], "nonnegative"),
        ],
    )
    def test_validates_variances(self, variances, message):
        """Variance errors have an explicit public contract."""
        with pytest.raises(ValueError, match=message):
            combine_plausible_values([1.0, 2.0], variances)


class TestPlausibleValueRegression:
    """Tests for regression using plausible values."""

    @staticmethod
    def _regression_data():
        rng = np.random.default_rng(20260828)
        n_people = 80
        latent = rng.normal(size=(n_people, 2))
        pvs = latent[:, :, None] + rng.normal(
            scale=0.2,
            size=(n_people, 2, 4),
        )
        covariate = rng.normal(size=n_people)
        outcome = (
            0.7
            + 1.2 * latent[:, 0]
            - 0.8 * latent[:, 1]
            + 0.4 * covariate
            + rng.normal(scale=0.3, size=n_people)
        )
        weights = rng.uniform(0.25, 3.0, size=n_people)
        return pvs, outcome, covariate, weights

    def test_pv_regression(self, fitted_2pl_model, dichotomous_responses):
        """Test regression with PVs as predictor."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        rng = np.random.default_rng(42)
        y = true_theta + rng.standard_normal(len(true_theta)) * 0.5

        reg_result = plausible_value_regression(pvs, y)

        assert "coefficients" in reg_result
        assert "se" in reg_result
        assert "pvalues" in reg_result

    def test_pv_regression_significance(self, fitted_2pl_model, dichotomous_responses):
        """Test that regression detects true relationship."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        true_theta = dichotomous_responses["theta"]
        rng = np.random.default_rng(42)
        y = 2 * true_theta + rng.standard_normal(len(true_theta)) * 0.1

        reg_result = plausible_value_regression(pvs, y)

        assert reg_result["coefficients"][1] > 0

    def test_weighted_regression_matches_drawwise_reference(self):
        """Weighted coefficients and uncertainty follow Rubin's rules."""
        pvs, outcome, covariate, weights = self._regression_data()
        sqrt_weights = np.sqrt(weights)
        residual_df = len(outcome) - 4
        coefficients = []
        variances = []
        identity = np.eye(4)
        for draw in range(pvs.shape[2]):
            design = np.column_stack(
                [np.ones(len(outcome)), pvs[:, :, draw], covariate]
            )
            weighted_design = design * sqrt_weights[:, None]
            coefficient = np.linalg.lstsq(
                weighted_design,
                outcome * sqrt_weights,
                rcond=None,
            )[0]
            residual = outcome - design @ coefficient
            mse = np.dot(weights, residual**2) / residual_df
            inverse_gram = np.linalg.solve(
                weighted_design.T @ weighted_design,
                identity,
            )
            coefficients.append(coefficient)
            variances.append(mse * np.diag(inverse_gram))

        coefficients = np.asarray(coefficients)
        variances = np.asarray(variances)
        expected_coefficients = np.mean(coefficients, axis=0)
        expected_within = np.mean(variances, axis=0)
        expected_between = np.var(coefficients, axis=0, ddof=1)
        expected_se = np.sqrt(expected_within + 1.25 * expected_between)

        result = plausible_value_regression(
            pvs,
            outcome,
            X=covariate,
            weights=weights,
        )

        np.testing.assert_allclose(result["coefficients"], expected_coefficients)
        np.testing.assert_allclose(result["se"], expected_se)
        assert np.isfinite(result["pvalues"]).all()
        assert np.isfinite(result["df"]).all()
        assert result["n_observations"] == len(outcome)
        assert result["n_plausible"] == pvs.shape[2]

    def test_weight_scale_does_not_change_regression(self):
        """Case-weight normalization has no effect on estimates or inference."""
        pvs, outcome, covariate, weights = self._regression_data()

        result = plausible_value_regression(pvs, outcome, covariate, weights)
        scaled = plausible_value_regression(
            pvs,
            outcome,
            covariate,
            weights * 1000.0,
        )

        for key in ("coefficients", "se", "pvalues", "df"):
            np.testing.assert_allclose(result[key], scaled[key], rtol=1e-11)

    def test_single_plausible_value_matches_ordinary_least_squares(self):
        """One plausible value retains standard regression inference."""
        pvs, outcome, covariate, _ = self._regression_data()
        single = pvs[:, :, :1]
        design = np.column_stack([np.ones(len(outcome)), single[:, :, 0], covariate])
        coefficient = np.linalg.lstsq(design, outcome, rcond=None)[0]
        residual_df = len(outcome) - design.shape[1]
        residual = outcome - design @ coefficient
        mse = (residual @ residual) / residual_df
        expected_se = np.sqrt(
            mse * np.diag(np.linalg.solve(design.T @ design, np.eye(4)))
        )

        result = plausible_value_regression(single, outcome, X=covariate)

        np.testing.assert_allclose(result["coefficients"], coefficient)
        np.testing.assert_allclose(result["se"], expected_se)
        np.testing.assert_array_equal(result["df"], np.full(4, residual_df))
        assert np.isfinite(result["pvalues"]).all()

    def test_one_dimensional_covariates_match_column_matrix(self):
        """A single covariate accepts either common array shape."""
        pvs, outcome, covariate, _ = self._regression_data()

        one_dimensional = plausible_value_regression(pvs, outcome, covariate)
        column_matrix = plausible_value_regression(pvs, outcome, covariate[:, None])

        for key in ("coefficients", "se", "pvalues", "df"):
            np.testing.assert_allclose(one_dimensional[key], column_matrix[key])

    @pytest.mark.parametrize(
        ("pvs", "message"),
        [
            (np.ones((4, 2)), "shape"),
            (np.empty((0, 1, 2)), "positive"),
            (np.full((4, 1, 2), np.nan), "finite"),
            (np.full((4, 1, 2), "invalid"), "numeric"),
        ],
    )
    def test_validates_plausible_values(self, pvs, message):
        """Malformed plausible-value arrays fail before regression."""
        with pytest.raises(ValueError, match=message):
            plausible_value_regression(pvs, np.ones(4))

    @pytest.mark.parametrize(
        ("argument", "value", "message"),
        [
            ("y", np.ones((8, 1)), "one outcome"),
            ("y", np.ones(7), "one outcome"),
            ("y", np.full(8, np.nan), "finite"),
            ("y", np.full(8, "invalid"), "numeric"),
            ("X", np.ones((8, 1, 1)), "one row"),
            ("X", np.ones((7, 1)), "one row"),
            ("X", np.full((8, 1), np.nan), "finite"),
            ("X", np.full((8, 1), "invalid"), "numeric"),
            ("weights", np.ones((8, 1)), "one value"),
            ("weights", np.ones(7), "one value"),
            ("weights", np.zeros(8), "positive finite"),
            ("weights", np.full(8, np.inf), "positive finite"),
            ("weights", np.full(8, "invalid"), "numeric"),
        ],
    )
    def test_validates_regression_inputs(self, argument, value, message):
        """Outcomes, covariates, and weights have explicit contracts."""
        rng = np.random.default_rng(7)
        kwargs = {"y": rng.normal(size=8), argument: value}

        with pytest.raises(ValueError, match=message):
            plausible_value_regression(rng.normal(size=(8, 1, 2)), **kwargs)

    def test_requires_residual_degrees_of_freedom(self):
        """Regression requires more rows than fitted coefficients."""
        pvs = np.arange(4, dtype=float).reshape(2, 1, 2)

        with pytest.raises(ValueError, match="more people"):
            plausible_value_regression(pvs, np.ones(2))

    def test_rejects_rank_deficient_design(self):
        """Collinear predictors produce a useful error instead of NaNs."""
        rng = np.random.default_rng(11)
        pvs = rng.normal(size=(12, 1, 2))

        with pytest.raises(ValueError, match="rank deficient"):
            plausible_value_regression(pvs, rng.normal(size=12), X=np.ones(12))

    def test_reports_linear_algebra_failure(self, monkeypatch):
        """Numerical failures identify the affected plausible value."""

        def fail_lstsq(*args, **kwargs):
            raise np.linalg.LinAlgError("did not converge")

        monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)

        with pytest.raises(ValueError, match="plausible value 0"):
            plausible_value_regression(
                np.arange(16, dtype=float).reshape(8, 1, 2),
                np.arange(8, dtype=float),
            )


class TestPlausibleValueStatistics:
    """Tests for computing statistics with PVs."""

    @staticmethod
    def _multidimensional_values() -> np.ndarray:
        rng = np.random.default_rng(20260828)
        return rng.normal(size=(40, 3, 5))

    @pytest.mark.parametrize("statistic", ["mean", "variance", "sd", "percentile_25"])
    def test_all_factors_match_drawwise_reference(self, statistic):
        """All-factor summaries retain one result per latent dimension."""
        pvs = self._multidimensional_values()
        if statistic == "mean":
            draw_estimates = np.mean(pvs, axis=0)
        elif statistic == "variance":
            draw_estimates = np.var(pvs, axis=0, ddof=1)
        elif statistic == "sd":
            draw_estimates = np.std(pvs, axis=0, ddof=1)
        else:
            draw_estimates = np.percentile(pvs, 25, axis=0)
        expected_estimate = np.mean(draw_estimates, axis=1)
        expected_between = np.var(draw_estimates, axis=1, ddof=1)
        expected_se = np.sqrt(expected_between * 1.2)

        result = plausible_value_statistics(
            pvs,
            statistic=statistic,
            factor="all",
        )

        np.testing.assert_allclose(result["estimate"], expected_estimate)
        np.testing.assert_allclose(result["between_var"], expected_between)
        np.testing.assert_allclose(result["se"], expected_se)
        assert result["n_plausible"] == 5

    def test_specific_factor_and_default_return_scalars(self):
        """Selecting one factor preserves the scalar result contract."""
        pvs = self._multidimensional_values()
        all_factors = plausible_value_statistics(pvs, factor="all")

        default = plausible_value_statistics(pvs)
        second = plausible_value_statistics(pvs, factor=1)

        assert isinstance(default["estimate"], float)
        assert default["estimate"] == pytest.approx(all_factors["estimate"][0])
        assert default["se"] == pytest.approx(all_factors["se"][0])
        assert second["estimate"] == pytest.approx(all_factors["estimate"][1])
        assert second["between_var"] == pytest.approx(all_factors["between_var"][1])

    @pytest.mark.parametrize(
        ("pvs", "message"),
        [
            (np.ones((3, 2)), "shape"),
            (np.empty((0, 2, 3)), "one person"),
            (np.empty((3, 0, 3)), "one person"),
            (np.ones((3, 2, 0)), "one plausible"),
            (np.full((3, 2, 3), np.nan), "finite"),
            (np.full((3, 2, 3), "invalid"), "numeric"),
        ],
    )
    def test_validates_plausible_value_matrix(self, pvs, message):
        with pytest.raises(ValueError, match=message):
            plausible_value_statistics(pvs)

    @pytest.mark.parametrize("factor", [-1, 3, True, 1.5, "first"])
    def test_validates_factor_selection(self, factor):
        with pytest.raises(ValueError, match="factor"):
            plausible_value_statistics(
                self._multidimensional_values(),
                factor=factor,
            )

    @pytest.mark.parametrize(
        ("statistic", "message"),
        [
            ("median", "Unknown"),
            ("percentile_", "end with"),
            ("percentile_nan", "finite"),
            ("percentile_-1", "from 0 to 100"),
            ("percentile_101", "from 0 to 100"),
            ("percentile_50_extra", "end with"),
            (1, "string"),
        ],
    )
    def test_validates_statistic(self, statistic, message):
        with pytest.raises(ValueError, match=message):
            plausible_value_statistics(
                self._multidimensional_values(),
                statistic=statistic,
            )

    @pytest.mark.parametrize("statistic", ["variance", "sd"])
    def test_dispersion_requires_two_people(self, statistic):
        with pytest.raises(ValueError, match="two people"):
            plausible_value_statistics(
                np.ones((1, 2, 3)),
                statistic=statistic,
                factor="all",
            )

    def test_single_plausible_value_has_undefined_uncertainty(self):
        pvs = np.arange(8, dtype=float).reshape(4, 2, 1)

        result = plausible_value_statistics(pvs, factor="all")

        np.testing.assert_allclose(result["estimate"], [3.0, 4.0])
        assert np.isnan(result["between_var"]).all()
        assert np.isnan(result["se"]).all()
        assert result["n_plausible"] == 1

    def test_pv_mean(self, fitted_2pl_model, dichotomous_responses):
        """Test population mean estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="mean")

        assert "estimate" in stats
        assert "se" in stats
        assert abs(stats["estimate"]) < 1.0

    def test_pv_variance(self, fitted_2pl_model, dichotomous_responses):
        """Test population variance estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="variance")

        assert "estimate" in stats
        assert 0.5 < stats["estimate"] < 2.0

    def test_pv_percentile(self, fitted_2pl_model, dichotomous_responses):
        """Test percentile estimation."""
        responses = dichotomous_responses["responses"]

        pvs = generate_plausible_values(
            fitted_2pl_model,
            responses,
            n_plausible=3,
            seed=42,
        )

        stats = plausible_value_statistics(pvs, statistic="percentile_50")

        assert "estimate" in stats
        assert abs(stats["estimate"]) < 1.0
