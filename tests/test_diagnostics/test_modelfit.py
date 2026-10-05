"""Tests for model fit statistics (M2, RMSEA, CFI, TLI)."""

import numpy as np
import pytest

from mirt import compute_fit_indices, compute_m2
from mirt.diagnostics import modelfit
from mirt.diagnostics.modelfit import (
    _compute_expected_margins,
    _compute_rmsea,
    _compute_rmsea_ci,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models import GradedResponseModel, TwoParameterLogistic


class TestM2:
    """Tests for M2 limited-information statistic."""

    def test_compute_m2(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)

        assert "M2" in m2_result
        assert "df" in m2_result
        assert "p_value" in m2_result

        assert m2_result["M2"] >= 0
        assert m2_result["df"] > 0
        assert 0 <= m2_result["p_value"] <= 1

    def test_m2_with_fit_result(self, fitted_2pl_model, dichotomous_responses):
        """Test M2 computation with FitResult object."""
        responses = dichotomous_responses["responses"]

        m2_result = compute_m2(fitted_2pl_model.model, responses)
        assert "M2" in m2_result

    def test_supplied_theta_changes_expected_moments(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=int)

        centered = compute_m2(model, responses, theta=np.zeros(4))
        high_ability = compute_m2(model, responses, theta=np.full(4, 2.0))

        # With fixed zero abilities the nuisance tangent contains only the
        # three item means. The remaining three pair associations are tested.
        assert centered["M2"] == pytest.approx(4.0)
        assert high_ability["M2"] > centered["M2"]

    def test_polytomous_collapsed_score_moments(self):
        model = GradedResponseModel(n_items=6, n_categories=3)
        responses = np.array(
            [
                [0, 1, 2, 0, 1, 2],
                [1, 2, 0, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
                [0, 0, 1, 2, 1, 2],
            ],
            dtype=int,
        )

        result = compute_m2(model, responses, theta=np.linspace(-1.5, 1.5, 4))
        quadrature_result = compute_m2(model, responses, n_quadpts=9)

        assert np.isfinite(result["M2"])
        assert result["M2"] >= 0.0
        assert 0.0 <= result["p_value"] <= 1.0
        assert np.isfinite(quadrature_result["M2"])

    @pytest.mark.parametrize(
        ("responses", "message"),
        [
            (np.zeros((1, 3)), "at least 2 persons"),
            (np.zeros((3, 2)), "expected 3"),
            (np.full((3, 3), np.nan), "no observed"),
            (np.full((3, 3), 0.5), "integer category"),
            (np.full((3, 3), 2), "between 0 and 1"),
        ],
    )
    def test_input_validation(self, responses, message):
        with pytest.raises(ValueError, match=message):
            compute_m2(TwoParameterLogistic(n_items=3), responses)

    def test_theta_and_quadrature_validation(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.zeros((4, 3), dtype=int)

        with pytest.raises(ValueError, match="theta must have shape"):
            compute_m2(model, responses, theta=np.zeros(3))
        with pytest.raises(ValueError, match="n_quadpts"):
            compute_m2(model, responses, n_quadpts=1)

    def test_vectorized_quadrature_matches_itemwise_calculation(self):
        model = TwoParameterLogistic(n_items=4)
        model.set_parameters(
            discrimination=np.array([0.7, 1.0, 1.3, 1.8]),
            difficulty=np.array([-1.0, -0.25, 0.5, 1.25]),
        )
        quadrature = GaussHermiteQuadrature(n_points=15, n_dimensions=1)

        expected_uni, expected_bi = _compute_expected_margins(model, 15)
        item_probabilities = np.column_stack(
            [model.probability(quadrature.nodes, idx) for idx in range(model.n_items)]
        )
        manual_uni = quadrature.weights @ item_probabilities
        manual_bi = (item_probabilities * quadrature.weights[:, None]).T @ (
            item_probabilities
        )

        np.testing.assert_allclose(expected_uni, manual_uni)
        np.testing.assert_allclose(expected_bi, manual_bi)


class TestFitIndices:
    """Tests for RMSEA, CFI, TLI, SRMSR."""

    def test_compute_fit_indices(self, fitted_2pl_model, dichotomous_responses):
        """Test fit indices computation."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert "RMSEA" in fit_stats
        assert "CFI" in fit_stats
        assert "TLI" in fit_stats
        assert "SRMSR" in fit_stats

    def test_rmsea_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that RMSEA is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["RMSEA"] >= 0

    def test_cfi_tli_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that CFI/TLI are in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["CFI"] >= 0
        assert fit_stats["TLI"] >= -0.5

    def test_rmsea_ci(self, fitted_2pl_model, dichotomous_responses):
        """Test RMSEA confidence intervals."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        if "RMSEA_CI_lower" in fit_stats:
            assert fit_stats["RMSEA_CI_lower"] <= fit_stats["RMSEA"]
            assert fit_stats["RMSEA_CI_upper"] >= fit_stats["RMSEA"]

    def test_srmsr_range(self, fitted_2pl_model, dichotomous_responses):
        """Test that SRMSR is in valid range."""
        responses = dichotomous_responses["responses"]

        fit_stats = compute_fit_indices(fitted_2pl_model.model, responses)

        assert fit_stats["SRMSR"] >= 0

    def test_missing_codes_are_equivalent(self):
        model = TwoParameterLogistic(n_items=3)
        responses = np.array(
            [[-1, 1, 0], [1, 0, 1], [1, -1, 0], [0, 0, 1]], dtype=float
        )
        theta = np.linspace(-1.0, 1.0, 4)
        nan_responses = responses.copy()
        nan_responses[nan_responses < 0] = np.nan

        coded = compute_fit_indices(model, responses, theta=theta)
        missing = compute_fit_indices(model, nan_responses, theta=theta)

        np.testing.assert_allclose(
            list(coded.values()),
            list(missing.values()),
            equal_nan=True,
        )

    def test_fit_indices_do_not_mutate_model_parameters(self):
        model = TwoParameterLogistic(n_items=4)
        responses = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
        original = model.parameters
        compute_fit_indices(model, responses, theta=np.linspace(-1.0, 1.0, 4))
        for name, values in original.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    def test_rmsea_interval_is_ordered_and_contains_estimate(self):
        estimate = _compute_rmsea(10.2, 1, 4)
        lower, upper = _compute_rmsea_ci(10.2, 1, 4)

        assert lower <= estimate <= upper


def _simulated_fit_data(model_name, n_items, n_persons, seed, missing=0.0):
    """Return a parameterized model and responses simulated from it."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_persons, 1))
    if model_name == "2PL":
        model = TwoParameterLogistic(n_items=n_items)
        model.set_parameters(
            discrimination=rng.uniform(0.8, 2.5, n_items),
            difficulty=rng.normal(size=n_items),
        )
        probabilities = model.probability(theta)
        responses = (rng.random(probabilities.shape) < probabilities).astype(float)
    else:
        model = GradedResponseModel(n_items=n_items, n_categories=4)
        probabilities = model.probability(theta)
        draws = rng.random((n_persons, n_items, 1))
        responses = np.minimum((draws > probabilities.cumsum(axis=2)).sum(axis=2), 3)
        responses = responses.astype(float)
    responses[rng.random(responses.shape) < missing] = np.nan
    return model, responses


class TestCholeskyWhitening:
    """M2 whitens well-conditioned covariances with a Cholesky factor."""

    @staticmethod
    def _eigenvalue_path_only(monkeypatch):
        monkeypatch.setattr(modelfit, "_well_conditioned_cholesky", lambda _: None)

    @pytest.mark.parametrize(
        ("model_name", "n_items", "missing"),
        [("2PL", 8, 0.0), ("2PL", 12, 0.1), ("GRM", 6, 0.0), ("GRM", 7, 0.1)],
    )
    def test_fit_indices_match_eigenvalue_whitening(
        self, monkeypatch, model_name, n_items, missing
    ):
        model, responses = _simulated_fit_data(
            model_name, n_items, 800, seed=n_items, missing=missing
        )
        fast = compute_fit_indices(model, responses)

        self._eigenvalue_path_only(monkeypatch)
        reference = compute_fit_indices(model, responses)

        assert fast.keys() == reference.keys()
        assert fast["M2_df"] == reference["M2_df"]
        np.testing.assert_allclose(
            list(fast.values()), list(reference.values()), rtol=1e-10, atol=1e-12
        )

    def test_sixty_item_m2_takes_the_cholesky_path(self, monkeypatch):
        model, responses = _simulated_fit_data("2PL", 60, 2000, seed=60)
        factors = []
        original = modelfit._well_conditioned_cholesky

        def spy(matrix):
            factor = original(matrix)
            factors.append(factor is not None)
            return factor

        monkeypatch.setattr(modelfit, "_well_conditioned_cholesky", spy)
        fast = compute_m2(model, responses)
        assert factors == [True]

        self._eigenvalue_path_only(monkeypatch)
        reference = compute_m2(model, responses)
        assert fast["df"] == reference["df"] == 1830 - 120
        np.testing.assert_allclose(fast["M2"], reference["M2"], rtol=1e-10)
        np.testing.assert_allclose(fast["p_value"], reference["p_value"], rtol=1e-8)

    def test_singular_covariance_uses_eigenvalue_rank(self):
        rng = np.random.default_rng(11)
        loadings = rng.normal(size=(6, 4))
        covariance = loadings @ loadings.T
        jacobian = loadings[:, :1].copy()
        scales = np.sqrt(np.diag(covariance))

        assert (
            modelfit._well_conditioned_cholesky(covariance / np.outer(scales, scales))
            is None
        )

        supported = loadings @ rng.normal(size=4)
        statistic, degrees = modelfit._projected_chi_square(
            supported, covariance, jacobian
        )
        # Weight the residual with the pseudoinverse after projecting out the
        # tangent direction within the four-dimensional covariance support.
        coefficients = np.linalg.lstsq(loadings, supported, rcond=None)[0]
        tangent = np.linalg.lstsq(loadings, jacobian[:, 0], rcond=None)[0]
        tangent /= np.linalg.norm(tangent)
        expected = coefficients @ coefficients - (tangent @ coefficients) ** 2
        assert degrees == 3
        assert statistic == pytest.approx(expected, rel=1e-8)

        unsupported = supported + np.linalg.svd(loadings)[0][:, -1]
        statistic, degrees = modelfit._projected_chi_square(
            unsupported, covariance, jacobian
        )
        assert degrees == 3
        assert np.isinf(statistic)

    def test_complete_rows_are_counted_once_per_design(self, monkeypatch):
        rng = np.random.default_rng(3)
        responses = (rng.random((50, 4)) < 0.5).astype(float)
        responses[[7, 31], [1, 3]] = np.nan
        monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 40)

        design = modelfit._moment_design(responses)

        present = modelfit._score_features(np.isfinite(responses).astype(float))
        np.testing.assert_array_equal(design.counts, present.sum(axis=0))
        np.testing.assert_array_equal(design.overlap, present.T @ present)
