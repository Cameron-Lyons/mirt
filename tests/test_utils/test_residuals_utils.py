"""Tests for utility residual/local-dependence helpers."""

import numpy as np

from mirt.scoring import fscores
from mirt.utils.residuals import LD_X2, Q3, residuals


class TestResidualsUtilities:
    """Smoke tests for mirt.utils.residuals public utilities."""

    def test_residuals_shapes(self, fitted_2pl_model, dichotomous_responses):
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        theta = fscores(model, responses, method="EAP").theta

        result = residuals(model, responses, theta, type="standardized")

        assert result.raw.shape == responses.shape
        assert result.standardized.shape == responses.shape
        assert result.ld_matrix is not None
        assert result.ld_matrix.shape == (responses.shape[1], responses.shape[1])

    def test_q3_matrix_properties(self, fitted_2pl_model, dichotomous_responses):
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        theta = fscores(model, responses, method="EAP").theta

        q3 = Q3(model, responses, theta, use_rust=False)

        assert q3.shape == (responses.shape[1], responses.shape[1])
        np.testing.assert_allclose(q3, q3.T)
        np.testing.assert_allclose(np.diag(q3), np.ones(responses.shape[1]))

    def test_ld_x2_shapes_and_ranges(self, fitted_2pl_model, dichotomous_responses):
        model = fitted_2pl_model.model
        responses = dichotomous_responses["responses"]
        theta = fscores(model, responses, method="EAP").theta

        chi2, p_values = LD_X2(model, responses, theta, use_rust=False)

        assert chi2.shape == (responses.shape[1], responses.shape[1])
        assert p_values.shape == (responses.shape[1], responses.shape[1])

        valid_chi2 = chi2[~np.isnan(chi2)]
        valid_p = p_values[~np.isnan(p_values)]

        assert np.all(valid_chi2 >= 0)
        assert np.all((valid_p >= 0) & (valid_p <= 1))
