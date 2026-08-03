"""Tests for model parameter extraction and estimating functions."""

import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
)
from mirt.utils import estfun as public_estfun
from mirt.utils import estfun_summary as public_estfun_summary
from mirt.utils.extraction import (
    coef,
    estfun,
    estfun_summary,
    extract_item,
    itemplot_data,
    mod2values,
)


def _central_scores(model, responses, theta):
    """Reference conditional scores from central finite differences."""
    work_model = model.copy()
    step_scale = np.cbrt(np.finfo(np.float64).eps)
    columns = []
    for name, values in model.parameters.items():
        work_values = work_model._parameters[name]
        for index in np.ndindex(values.shape):
            original = float(work_values[index])
            step = step_scale * max(1.0, abs(original))
            work_values[index] = original + step
            ll_plus = work_model.log_likelihood(responses, theta)
            work_values[index] = original - step
            ll_minus = work_model.log_likelihood(responses, theta)
            work_values[index] = original
            columns.append((ll_plus - ll_minus) / (2 * step))
    return np.column_stack(columns)


class TestModelValues:
    def test_estimating_functions_are_exported_from_utils(self):
        assert public_estfun is estfun
        assert public_estfun_summary is estfun_summary

    def test_uses_public_model_name_and_copies_parameters(self):
        model = TwoParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.5, 0.8]),
            difficulty=np.array([-0.5, 1.0]),
        )

        values = mod2values(model)

        assert values.model_type == "2PL"
        assert values.n_dimensions == 1
        np.testing.assert_array_equal(values.discrimination, np.array([[1.5], [0.8]]))
        np.testing.assert_array_equal(values.difficulty, model.difficulty)
        values.parameters["difficulty"][0] = 99
        assert model.difficulty[0] == -0.5

    def test_maps_polytomous_native_parameters(self):
        grm = GradedResponseModel(n_items=2, n_categories=3)
        nrm = NominalResponseModel(n_items=2, n_categories=3)

        grm_values = mod2values(grm)
        nrm_values = mod2values(nrm)

        assert grm_values.model_type == "GRM"
        np.testing.assert_array_equal(grm_values.difficulty, grm.thresholds)
        assert set(grm_values.parameters) == {"discrimination", "thresholds"}
        assert nrm_values.model_type == "NRM"
        np.testing.assert_array_equal(nrm_values.discrimination, nrm.slopes)
        np.testing.assert_array_equal(nrm_values.difficulty, nrm.intercepts)

    def test_exposes_four_parameter_upper_asymptote(self):
        model = FourParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.1, 0.9]),
            difficulty=np.array([-0.2, 0.4]),
            guessing=np.array([0.1, 0.2]),
            upper=np.array([0.8, 0.9]),
        )

        values = mod2values(model)
        item = extract_item(model, 1)

        assert values.model_type == "4PL"
        np.testing.assert_array_equal(values.upper, model.upper)
        assert item.upper == pytest.approx(0.9)
        assert item.parameters["upper"] == pytest.approx(0.9)

    def test_extract_item_preserves_item_and_shared_parameters(self):
        grm = GradedResponseModel(n_items=2, n_categories=3)
        rating = GradedRatingScaleModel(n_items=2, n_categories=3)

        grm_item = extract_item(grm, 1)
        rating_item = extract_item(rating, 1)

        np.testing.assert_array_equal(
            grm_item.parameters["thresholds"], grm.thresholds[1]
        )
        np.testing.assert_array_equal(
            rating_item.parameters["thresholds"], rating.thresholds
        )
        np.testing.assert_array_equal(
            rating_item.parameters["discrimination"],
            rating.parameters["discrimination"],
        )

    @pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
    def test_extract_item_validates_index(self, item_idx):
        model = TwoParameterLogistic(n_items=2)

        with pytest.raises(MirtValidationError, match="item_idx"):
            extract_item(model, item_idx)


class TestCoefficients:
    def test_irt_parameters_use_exact_native_names(self):
        grm = GradedResponseModel(n_items=2, n_categories=3)
        four_pl = FourParameterLogistic(n_items=2)

        grm_coefficients = coef(grm)
        four_pl_coefficients = coef(four_pl)

        assert set(grm_coefficients) == {"discrimination", "thresholds"}
        assert set(four_pl_coefficients) == {
            "discrimination",
            "difficulty",
            "guessing",
            "upper",
        }

    def test_multidimensional_slope_intercept_conversion(self):
        model = TwoParameterLogistic(n_items=2, n_factors=2)
        discrimination = np.array([[1.0, 2.0], [-0.5, 1.5]])
        difficulty = np.array([0.5, -1.0])
        model.set_parameters(
            discrimination=discrimination,
            difficulty=difficulty,
        )

        coefficients = coef(model, irt_pars=False)

        np.testing.assert_array_equal(coefficients["slope"], discrimination)
        np.testing.assert_array_equal(
            coefficients["intercept"], -discrimination.sum(axis=1) * difficulty
        )

    def test_slope_intercept_rejects_incompatible_parameterization(self):
        model = GradedResponseModel(n_items=2, n_categories=3)

        with pytest.raises(MirtModelError, match="requires"):
            coef(model, irt_pars=False)


class TestItemPlotData:
    def test_uses_requested_polytomous_item_information(self):
        model = GradedResponseModel(n_items=3, n_categories=3)
        theta = np.linspace(-2, 2, 9).reshape(-1, 1)

        plot_data = itemplot_data(
            model,
            item_idx=1,
            theta_range=(-2, 2),
            n_points=9,
        )

        assert plot_data["probability"].shape == (9, 3)
        np.testing.assert_allclose(
            plot_data["information"], model.information(theta, item_idx=1)
        )
        assert not np.allclose(plot_data["information"], model.information(theta))

    def test_supports_multidimensional_reference_direction(self):
        model = TwoParameterLogistic(n_items=2, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[1.0, 0.0], [0.0, 1.0]]),
            difficulty=np.zeros(2),
        )

        first_axis = itemplot_data(model, 0, theta_range=(-1, 1), n_points=3)
        second_axis = itemplot_data(
            model,
            0,
            theta_range=(-1, 1),
            n_points=3,
            reference_direction=np.array([0.0, 2.0]),
        )

        assert first_axis["probability"][0] < first_axis["probability"][-1]
        np.testing.assert_allclose(second_axis["probability"], 0.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"item_idx": -1}, "item_idx"),
            ({"item_idx": 0, "n_points": 1}, "n_points"),
            ({"item_idx": 0, "theta_range": (1, -1)}, "theta_range"),
            (
                {"item_idx": 0, "reference_direction": np.array([0.0])},
                "reference_direction",
            ),
        ],
    )
    def test_validates_plot_configuration(self, kwargs, match):
        model = TwoParameterLogistic(n_items=2, n_factors=2)

        with pytest.raises(MirtValidationError, match=match):
            itemplot_data(model, **kwargs)


class TestEstimatingFunctions:
    def test_matches_numerical_scores_for_four_parameter_model(self):
        model = FourParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            difficulty=np.array([-0.4, 0.6]),
            guessing=np.array([0.1, 0.2]),
            upper=np.array([0.85, 0.9]),
        )
        responses = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        theta = np.array([-1.0, -0.2, 0.4, 1.1])

        actual = estfun(model, responses, theta)
        expected = _central_scores(model, responses, theta.reshape(-1, 1))

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_supports_multidimensional_logistic_model(self):
        model = TwoParameterLogistic(n_items=2, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[1.0, 0.4], [-0.2, 1.3]]),
            difficulty=np.array([-0.3, 0.7]),
        )
        responses = np.array([[1, 0], [0, 1], [1, 1]])
        theta = np.array([[-0.5, 0.2], [0.1, 0.8], [1.0, -0.4]])

        actual = estfun(model, responses, theta)
        expected = _central_scores(model, responses, theta)

        assert actual.shape == (3, model.n_parameters)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_supports_polytomous_models_and_missing_values(self):
        model = GradedResponseModel(n_items=2, n_categories=3)
        responses = np.array([[0, 1], [2, 0], [-1, -1]])
        theta = np.array([-0.5, 0.7, 0.0])

        scores = estfun(model, responses, theta)

        assert scores.shape == (3, model.n_parameters)
        assert np.all(np.isfinite(scores))
        assert np.any(scores[:2] != 0)
        np.testing.assert_array_equal(scores[2], 0.0)

    def test_accepts_nan_missing_values(self):
        model = TwoParameterLogistic(n_items=2)
        responses = np.array([[1.0, np.nan], [np.nan, np.nan]])

        scores = estfun(model, responses, np.array([0.2, -0.1]))

        assert np.any(scores[0] != 0)
        np.testing.assert_array_equal(scores[1], 0.0)

    @pytest.mark.parametrize(
        ("responses", "theta", "error"),
        [
            (np.array([[2, 0]]), np.array([0.0]), MirtDataError),
            (np.array([[1, 0], [0, 1]]), np.array([0.0]), MirtDataError),
            (
                np.array([[1, 0]]),
                np.array([[np.nan]]),
                MirtValidationError,
            ),
        ],
    )
    def test_validates_inputs(self, responses, theta, error):
        model = TwoParameterLogistic(n_items=2)

        with pytest.raises(error):
            estfun(model, responses, theta)

    def test_summary_matches_score_moments(self):
        model = TwoParameterLogistic(n_items=2)
        responses = np.array([[1, 0], [0, 1], [1, 1]])
        theta = np.array([-0.5, 0.2, 0.8])

        scores = estfun(model, responses, theta)
        summary = estfun_summary(model, responses, theta)

        np.testing.assert_allclose(summary["sum"], scores.sum(axis=0))
        np.testing.assert_allclose(summary["mean"], scores.mean(axis=0))
        np.testing.assert_allclose(summary["var"], scores.var(axis=0))
        np.testing.assert_allclose(summary["meat"], scores.T @ scores / 3)
