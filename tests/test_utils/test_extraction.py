"""Tests for model parameter extraction and estimating functions."""

import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
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
        model = RatingScaleModel(n_items=2, n_categories=3)

        with pytest.raises(MirtModelError, match="requires"):
            coef(model, irt_pars=False)

    def test_ordinal_intercepts_match_r_coef(self):
        # mirt(..., itemtype = "graded" / "gpcm" / "Rasch") with fixed
        # parameters in R 1.48: coef(IRTpars = TRUE) gives the thresholds or
        # steps set here, coef(IRTpars = FALSE) the intercepts expected below,
        # and probtrace the category probabilities.
        theta = np.array([[-1.0], [0.5]])
        slopes = np.array([1.3, 0.7, 1.8])
        graded = GradedResponseModel(n_items=3, n_categories=[4, 4, 3])
        graded.set_parameters(
            discrimination=slopes,
            thresholds=np.array(
                [
                    [-1.153846, -0.1538462, 0.8461538],
                    [-1.285714, 0.5714286, 2.857143],
                    [-0.3333333, 0.4444444, 0.0],
                ]
            ),
        )
        gpcm = GeneralizedPartialCredit(n_items=3, n_categories=[4, 4, 3])
        gpcm.set_parameters(
            discrimination=slopes,
            steps=np.array(
                [
                    [-0.3846154, -0.5384615, 0.6153846],
                    [0.4285714, -0.5714286, 1.428571],
                    [-0.3333333, 0.2222222, 0.0],
                ]
            ),
        )
        pcm = PartialCreditModel(n_items=3, n_categories=[4, 4, 3])
        pcm.set_parameters(
            steps=np.array([[-0.5, -0.7, 0.8], [0.3, -0.4, 1.0], [-0.6, 0.4, 0.0]])
        )
        partial_credit_intercepts = [
            [0.5, 1.2, 0.4],
            [-0.3, 0.1, -0.9],
            [0.6, 0.2, 0.0],
        ]
        cases = [
            (
                graded,
                slopes,
                [[1.5, 0.2, -1.1], [0.9, -0.4, -2.0], [0.6, -0.8, 0.0]],
                [0.4501660, 0.3000941, 0.1665672, 0.0831727],
            ),
            (
                gpcm,
                slopes,
                partial_credit_intercepts,
                [0.57933289, 0.2603110, 0.1428617, 0.01749434],
            ),
            (
                pcm,
                np.ones(3),
                partial_credit_intercepts,
                [0.46945421, 0.2847384, 0.2109394, 0.03486804],
            ),
        ]
        for model, expected_slopes, expected_intercepts, first_curve in cases:
            converted = coef(model, irt_pars=False)

            assert set(converted) == {"slope", "intercept"}
            np.testing.assert_allclose(
                converted["slope"], expected_slopes[:, None], atol=1e-12
            )
            np.testing.assert_allclose(
                converted["intercept"], expected_intercepts, atol=1e-6
            )
            np.testing.assert_allclose(
                model.probability(theta, 0)[0], first_curve, atol=1e-6
            )
            assert set(coef(model)) == set(model.parameters)

    def test_four_parameter_intercepts_match_r_coef(self):
        # coef(mirt(..., "4PL"), IRTpars = FALSE) in R for a = (1.2, 0.6),
        # b = (1/3, -4/3): d = (-0.4, 0.8), with g and u unchanged.
        model = FourParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.array([1.2, 0.6]),
            difficulty=np.array([1 / 3, -4 / 3]),
            guessing=np.array([0.15, 0.05]),
            upper=np.array([0.9, 0.97]),
        )

        converted = coef(model, irt_pars=False)

        assert set(converted) == {"slope", "intercept", "guessing", "upper"}
        np.testing.assert_allclose(converted["intercept"], [-0.4, 0.8])
        np.testing.assert_array_equal(converted["upper"], [0.9, 0.97])

    def test_slope_intercept_models_are_returned_unconverted(self):
        # As R, which leaves models with cross-loadings in a/d form.
        mirt_model = MultidimensionalModel(n_items=3, n_factors=2)
        nominal = NominalResponseModel(n_items=2, n_categories=3)

        for model in (mirt_model, nominal):
            for irt_pars in (True, False):
                coefficients = coef(model, irt_pars=irt_pars)
                assert set(coefficients) == {"slopes", "intercepts"}
                for name, values in coefficients.items():
                    np.testing.assert_array_equal(values, model.parameters[name])

    def test_mixed_format_models_convert_each_component(self):
        from mirt.models.mixed_format import MixedItemModel

        model = MixedItemModel.from_itemtypes(
            ["2PL", "2PL", "GRM"], n_categories=[2, 2, 3]
        )
        grm = model.component_models[1]
        grm.set_parameters(
            discrimination=np.array([1.5]), thresholds=np.array([[-1.0, 0.5]])
        )

        converted = coef(model, irt_pars=False)

        assert set(converted) == {
            "2PL.slope",
            "2PL.intercept",
            "GRM.slope",
            "GRM.intercept",
        }
        np.testing.assert_allclose(converted["GRM.intercept"], [[1.5, -0.75]])
        assert set(coef(model)) == set(model.parameters)


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

        # One column per free coordinate, including the rotation's.
        assert actual.shape == (3, 6) == (3, model.n_parameters + 1)
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


def test_coef_accepts_fit_results_and_multigroup_results():
    import mirt

    responses = mirt.simdata(n_persons=400, n_items=5, seed=1)
    result = mirt.fit_mirt(responses, compute_standard_errors=False)
    for key, values in mirt.coef(result).items():
        np.testing.assert_array_equal(values, mirt.coef(result.model)[key])

    groups = np.repeat(["a", "b"], 200)
    multigroup = mirt.fit_multigroup(
        responses, groups, invariance="metric", compute_standard_errors=False
    )
    by_group = mirt.coef(multigroup)
    assert list(by_group) == ["a", "b"]
    np.testing.assert_array_equal(
        by_group["a"]["discrimination"], by_group["b"]["discrimination"]
    )
    assert "intercept" in mirt.coef(multigroup, irt_pars=False)["b"]
