"""Tests for item, score, and delta-method confidence intervals."""

import numpy as np
import pytest
from scipy import stats

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError, MirtEstimationError, MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.confidence import (
    PLCI,
    _marginal_log_likelihood,
    delta_method,
    score_CI,
)


@pytest.fixture
def two_pl():
    model = TwoParameterLogistic(2)
    model.set_parameters(
        discrimination=np.array([1.2, 0.8]),
        difficulty=np.array([-0.3, 0.4]),
    )
    return model


@pytest.fixture
def response_matrix():
    return np.tile(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]), (8, 1))


class TestParameterIntervals:
    def test_profile_callback_recovers_known_interval(self, two_pl, response_matrix):
        center = float(two_pl.difficulty[0])
        scale = 0.4

        def profile(value):
            return -(((value - center) / scale) ** 2)

        result = PLCI(
            two_pl,
            response_matrix,
            param_idx=0,
            param_name="difficulty",
            tol=1e-10,
            profile_log_likelihood=profile,
        )

        critical = stats.chi2.ppf(0.95, df=1) / 2
        expected_distance = scale * np.sqrt(critical)
        assert result.lower == pytest.approx(center - expected_distance, abs=1e-8)
        assert result.upper == pytest.approx(center + expected_distance, abs=1e-8)
        assert result.log_likelihood == pytest.approx(0.0)
        assert result.critical_drop == pytest.approx(critical)
        assert result.n_evaluations > 2
        assert result.used_profile_callback
        assert result.converged

    def test_reports_unbounded_profile_interval(self, two_pl, response_matrix):
        result = PLCI(
            two_pl,
            response_matrix,
            param_idx=0,
            param_name="difficulty",
            profile_log_likelihood=lambda value: 0.0,
        )

        assert np.isneginf(result.lower)
        assert np.isposinf(result.upper)
        assert not result.converged

    def test_default_interval_respects_three_parameter_guessing(self, response_matrix):
        no_guessing = ThreeParameterLogistic(2)
        no_guessing.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            difficulty=np.array([-0.3, 0.4]),
            guessing=np.array([0.0, 0.0]),
        )
        guessing = ThreeParameterLogistic(2)
        guessing.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            difficulty=np.array([-0.3, 0.4]),
            guessing=np.array([0.45, 0.35]),
        )

        baseline = PLCI(no_guessing, response_matrix, 0, "difficulty")
        adjusted = PLCI(guessing, response_matrix, 0, "difficulty")

        assert (adjusted.lower, adjusted.upper) != pytest.approx(
            (baseline.lower, baseline.upper)
        )

    def test_marginal_likelihood_ignores_negative_missing_codes(self):
        two_item = TwoParameterLogistic(2)
        two_item.set_parameters(
            discrimination=np.array([1.1, 2.0]),
            difficulty=np.array([-0.2, 1.0]),
        )
        one_item = TwoParameterLogistic(1)
        one_item.set_parameters(
            discrimination=np.array([1.1]),
            difficulty=np.array([-0.2]),
        )
        two_item_responses = np.array([[1, -1], [0, -1], [1, -1]])
        one_item_responses = two_item_responses[:, :1]
        quadrature = GaussHermiteQuadrature(n_points=11)

        actual = _marginal_log_likelihood(
            two_item,
            two_item_responses,
            quadrature.nodes,
            quadrature.weights,
        )
        expected = _marginal_log_likelihood(
            one_item,
            one_item_responses,
            quadrature.nodes,
            quadrature.weights,
        )

        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"param_idx": -1}, MirtValidationError),
            ({"param_idx": 2}, MirtValidationError),
            ({"param_idx": True}, MirtValidationError),
            ({"param_idx": 0, "param_name": "guessing"}, MirtValidationError),
            ({"param_idx": 0, "alpha": 0.0}, MirtValidationError),
            ({"param_idx": 0, "max_iter": 0}, MirtValidationError),
            ({"param_idx": 0, "tol": 0.0}, MirtValidationError),
            ({"param_idx": 0, "n_quadpts": 0}, MirtValidationError),
            (
                {"param_idx": 0, "profile_log_likelihood": 3.0},
                MirtValidationError,
            ),
        ],
    )
    def test_rejects_invalid_configuration(
        self, two_pl, response_matrix, kwargs, error
    ):
        with pytest.raises(error):
            PLCI(two_pl, response_matrix, **kwargs)

    def test_rejects_item_with_no_observed_responses(self, two_pl):
        with pytest.raises(MirtDataError, match="no observed"):
            PLCI(two_pl, np.array([[1, -1], [0, -1]]), 1, "difficulty")

    def test_rejects_fixed_discrimination(self, response_matrix):
        model = OneParameterLogistic(2)

        with pytest.raises(MirtValidationError, match="fixed"):
            PLCI(model, response_matrix, 0, "discrimination")


class TestScoreIntervals:
    def test_wald_interval_matches_information_formula(self, two_pl):
        theta = 0.25
        lower, upper = score_CI(two_pl, theta, alpha=0.1)
        information = float(np.sum(two_pl.information(np.array([[theta]]))))
        margin = stats.norm.ppf(0.95) / np.sqrt(information)

        assert lower == pytest.approx(theta - margin)
        assert upper == pytest.approx(theta + margin)

    def test_likelihood_interval_ignores_missing_items(self):
        two_item = TwoParameterLogistic(2)
        two_item.set_parameters(
            discrimination=np.array([1.1, 3.0]),
            difficulty=np.array([-0.2, 2.0]),
        )
        one_item = TwoParameterLogistic(1)
        one_item.set_parameters(
            discrimination=np.array([1.1]),
            difficulty=np.array([-0.2]),
        )

        two_bounds = score_CI(two_item, 0.0, [1, -1], method="likelihood")
        one_bounds = score_CI(one_item, 0.0, [1], method="likelihood")

        assert two_bounds[0] == pytest.approx(one_bounds[0])
        assert np.isposinf(two_bounds[1])
        assert np.isposinf(one_bounds[1])

    def test_likelihood_interval_respects_guessing(self):
        no_guessing = ThreeParameterLogistic(2)
        no_guessing.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            difficulty=np.array([-0.3, 0.4]),
            guessing=np.array([0.0, 0.0]),
        )
        guessing = ThreeParameterLogistic(2)
        guessing.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            difficulty=np.array([-0.3, 0.4]),
            guessing=np.array([0.4, 0.3]),
        )

        baseline = score_CI(no_guessing, 0.0, [1, 0], method="likelihood")
        adjusted = score_CI(guessing, 0.0, [1, 0], method="likelihood")

        assert adjusted != pytest.approx(baseline)

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"method": "typo"}, MirtValidationError),
            ({"method": "likelihood"}, MirtValidationError),
            ({"alpha": 0.0}, MirtValidationError),
            ({"alpha": 1.0}, MirtValidationError),
            ({"theta": np.nan}, MirtValidationError),
            ({"max_iter": 0}, MirtValidationError),
            ({"tol": 0.0}, MirtValidationError),
        ],
    )
    def test_rejects_invalid_configuration(self, two_pl, kwargs, error):
        options = dict(kwargs)
        theta = options.pop("theta", 0.0)
        with pytest.raises(error):
            score_CI(two_pl, theta, **options)

    @pytest.mark.parametrize("responses", [[1], [1, 2], [[1, 0]], [-1, -1]])
    def test_rejects_invalid_likelihood_patterns(self, two_pl, responses):
        with pytest.raises((MirtDataError, MirtValidationError)):
            score_CI(two_pl, 0.0, responses, method="likelihood")

    def test_rejects_zero_information(self):
        model = TwoParameterLogistic(2)
        model.set_parameters(
            discrimination=np.zeros(2),
            difficulty=np.zeros(2),
        )

        with pytest.raises(MirtEstimationError, match="positive test information"):
            score_CI(model, 0.0)

    @pytest.mark.parametrize(
        "model",
        [TwoParameterLogistic(2, n_factors=2), GradedResponseModel(2, 3)],
    )
    def test_rejects_unsupported_models(self, model):
        with pytest.raises(MirtValidationError):
            score_CI(model, 0.0)


class TestDeltaMethod:
    def test_matches_analytic_gradient(self):
        estimates = np.array([2.0, 3.0])
        covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

        transformed, standard_error = delta_method(
            estimates,
            covariance,
            lambda values: values[0] * values[1],
        )

        gradient = np.array([3.0, 2.0])
        expected_se = np.sqrt(gradient @ covariance @ gradient)
        assert transformed == pytest.approx(6.0)
        assert standard_error == pytest.approx(expected_se)

    @pytest.mark.parametrize(
        ("vcov", "match"),
        [
            (np.eye(3), "shape"),
            (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
            (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
            (np.array([[1.0, 2.0], [2.0, 1.0]]), "semidefinite"),
        ],
    )
    def test_rejects_invalid_covariance(self, vcov, match):
        with pytest.raises(MirtValidationError, match=match):
            delta_method([1.0, 2.0], vcov, np.sum)

    @pytest.mark.parametrize(
        ("estimates", "transform", "eps", "match"),
        [
            ([], np.sum, 1e-6, "estimates"),
            ([1.0, np.nan], np.sum, 1e-6, "estimates"),
            ([1.0, 2.0], None, 1e-6, "callable"),
            ([1.0, 2.0], lambda values: values, 1e-6, "scalar"),
            ([1.0, 2.0], lambda values: np.nan, 1e-6, "finite"),
            ([1.0, 2.0], np.sum, 0.0, "eps"),
        ],
    )
    def test_rejects_invalid_delta_inputs(self, estimates, transform, eps, match):
        with pytest.raises(MirtValidationError, match=match):
            delta_method(estimates, np.eye(2), transform, eps=eps)
