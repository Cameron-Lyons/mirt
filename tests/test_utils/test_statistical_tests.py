"""Regression tests for public hypothesis-testing utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt
from mirt._core import sigmoid
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import GradedResponseModel
from mirt.results import FitResult
from mirt.utils.statistical_tests import lagrange, likelihood_ratio, wald


def test_wald_uses_model_parameter_order_instead_of_alphabetical_order() -> None:
    model = MultidimensionalModel(2, n_factors=2)
    model.set_parameters(
        slopes=np.array([[1.0, 2.0], [3.0, 4.0]]),
        intercepts=np.array([10.0, 20.0]),
    )

    result = wald(
        model,
        param_indices=[0],
        constraint_values=[0.0],
        vcov=np.eye(6),
    )

    assert_allclose(result.parameter_estimates, [1.0])
    assert_allclose(result.standard_errors, [1.0])
    assert result.statistic == pytest.approx(1.0)
    assert result.df == 1
    assert result.p_value == pytest.approx(stats.chi2.sf(1.0, 1))


def test_wald_supports_general_linear_hypotheses() -> None:
    model = TwoParameterLogistic(2)
    model.set_parameters(
        discrimination=np.array([1.0, 2.0]),
        difficulty=np.array([3.0, 4.0]),
    )
    contrast = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )

    result = wald(
        model,
        constraint_values=[0.0, 0.0],
        vcov=np.eye(4) * 0.25,
        contrast_matrix=contrast,
    )

    assert_allclose(result.parameter_estimates, [-1.0, -1.0])
    assert_allclose(result.standard_errors, np.sqrt([0.5, 0.5]))
    assert result.statistic == pytest.approx(4.0)
    assert result.df == 2
    assert result.p_value == pytest.approx(stats.chi2.sf(4.0, 2))


def test_wald_accepts_a_single_vector_contrast() -> None:
    model = TwoParameterLogistic(2)
    contrast = np.array([1.0, -1.0, 0.0, 0.0])

    result = wald(
        model,
        vcov=np.eye(4),
        contrast_matrix=contrast,
    )

    assert result.df == 1
    assert_allclose(result.parameter_estimates, [0.0])
    assert_allclose(result.standard_errors, [np.sqrt(2.0)])


def test_wald_uses_model_covariance_when_available() -> None:
    model = TwoParameterLogistic(1)
    model.vcov = np.diag([0.25, 0.5])

    result = wald(model, [0], [0.0])

    assert result.statistic == pytest.approx(4.0)
    assert_allclose(result.standard_errors, [0.5])


def test_wald_inverts_model_information_when_available() -> None:
    model = TwoParameterLogistic(1)
    model.information_matrix = lambda: np.diag([4.0, 2.0])

    result = wald(model, [0], [0.0])

    assert result.statistic == pytest.approx(4.0)
    assert_allclose(result.standard_errors, [0.5])


def test_wald_reads_the_covariance_of_a_fit_result() -> None:
    data = mirt.simdata(model="2PL", n_persons=600, n_items=4, seed=5)
    result = mirt.fit_mirt(data, model="2PL", tol=1e-7)
    vcov = result.vcov
    estimates = np.concatenate([result.model.discrimination, result.model.difficulty])

    single = wald(result, param_indices=[5], constraint_values=[0.0])
    assert single.statistic == pytest.approx(estimates[5] ** 2 / vcov[5, 5])
    assert_allclose(single.standard_errors, [result.standard_errors["difficulty"][1]])

    contrast = np.zeros(8)
    contrast[[0, 4]] = [1.0, -1.0]
    combined = wald(result, contrast_matrix=contrast)
    variance = vcov[0, 0] + vcov[4, 4] - 2.0 * vcov[0, 4]
    assert combined.statistic == pytest.approx(
        (estimates[0] - estimates[4]) ** 2 / variance
    )


def test_wald_rejects_fixed_parameters_and_warns_without_covariance() -> None:
    data = mirt.simdata(model="2PL", n_persons=300, n_items=3, seed=6)
    rasch = mirt.fit_mirt(data, model="1PL")
    with pytest.raises(MirtValidationError, match="fixed"):
        wald(rasch, param_indices=[0], constraint_values=[1.0])

    errors_only = FitResult(rasch.model, -1.0, 1, True, rasch.standard_errors, 0.0, 0.0)
    with pytest.warns(UserWarning, match="squared standard errors"):
        test = wald(errors_only, param_indices=[3])
    difficulty_se = rasch.standard_errors["difficulty"][0]
    assert test.statistic == pytest.approx(
        (rasch.model.difficulty[0] / difficulty_se) ** 2
    )


def test_wald_requires_real_covariance_information() -> None:
    model = TwoParameterLogistic(1)

    with pytest.raises(ValueError, match="vcov is required"):
        wald(model, [0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "required"),
        ({"param_indices": []}, "non-empty"),
        ({"param_indices": [0.0]}, "integers"),
        ({"param_indices": [-1]}, "between"),
        ({"param_indices": [0, 0]}, "duplicates"),
        (
            {"param_indices": [0], "constraint_values": [0.0, 1.0]},
            "length 1",
        ),
        (
            {"param_indices": [0], "vcov": np.eye(3)},
            "shape",
        ),
        (
            {
                "param_indices": [0],
                "vcov": np.array([[1.0, 0.2], [0.0, 1.0]]),
            },
            "symmetric",
        ),
        (
            {
                "param_indices": [0],
                "vcov": np.array([[1.0, 2.0], [2.0, 1.0]]),
            },
            "positive semidefinite",
        ),
        (
            {
                "param_indices": [0],
                "contrast_matrix": np.array([1.0, 0.0]),
            },
            "mutually exclusive",
        ),
        (
            {"contrast_matrix": np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)},
            "linearly independent",
        ),
    ],
)
def test_wald_validates_hypothesis_inputs(kwargs: dict, message: str) -> None:
    model = TwoParameterLogistic(1)
    kwargs.setdefault("vcov", np.eye(2))

    with pytest.raises(ValueError, match=message):
        wald(model, **kwargs)


def _three_pl_with_bound_covariance() -> FitResult:
    model = ThreeParameterLogistic(3).set_parameters(
        discrimination=np.array([1.0, 1.4, 0.8]),
        difficulty=np.array([-0.5, 0.2, 0.9]),
        guessing=np.array([0.0, 0.2, 0.15]),
    )
    covariance = np.diag([0.04, 0.05, 0.03, 0.02, 0.03, 0.04, np.nan, 0.004, 0.009])
    covariance[0, 3] = covariance[3, 0] = 0.01
    # The first guessing coordinate sits on its lower bound.
    covariance[6, :] = covariance[:, 6] = np.nan
    return FitResult(model, -1.0, 1, True, {}, 0.0, 0.0, vcov=covariance)


def test_wald_accepts_the_nan_rows_of_a_fit_covariance() -> None:
    # Regression: wald(model, vcov=result.vcov) rejected the NaN rows that
    # mark parameters on an optimizer bound.
    result = _three_pl_with_bound_covariance()
    contrast = np.zeros(9)
    contrast[[0, 3, 7]] = [1.0, -1.0, 2.0]

    explicit = wald(result.model, contrast_matrix=contrast, vcov=result.vcov)
    fitted = wald(result, contrast_matrix=contrast)
    assert explicit.statistic == pytest.approx(fitted.statistic)
    assert_allclose(explicit.standard_errors, fitted.standard_errors)

    with pytest.raises(MirtValidationError, match="NaN"):
        wald(result.model, param_indices=[6], vcov=result.vcov)
    with pytest.raises(MirtValidationError, match="NaN"):
        lagrange(
            result.model,
            np.array([[0, 1, 1], [1, 0, 1]]),
            np.array([0.0, 1.0]),
            param_indices=[6],
            vcov=result.vcov,
        )
    vcov = result.vcov.copy()
    vcov[0, 1] = vcov[1, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        wald(result.model, param_indices=[0], vcov=vcov)


def _discrimination_fixed_fit(
    true_discrimination: float,
) -> tuple[FitResult, np.ndarray]:
    rng = np.random.default_rng(8)
    discrimination = np.array([true_discrimination, 1.0, 1.2, 0.8, 1.5])
    truth = TwoParameterLogistic(5).set_parameters(
        discrimination=discrimination, difficulty=rng.normal(0.0, 0.8, 5)
    )
    data = np.asarray(truth.simulate(rng.standard_normal((1200, 1)), seed=4))
    data[rng.random(data.shape) < 0.03] = -1
    constrained = mirt.fit_mirt(
        data,
        model="2PL",
        fixed={"discrimination": np.arange(5) == 0},
        start_values={"discrimination": np.ones(5)},
        tol=1e-8,
    )
    return constrained, data


def test_lagrange_scores_a_fit_result_with_the_marginal_likelihood() -> None:
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.standard_errors import (
        _finite_difference_information,
        _finite_difference_scores,
    )

    # Regression: lagrange(result, ...) raised a bare AttributeError.
    constrained, data = _discrimination_fixed_fit(2.0)
    test = lagrange(constrained, data, param_indices=[0])

    # Reference: free the discrimination and difference the marginal
    # log-likelihood of the freed model at the constrained estimates.
    freed = constrained.model.copy().set_free_parameter_masks(None)
    quadrature = GaussHermiteQuadrature(21)
    mass = quadrature.weights / quadrature.weights.sum()
    information, _ = _finite_difference_information(freed, data, quadrature, mass, 1e-4)
    scores, _ = _finite_difference_scores(freed, data, quadrature, mass, 1e-5)
    score = scores.sum(axis=0)
    expected = score[0] ** 2 * np.linalg.inv(information)[0, 0]

    assert_allclose(test.scores, score[:1], rtol=1e-6)
    assert test.statistic == pytest.approx(expected, rel=1e-5)
    assert test.df == 1
    assert test.p_value < 1e-3
    # Under the null the statistic is small, like the likelihood ratio.
    null, null_data = _discrimination_fixed_fit(1.0)
    full = mirt.fit_mirt(null_data, model="2PL", tol=1e-8)
    ratio, _ = likelihood_ratio(full.log_likelihood, null.log_likelihood, 1)
    statistic = lagrange(null, null_data, param_indices=[0]).statistic
    assert statistic == pytest.approx(ratio, abs=0.1)


def test_lagrange_validates_fit_result_hypotheses() -> None:
    constrained, data = _discrimination_fixed_fit(1.0)

    with pytest.raises(ValueError, match="theta is not used"):
        lagrange(constrained, data, np.zeros(data.shape[0]), [0])
    with pytest.raises(MirtValidationError, match="must name fixed parameters"):
        lagrange(constrained, data, param_indices=[1])
    rasch = mirt.fit_mirt(data, model="1PL", max_iter=50)
    with pytest.raises(MirtValidationError, match="model family fixes"):
        lagrange(rasch, data, param_indices=[0])
    with pytest.raises(ValueError, match="theta is required"):
        lagrange(constrained.model, data, param_indices=[0], vcov=np.eye(10))
    with pytest.raises(ValueError, match="param_indices is required"):
        lagrange(constrained, data)
    with pytest.raises(ValueError, match="n_quadpts must be a positive integer"):
        lagrange(constrained, data, param_indices=[0], n_quadpts=0)
    # The likelihood score of the free parameters is not zero at a posterior
    # mode, so the marginal score test needs a maximum-likelihood fit.
    bayes_modal = FitResult(
        constrained.model, -1.0, 1, True, {}, 0.0, 0.0, log_posterior=-2.0
    )
    with pytest.raises(MirtValidationError, match="maximum-likelihood fit"):
        lagrange(bayes_modal, data, param_indices=[0])


def test_lagrange_fit_result_accepts_an_explicit_covariance() -> None:
    constrained, data = _discrimination_fixed_fit(2.0)
    marginal = lagrange(constrained, data, param_indices=[0])

    explicit = lagrange(constrained, data, param_indices=[0], vcov=np.eye(10) * 0.5)
    assert_allclose(explicit.scores, marginal.scores)
    assert explicit.statistic == pytest.approx(0.5 * marginal.scores[0] ** 2)


def test_lagrange_rejects_indefinite_observed_information(monkeypatch) -> None:
    import mirt.estimation.standard_errors as standard_errors

    constrained, data = _discrimination_fixed_fit(1.0)
    original = standard_errors._score_and_information

    def indefinite(*args, **kwargs):
        score, information, layouts = original(*args, **kwargs)
        information = information.copy()
        information[0, 0] = -1.0
        return score, information, layouts

    monkeypatch.setattr(standard_errors, "_score_and_information", indefinite)
    with pytest.raises(MirtValidationError, match="not positive definite"):
        lagrange(constrained, data, param_indices=[0])


def test_lagrange_matches_analytic_two_parameter_scores_with_missing_data() -> None:
    model = TwoParameterLogistic(1)
    model.set_parameters(
        discrimination=np.array([1.2]),
        difficulty=np.array([0.3]),
    )
    original_parameters = model.parameters
    responses = np.array([[0.0], [1.0], [np.nan]])
    theta = np.array([-1.0, 0.5, 1.0])
    covariance = np.diag([0.2, 0.1])

    result = lagrange(
        model,
        responses,
        theta,
        param_indices=[0, 1],
        vcov=covariance,
    )

    observed_theta = theta[:2]
    probabilities = sigmoid(1.2 * (observed_theta - 0.3))
    residuals = responses[:2, 0] - probabilities
    expected_scores = np.array(
        [
            np.sum(residuals * (observed_theta - 0.3)),
            np.sum(residuals * -1.2),
        ]
    )
    expected_statistic = float(expected_scores @ covariance @ expected_scores)

    assert_allclose(result.scores, expected_scores, rtol=1e-8, atol=1e-9)
    assert result.statistic == pytest.approx(expected_statistic)
    assert result.p_value == pytest.approx(stats.chi2.sf(expected_statistic, 2))
    for name, values in original_parameters.items():
        assert_allclose(model.parameters[name], values)


def test_lagrange_accepts_one_person_multidimensional_theta() -> None:
    model = MultidimensionalModel(2, n_factors=2)
    original_parameters = model.parameters

    result = lagrange(
        model,
        responses=np.array([[1, 0]]),
        theta=np.array([0.1, 0.2]),
        param_indices=[0, 1],
        vcov=np.eye(6),
    )

    assert result.df == 2
    assert np.isfinite(result.statistic)
    assert np.all(np.isfinite(result.scores))
    for name, values in original_parameters.items():
        assert_allclose(model.parameters[name], values)


def test_lagrange_supports_polytomous_native_log_likelihood() -> None:
    model = GradedResponseModel(1, n_categories=3)

    result = lagrange(
        model,
        responses=np.array([[0], [1], [2]]),
        theta=np.array([-1.0, 0.0, 1.0]),
        param_indices=[0, 1, 2],
        vcov=np.eye(3),
    )

    assert result.df == 3
    assert np.isfinite(result.statistic)
    assert np.all(np.isfinite(result.scores))


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        (
            ThreeParameterLogistic(2),
            {
                "discrimination": np.array([1.2, 0.9]),
                "difficulty": np.array([0.3, -0.4]),
                "guessing": np.array([0.15, 0.2]),
            },
        ),
        (
            FourParameterLogistic(2),
            {
                "discrimination": np.array([1.2, 0.9]),
                "difficulty": np.array([0.3, -0.4]),
                "guessing": np.array([0.15, 0.2]),
                "upper": np.array([0.95, 0.9]),
            },
        ),
    ],
)
def test_lagrange_logistic_scores_match_finite_differences(
    model: ThreeParameterLogistic | FourParameterLogistic,
    parameters: dict[str, np.ndarray],
) -> None:
    model.set_parameters(**parameters)
    responses = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    theta = np.array([-1.5, -0.25, 0.5, 1.25])
    flattened = np.concatenate([values.ravel() for values in model.parameters.values()])
    indices = list(range(flattened.size))

    result = lagrange(
        model,
        responses,
        theta,
        param_indices=indices,
        vcov=np.eye(flattened.size),
    )

    expected = np.empty_like(flattened)
    step = 1e-6
    offset = 0
    original = model.parameters
    for name, values in original.items():
        for local_index in range(values.size):
            plus = {key: value.copy() for key, value in original.items()}
            minus = {key: value.copy() for key, value in original.items()}
            plus[name].flat[local_index] += step
            minus[name].flat[local_index] -= step
            model.set_parameters(**plus)
            likelihood_plus = float(np.sum(model.log_likelihood(responses, theta)))
            model.set_parameters(**minus)
            likelihood_minus = float(np.sum(model.log_likelihood(responses, theta)))
            expected[offset + local_index] = (likelihood_plus - likelihood_minus) / (
                2 * step
            )
        offset += values.size
    model.set_parameters(**original)

    assert_allclose(result.scores, expected, rtol=1e-7, atol=1e-8)


def test_lagrange_uses_compatible_model_score_function() -> None:
    model = TwoParameterLogistic(1)
    model.score_function = lambda responses, theta: np.array([[1.0, 2.0], [3.0, 4.0]])

    result = lagrange(
        model,
        responses=np.array([[0], [1]]),
        theta=np.array([-0.5, 0.5]),
        param_indices=[1],
        vcov=np.eye(2),
    )

    assert_allclose(result.scores, [6.0])
    assert result.statistic == pytest.approx(36.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"responses": np.array([0, 1])}, "2D matrix"),
        ({"responses": np.ones((2, 2))}, "1 items"),
        ({"responses": np.array([[0.5], [1.0]])}, "integer-valued"),
        ({"responses": np.array([[2], [1]])}, "dichotomous"),
        ({"theta": np.array([0.0])}, "theta must have shape"),
        ({"param_indices": [2]}, "between"),
        ({"step": 0.0}, "positive"),
        ({"vcov": np.zeros((2, 2))}, "positive definite"),
    ],
)
def test_lagrange_validates_inputs(kwargs: dict, message: str) -> None:
    model = TwoParameterLogistic(1)
    inputs = {
        "responses": np.array([[0], [1]]),
        "theta": np.array([-0.5, 0.5]),
        "param_indices": [0],
        "vcov": np.eye(2),
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        lagrange(model, **inputs)


def test_likelihood_ratio_uses_stable_survival_probability() -> None:
    statistic, p_value = likelihood_ratio(0.0, -50.0, 1)

    assert statistic == 100.0
    assert p_value == pytest.approx(stats.chi2.sf(100.0, 1))
    assert p_value > 0.0


def test_likelihood_ratio_tolerates_roundoff_at_zero() -> None:
    full = -100.0
    reduced = np.nextafter(full, np.inf)
    statistic, p_value = likelihood_ratio(full, reduced, 1)

    assert statistic == 0.0
    assert p_value == 1.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((np.nan, -2.0, 1), "finite"),
        ((-2.0, np.inf, 1), "finite"),
        ((-2.0, -3.0, 0), "positive"),
        ((-2.0, -3.0, 1.5), "integer"),
        ((-10.0, -9.0, 1), "at least as large"),
    ],
)
def test_likelihood_ratio_validates_nested_models(
    args: tuple[float, float, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        likelihood_ratio(*args)
