"""Regression tests for reliability utilities."""

import numpy as np
import pytest

import mirt
from mirt import _information
from mirt.exceptions import MirtValidationError
from mirt.models import GradedResponseModel, MultidimensionalModel, TwoParameterLogistic
from mirt.results.fit_result import FitResult
from mirt.results.score_result import ScoreResult
from mirt.utils.reliability import conditional_rxx, empirical_rxx, marginal_rxx, sem


@pytest.fixture
def logistic_model() -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=3)
    model.set_parameters(
        discrimination=np.array([0.8, 1.2, 1.5]),
        difficulty=np.array([-0.5, 0.0, 0.75]),
    )
    return model


@pytest.fixture
def graded_model() -> GradedResponseModel:
    model = GradedResponseModel(n_items=3, n_categories=[3, 4, 5])
    model.set_parameters(
        discrimination=np.array([0.8, 1.2, 1.5]),
        thresholds=np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-1.5, -0.25, 1.0, 0.0],
                [-2.0, -0.75, 0.5, 1.5],
            ]
        ),
    )
    return model


def test_sem_supports_item_and_test_information_shapes(
    logistic_model: TwoParameterLogistic,
    graded_model: GradedResponseModel,
) -> None:
    theta = np.array([-1.0, 0.0, 1.0])

    expected_logistic = 1.0 / np.sqrt(logistic_model.information(theta).sum(axis=1))
    expected_graded = 1.0 / np.sqrt(graded_model.information(theta))

    np.testing.assert_allclose(sem(logistic_model, theta), expected_logistic)
    np.testing.assert_allclose(sem(graded_model, theta), expected_graded)


def test_sem_accepts_one_multidimensional_point() -> None:
    model = MultidimensionalModel(n_items=4, n_factors=2)

    result = sem(model, [0.25, -0.5])

    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_zero_information_has_unbounded_error_and_zero_reliability(
    logistic_model: TwoParameterLogistic,
) -> None:
    theta = np.array([1e6, 2e6])
    np.testing.assert_array_equal(logistic_model.information(theta), 0.0)

    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        standard_errors = sem(logistic_model, theta)
        empirical = empirical_rxx(logistic_model, theta, method="information")
        marginal = marginal_rxx(
            logistic_model,
            theta_range=(1e6, 2e6),
            n_points=3,
            density="uniform",
        )

    assert np.all(np.isposinf(standard_errors))
    assert empirical == 0.0
    assert marginal == 0.0


def test_marginal_reliability_supports_polytomous_models(
    graded_model: GradedResponseModel,
) -> None:
    reliability = marginal_rxx(graded_model)

    assert 0.0 < reliability < 1.0


def test_marginal_reliability_uses_density_variance(
    logistic_model: TwoParameterLogistic,
) -> None:
    narrow = marginal_rxx(
        logistic_model,
        theta_range=(-1.0, 1.0),
        density="uniform",
    )
    wide = marginal_rxx(
        logistic_model,
        theta_range=(-3.0, 3.0),
        density="uniform",
    )

    assert wide > narrow


def test_marginal_reliability_accepts_callable_density(
    logistic_model: TwoParameterLogistic,
) -> None:
    result = marginal_rxx(
        logistic_model,
        density=lambda theta: np.ones_like(theta),
    )

    assert 0.0 < result < 1.0
    assert result == pytest.approx(marginal_rxx(logistic_model, density="uniform"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"theta_range": (1.0, -1.0)}, "lower < upper"),
        ({"theta_range": (0.0,)}, "exactly two"),
        ({"n_points": 1}, "greater than or equal to 2"),
        ({"density": "other"}, "density must be"),
        ({"density": lambda theta: np.zeros_like(theta)}, "positive sum"),
        ({"density": lambda theta: -np.ones_like(theta)}, "non-negative"),
    ],
)
def test_marginal_reliability_validates_arguments(
    logistic_model: TwoParameterLogistic,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        marginal_rxx(logistic_model, **kwargs)  # type: ignore[arg-type]


def test_marginal_reliability_rejects_multidimensional_models() -> None:
    model = MultidimensionalModel(n_items=4, n_factors=2)

    with pytest.raises(ValueError, match="unidimensional"):
        marginal_rxx(model)


def test_information_reliability_supports_polytomous_models(
    graded_model: GradedResponseModel,
) -> None:
    theta = np.linspace(-1.5, 1.5, 20)

    result = empirical_rxx(graded_model, theta, method="information")

    assert 0.0 < result < 1.0


def test_posterior_reliability_uses_score_standard_errors(
    logistic_model: TwoParameterLogistic,
) -> None:
    theta = np.array([-1.0, 0.0, 1.0])
    standard_errors = np.full(3, 0.5)

    result = empirical_rxx(
        logistic_model,
        theta,
        method="posterior_variance",
        standard_errors=standard_errors,
    )

    assert result == pytest.approx(0.8)


def test_posterior_reliability_returns_each_factor() -> None:
    model = MultidimensionalModel(n_items=4, n_factors=2)
    theta = np.array([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    standard_errors = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])

    result = empirical_rxx(
        model,
        theta,
        method="posterior_variance",
        standard_errors=standard_errors,
    )

    np.testing.assert_allclose(result, np.array([0.8, 0.8]))


@pytest.mark.parametrize(
    ("theta", "method", "standard_errors", "message"),
    [
        ([0.0], "information", None, "at least two"),
        ([0.0, np.nan], "information", None, "finite"),
        ([0.0, 1.0], "unknown", None, "method must be"),
        ([0.0, 1.0], "posterior_variance", None, "standard_errors are required"),
        (
            [0.0, 1.0],
            "posterior_variance",
            [0.5],
            "standard_errors has shape",
        ),
        (
            [0.0, 1.0],
            "posterior_variance",
            [0.5, -0.5],
            "non-negative",
        ),
    ],
)
def test_empirical_reliability_validates_arguments(
    logistic_model: TwoParameterLogistic,
    theta: list[float],
    method: str,
    standard_errors: list[float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        empirical_rxx(
            logistic_model,
            theta,
            method=method,  # type: ignore[arg-type]
            standard_errors=standard_errors,
        )


def test_information_reliability_rejects_multidimensional_models() -> None:
    model = MultidimensionalModel(n_items=4, n_factors=2)

    with pytest.raises(ValueError, match="factor-specific"):
        empirical_rxx(model, [[-1.0, -0.5], [1.0, 0.5]], method="information")


def test_empirical_reliability_handles_zero_variance(
    logistic_model: TwoParameterLogistic,
) -> None:
    assert empirical_rxx(logistic_model, [0.0, 0.0, 0.0], method="information") == 0.0


@pytest.mark.parametrize("scale", [1e-308, 1e-160, 1e-8, 1.0, 1e160, 1e308])
def test_posterior_reliability_is_independent_of_score_units(
    logistic_model: TwoParameterLogistic, scale: float
) -> None:
    theta = np.array([-1.0, 0.0, 1.0]) * scale
    errors = np.full(3, 0.5 * scale)
    theta.flags.writeable = errors.flags.writeable = False

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = empirical_rxx(
            logistic_model, theta, method="posterior_variance", standard_errors=errors
        )

    assert actual == pytest.approx(0.8, abs=1e-14)
    np.testing.assert_array_equal(theta, np.array([-1.0, 0.0, 1.0]) * scale)
    np.testing.assert_array_equal(errors, 0.5 * scale)


def test_posterior_reliability_scales_each_factor_separately() -> None:
    model = MultidimensionalModel(n_items=4, n_factors=2)
    theta = np.array([[-1e-300, -1e300], [0.0, 0.0], [1e-300, 1e300]], order="F")
    errors = np.full_like(theta, [0.5e-300, 1e300])
    theta.flags.writeable = errors.flags.writeable = False

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = empirical_rxx(
            model, theta, method="posterior_variance", standard_errors=errors
        )

    np.testing.assert_allclose(actual, [0.8, 0.5], atol=1e-14)


@pytest.mark.parametrize("offset", [1e15, 1e150, -1e150])
def test_posterior_reliability_preserves_small_differences_at_large_offsets(
    logistic_model: TwoParameterLogistic, offset: float
) -> None:
    unit = abs(np.spacing(offset))
    theta = offset + np.array([-3.0, -1.0, 0.0, 2.0, 3.0]) * unit
    errors = np.full(theta.size, unit)
    reference_variance = np.var((theta - offset) / unit, ddof=1)
    expected = reference_variance / (reference_variance + 1.0)

    actual = empirical_rxx(
        logistic_model, theta, method="posterior_variance", standard_errors=errors
    )

    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("error", [0.0, 1e-300, 1e308])
def test_constant_extreme_scores_have_zero_posterior_reliability(
    logistic_model: TwoParameterLogistic, error: float
) -> None:
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = empirical_rxx(
            logistic_model,
            np.full(3, np.finfo(float).max),
            method="posterior_variance",
            standard_errors=np.full(3, error),
        )

    assert actual == 0.0


@pytest.mark.parametrize("scale,information", [(1e-8, 1e20), (1e160, 1e-320)])
def test_information_reliability_avoids_overflowing_error_variances(
    logistic_model: TwoParameterLogistic,
    monkeypatch: pytest.MonkeyPatch,
    scale: float,
    information: float,
) -> None:
    monkeypatch.setattr(
        logistic_model, "information", lambda theta: np.full(len(theta), information)
    )
    # Information-based errors are those of ML estimates, whose variance
    # already contains them (R's T_as_X = TRUE).
    expected = max(0.0, 1.0 - (1.0 / np.sqrt(information) / scale) ** 2)

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = empirical_rxx(
            logistic_model, np.array([-1.0, 0.0, 1.0]) * scale, method="information"
        )

    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("density_scale", [np.nextafter(0.0, 1.0), 1e-308, 1e308])
def test_marginal_reliability_is_independent_of_density_scale(
    logistic_model: TwoParameterLogistic, density_scale: float
) -> None:
    density = np.full(61, density_scale)
    density.flags.writeable = False
    expected = marginal_rxx(logistic_model, density="uniform")

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = marginal_rxx(logistic_model, density=lambda theta: density)

    assert actual == pytest.approx(expected, abs=1e-14)
    np.testing.assert_array_equal(density, density_scale)


@pytest.mark.parametrize(
    "scale,information", [(1e-8, 1e16), (1e160, 1e-320), (1e-160, 1e308), (1e308, 1.0)]
)
def test_marginal_reliability_preserves_extreme_positive_variances(
    logistic_model: TwoParameterLogistic,
    monkeypatch: pytest.MonkeyPatch,
    scale: float,
    information: float,
) -> None:
    monkeypatch.setattr(
        logistic_model, "information", lambda theta: np.full(len(theta), information)
    )
    grid = np.linspace(-1.0, 1.0, 61)
    weights = np.ones(61)
    weights[[0, -1]] = 0.5
    weights /= weights.sum()
    if scale == 1e308:
        expected = 1.0
    else:
        product = np.dot(weights, grid**2) * (scale * np.sqrt(information)) ** 2
        expected = product / (1.0 + product)

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = marginal_rxx(
            logistic_model, theta_range=(-scale, scale), density="uniform"
        )

    assert actual == pytest.approx(expected, rel=2e-13, abs=0.0)


def test_normal_density_in_distant_tail_does_not_underflow(
    logistic_model: TwoParameterLogistic, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        logistic_model, "information", lambda theta: np.ones(len(theta))
    )
    theta = np.linspace(40.0, 41.0, 61)
    weights = np.exp(-0.5 * (theta**2 - theta[0] ** 2))
    weights[[0, -1]] *= 0.5
    weights /= weights.sum()
    mean = np.dot(weights, theta)
    variance = np.dot(weights, (theta - mean) ** 2)

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        actual = marginal_rxx(logistic_model, theta_range=(40.0, 41.0))

    assert actual == pytest.approx(variance / (1.0 + variance), rel=2e-13)


@pytest.mark.parametrize("model_name", ["logistic_model", "graded_model"])
@pytest.mark.parametrize("statistic", ["sem", "conditional", "empirical"])
def test_information_evaluation_uses_bounded_blocks_without_changing_results(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    statistic: str,
) -> None:
    model = request.getfixturevalue(model_name)
    theta = np.linspace(-3.0, 3.0, 53)
    original = theta.copy()
    theta.flags.writeable = False
    information_function = model.information
    information = information_function(theta)
    if information.ndim == 2:
        information = information.sum(axis=1)
    calls = []

    def tracked_information(points):
        calls.append(points.copy())
        assert len(points) * model.n_items <= 23
        return information_function(points)

    monkeypatch.setattr(_information, "_INFORMATION_CHUNK_ELEMENTS", 23)
    monkeypatch.setattr(model, "information", tracked_information)
    if statistic == "sem":
        actual = sem(model, theta)
        expected = 1.0 / np.sqrt(information)
    elif statistic == "conditional":
        actual = conditional_rxx(model, theta, latent_variance=2.5)
        expected = information * 2.5 / (1.0 + information * 2.5)
    else:
        actual = empirical_rxx(model, theta, method="information")
        variance = np.var(theta, ddof=1)
        expected = 1.0 - np.mean(1.0 / information) / variance

    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(np.concatenate(calls)[:, 0], original)
    np.testing.assert_array_equal(theta, original)
    assert len(calls) > 1


@pytest.mark.parametrize(
    "values,message",
    [
        ([-1.0, 2.0, 3.0], "non-negative"),
        ([np.nan, 1.0, 2.0], "finite"),
        ([np.inf, 1.0, 2.0], "finite"),
        ([1e308, 1e308, 1e308], "total model information"),
        ([1.0, 2.0], "number of items"),
        ([], "number of items"),
    ],
)
def test_reliability_validates_item_information_before_reducing(
    logistic_model: TwoParameterLogistic,
    monkeypatch: pytest.MonkeyPatch,
    values: list[float],
    message: str,
) -> None:
    monkeypatch.setattr(
        logistic_model,
        "information",
        lambda theta: np.tile(values, (len(theta), 1)),
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError, match=message):
            sem(logistic_model, [-1.0, 0.0, 1.0])


def test_posterior_score_result_uses_r_empirical_rxx_formula(
    logistic_model: TwoParameterLogistic,
) -> None:
    theta = np.array([-1.2, -0.3, 0.1, 0.4, 1.0])
    errors = np.array([0.5, 0.4, 0.45, 0.4, 0.55])
    variance = np.var(theta, ddof=1)
    expected = variance / (variance + np.mean(errors**2))

    actual = empirical_rxx(logistic_model, ScoreResult(theta, errors, "EAP"))

    assert actual == pytest.approx(expected, abs=1e-14)
    assert actual == pytest.approx(
        empirical_rxx(logistic_model, theta, standard_errors=errors), abs=1e-14
    )


def test_ml_score_result_removes_error_variance_and_drops_infinite_rows(
    logistic_model: TwoParameterLogistic,
) -> None:
    # R's empirical_rxx(..., T_as_X = TRUE) after dropping non-finite scores.
    theta = np.array([-1.5, -0.4, 0.2, 0.9, 6.0])
    errors = np.array([0.6, 0.45, 0.4, 0.5, np.inf])
    finite = np.isfinite(errors)
    expected = 1.0 - np.mean(errors[finite] ** 2) / np.var(theta[finite], ddof=1)

    actual = empirical_rxx(logistic_model, ScoreResult(theta, errors, "ML"))

    assert actual == pytest.approx(expected, abs=1e-14)
    assert actual == pytest.approx(
        empirical_rxx(
            logistic_model,
            theta[finite],
            standard_errors=errors[finite],
            scores_include_error=True,
        ),
        abs=1e-14,
    )


def test_empirical_reliability_is_unbiased_for_eap_and_ml_scores() -> None:
    rng = np.random.default_rng(4)
    n_items = 15
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
    )
    model._is_fitted = True
    theta = rng.standard_normal(3000)
    responses = mirt.simdata(
        model="2PL",
        theta=theta,
        discrimination=model.parameters["discrimination"],
        difficulty=model.parameters["difficulty"],
        seed=1,
    )
    eap = mirt.fscores(model, responses, method="EAP")
    ml = mirt.fscores(model, responses, method="ML")
    finite = np.isfinite(ml.standard_error.ravel())
    eap_truth = np.corrcoef(theta, eap.theta.ravel())[0, 1] ** 2
    ml_truth = np.corrcoef(theta[finite], ml.theta.ravel()[finite])[0, 1] ** 2

    # The former information default gave 0.67 for EAP and 0.78 for ML
    # scores against true reliabilities of 0.73 and 0.72.
    assert empirical_rxx(model, eap) == pytest.approx(eap_truth, abs=0.02)
    assert empirical_rxx(model, ml) == pytest.approx(ml_truth, abs=0.02)
    assert empirical_rxx(
        model, ml.theta.ravel()[finite], method="information"
    ) == pytest.approx(ml_truth, abs=0.02)


def test_empirical_reliability_without_errors_warns_about_ml_assumption(
    logistic_model: TwoParameterLogistic,
) -> None:
    theta = np.array([-1.0, 0.0, 1.0])

    with pytest.warns(UserWarning, match="assumes ML or WLE"):
        implicit = empirical_rxx(logistic_model, theta)

    assert implicit == empirical_rxx(logistic_model, theta, method="information")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"standard_errors": [0.5, 0.5]}, "carries its own"),
        ({"scores_include_error": "yes"}, "must be boolean"),
    ],
)
def test_empirical_reliability_validates_score_result_options(
    logistic_model: TwoParameterLogistic,
    kwargs: dict[str, object],
    message: str,
) -> None:
    scores = ScoreResult(np.array([-1.0, 1.0]), np.array([0.5, 0.5]), "EAP")

    with pytest.raises(MirtValidationError, match=message):
        empirical_rxx(logistic_model, scores, **kwargs)  # type: ignore[arg-type]


def test_reliability_helpers_accept_fit_results(
    logistic_model: TwoParameterLogistic,
) -> None:
    fit = FitResult(logistic_model, -1.0, 1, True, {}, 4.0, 5.0)
    theta = np.array([-1.0, 0.0, 1.0])
    errors = np.full(3, 0.5)

    np.testing.assert_array_equal(sem(fit, theta), sem(logistic_model, theta))
    assert empirical_rxx(fit, theta, standard_errors=errors) == empirical_rxx(
        logistic_model, theta, standard_errors=errors
    )
    with pytest.raises(MirtValidationError, match="item model or a FitResult"):
        sem(np.ones(3), theta)  # type: ignore[arg-type]
