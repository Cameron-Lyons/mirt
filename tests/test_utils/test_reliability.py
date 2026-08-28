"""Regression tests for reliability utilities."""

import numpy as np
import pytest

from mirt.models import GradedResponseModel, MultidimensionalModel, TwoParameterLogistic
from mirt.utils.reliability import empirical_rxx, marginal_rxx, sem


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
        empirical = empirical_rxx(logistic_model, theta)
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

    result = empirical_rxx(graded_model, theta)

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
        empirical_rxx(model, [[-1.0, -0.5], [1.0, 0.5]])


def test_empirical_reliability_handles_zero_variance(
    logistic_model: TwoParameterLogistic,
) -> None:
    assert empirical_rxx(logistic_model, [0.0, 0.0, 0.0]) == 0.0
