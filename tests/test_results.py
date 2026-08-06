"""Contract tests for fitted-model and person-score results."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.results import FitResult, ScoreResult


def _fit_result(
    *,
    standard_errors: dict[str, np.ndarray] | None = None,
    **overrides: Any,
) -> FitResult:
    model = TwoParameterLogistic(
        n_items=3,
        item_names=["item-a", "item-b", "item-c"],
    )
    model.set_parameters(
        discrimination=np.array([10.0, 1.2, 0.8]),
        difficulty=np.array([-1.0, 0.0, 1.0]),
    )
    values: dict[str, Any] = {
        "model": model,
        "log_likelihood": -123.5,
        "n_iterations": 12,
        "converged": True,
        "standard_errors": standard_errors
        if standard_errors is not None
        else {
            "discrimination": np.array([1.0, 0.2, 0.1]),
            "difficulty": np.array([0.2, 0.25, 0.3]),
        },
        "aic": 255.0,
        "bic": 270.0,
        "n_observations": 250,
        "n_parameters": 6,
    }
    values.update(overrides)
    return FitResult(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_iterations", -1),
        ("n_iterations", 1.5),
        ("n_iterations", True),
        ("n_observations", -1),
        ("n_parameters", -1),
    ],
)
def test_fit_result_validates_count_metadata(field: str, value: Any) -> None:
    with pytest.raises(MirtValidationError, match="non-negative integer"):
        _fit_result(**{field: value})


def test_fit_result_validates_convergence_metadata() -> None:
    with pytest.raises(MirtValidationError, match="converged must be a boolean"):
        _fit_result(converged=1)

    result = _fit_result(converged=np.bool_(True))
    assert result.converged is True


def test_fit_result_validates_standard_errors() -> None:
    with pytest.raises(MirtValidationError, match="must match its parameter shape"):
        _fit_result(standard_errors={"difficulty": np.ones(2)})
    with pytest.raises(MirtValidationError, match="cannot be negative"):
        _fit_result(standard_errors={"difficulty": np.array([0.1, -0.1, 0.2])})
    with pytest.raises(MirtValidationError, match="cannot be negative"):
        _fit_result(standard_errors={"difficulty": np.array([0.1, -np.inf, 0.2])})


def test_fit_result_copies_standard_error_arrays() -> None:
    errors = np.array([0.1, 0.2, 0.3])
    result = _fit_result(standard_errors={"difficulty": errors})

    errors[:] = 99.0
    assert np.allclose(result.standard_errors["difficulty"], [0.1, 0.2, 0.3])


def test_parameter_statistics_are_vectorized_and_tail_stable() -> None:
    result = _fit_result()
    statistics = result.parameter_statistics()

    discrimination = statistics["discrimination"]
    assert discrimination["p_value"][0] > 0.0
    assert discrimination["p_value"][0] < 1e-20
    assert discrimination["z"][0] == pytest.approx(10.0)
    assert discrimination["ci_lower"][1] < 1.2 < discrimination["ci_upper"][1]
    assert discrimination["estimate"].shape == (3,)


def test_missing_standard_errors_remain_unknown() -> None:
    result = _fit_result(standard_errors={})
    statistics = result.parameter_statistics()

    assert np.isnan(statistics["difficulty"]["standard_error"]).all()
    assert np.isnan(statistics["difficulty"]["z"]).all()
    assert np.isnan(statistics["difficulty"]["p_value"]).all()
    assert "nan" in result.summary().lower()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf, True, "bad"])
def test_fit_result_rejects_invalid_alpha(alpha: Any) -> None:
    result = _fit_result()
    with pytest.raises(MirtValidationError, match="0 < alpha < 1"):
        result.parameter_statistics(alpha)
    with pytest.raises(MirtValidationError, match="0 < alpha < 1"):
        result.summary(alpha)


def test_fit_result_confidence_intervals_are_defensive() -> None:
    result = _fit_result()
    intervals = result.confidence_intervals(alpha=0.10)
    lower, upper = intervals["difficulty"]

    assert lower.shape == (3,)
    assert np.all(lower < result.model.parameters["difficulty"])
    assert np.all(upper > result.model.parameters["difficulty"])
    lower[:] = 100.0
    assert not np.all(result.confidence_intervals()["difficulty"][0] == 100.0)


def test_summary_supports_higher_dimensional_parameters() -> None:
    from mirt.models.polytomous import NominalResponseModel

    model = NominalResponseModel(n_items=2, n_categories=3, n_factors=2)
    result = FitResult(
        model=model,
        log_likelihood=-10.0,
        n_iterations=1,
        converged=True,
        standard_errors={},
        aic=30.0,
        bic=35.0,
        n_observations=20,
        n_parameters=model.n_parameters,
    )

    summary = result.summary()
    assert "Item_0[1,1]" in summary
    assert "slopes" in summary
    with pytest.raises(MirtValidationError, match="wide coefficient output"):
        result.coef()


def test_coefficient_tables_use_nan_for_missing_uncertainty() -> None:
    result = _fit_result(standard_errors={})
    coefficients = result.coef_with_se()

    difficulty_se = coefficients["difficulty_se"].to_numpy()
    assert np.isnan(difficulty_se).all()
    assert len(coefficients) == result.model.n_items


def test_fit_result_dictionary_is_json_compatible() -> None:
    result = _fit_result()
    payload = result.to_dict()

    assert payload["model"]["name"] == "2PL"
    assert payload["model"]["item_names"] == ["item-a", "item-b", "item-c"]
    assert payload["parameters"]["difficulty"] == [-1.0, 0.0, 1.0]
    assert payload["n_observations"] == 250
    json.dumps(payload)

    compact = result.to_dict(
        include_parameters=False,
        include_standard_errors=False,
    )
    assert "parameters" not in compact
    assert "standard_errors" not in compact


def test_fit_statistics_preserve_scalar_types() -> None:
    statistics = _fit_result().fit_statistics()

    assert isinstance(statistics["converged"], bool)
    assert isinstance(statistics["n_iterations"], int)
    assert isinstance(statistics["log_likelihood"], float)


@pytest.mark.parametrize(
    ("theta", "standard_error", "message"),
    [
        (np.zeros((2, 1, 1)), np.zeros((2, 1, 1)), "one- or two-dimensional"),
        (np.zeros((2, 0)), np.zeros((2, 0)), "at least one factor"),
        (np.zeros(2), np.zeros(1), "same shape"),
        (np.zeros(2), np.array([0.1, -0.1]), "negative values"),
        (np.zeros(2), np.array([0.1, -np.inf]), "negative values"),
    ],
)
def test_score_result_validates_arrays(
    theta: np.ndarray,
    standard_error: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        ScoreResult(theta, standard_error, "EAP")


@pytest.mark.parametrize("method", ["", "   ", None, 1])
def test_score_result_validates_method(method: Any) -> None:
    with pytest.raises(MirtValidationError, match="non-empty string"):
        ScoreResult(np.zeros(2), np.ones(2), method)


def test_score_result_validates_and_normalizes_person_ids() -> None:
    result = ScoreResult(
        np.array([0.0, 1.0]),
        np.array([0.2, 0.3]),
        "EAP",
        np.array([10, 11]),
    )
    assert result.person_ids == [10, 11]

    with pytest.raises(MirtValidationError, match="one identifier per score row"):
        ScoreResult(np.zeros(2), np.ones(2), "EAP", ["only-one"])
    with pytest.raises(MirtValidationError, match="one-dimensional"):
        ScoreResult(np.zeros(2), np.ones(2), "EAP", np.array([[1], [2]]))


def test_score_result_normalizes_method_whitespace() -> None:
    result = ScoreResult(np.zeros(2), np.ones(2), "  EAP  ")
    assert result.method == "EAP"


def test_score_result_copies_inputs() -> None:
    theta = np.array([0.0, 1.0])
    standard_error = np.array([0.2, 0.3])
    person_ids = ["a", "b"]
    result = ScoreResult(theta, standard_error, "EAP", person_ids)

    theta[:] = 99.0
    standard_error[:] = 99.0
    person_ids[0] = "changed"
    assert np.allclose(result.theta, [0.0, 1.0])
    assert np.allclose(result.standard_error, [0.2, 0.3])
    assert result.person_ids == ["a", "b"]


def test_score_confidence_intervals_for_multiple_factors() -> None:
    result = ScoreResult(
        theta=np.array([[0.0, 1.0], [0.5, -0.5]]),
        standard_error=np.full((2, 2), 0.2),
        method="EAP",
    )
    lower, upper = result.confidence_intervals(alpha=0.05)

    assert lower.shape == result.theta.shape
    assert upper.shape == result.theta.shape
    assert np.all(lower < result.theta)
    assert np.all(upper > result.theta)
    with pytest.raises(MirtValidationError, match="0 < alpha < 1"):
        result.confidence_intervals(0.0)


def test_score_dataframe_accepts_numpy_person_ids() -> None:
    result = ScoreResult(
        np.array([0.0, 1.0]),
        np.array([0.2, 0.3]),
        "EAP",
        np.array([10, 11]),
    )
    dataframe = result.to_dataframe()

    assert len(dataframe) == 2
    if "person" in dataframe.columns:
        assert dataframe["person"].to_list() == [10, 11]
    else:
        assert list(dataframe.index) == [10, 11]
        assert dataframe.index.name == "person"


def test_score_array_and_dictionary_are_defensive() -> None:
    result = ScoreResult(
        np.array([[0.0, 1.0], [0.5, -0.5]]),
        np.full((2, 2), 0.2),
        "MAP",
        ["p1", "p2"],
    )

    combined = result.to_array(include_se=True)
    assert combined.shape == (2, 4)
    combined[:] = 99.0
    assert not np.all(result.theta == 99.0)

    payload = result.to_dict()
    assert payload["n_persons"] == 2
    assert payload["n_factors"] == 2
    assert payload["person_ids"] == ["p1", "p2"]
    json.dumps(payload)


def test_score_summary_handles_multiple_factors_and_empty_results() -> None:
    result = ScoreResult(
        np.array([[0.0, 1.0], [0.5, -0.5]]),
        np.full((2, 2), 0.2),
        "WLE",
    )
    summary = result.summary()
    assert "Method: WLE" in summary
    assert "Factors: 2" in summary
    assert "Mean SE" in summary

    empty = ScoreResult(np.empty(0), np.empty(0), "EAP")
    assert "Persons: 0" in empty.summary()
