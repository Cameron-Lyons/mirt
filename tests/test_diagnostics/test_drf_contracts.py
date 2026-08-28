from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, stats

from mirt.diagnostics.drf import (
    _compute_item_information,
    _compute_marginal_reliability,
    _compute_test_information,
    compute_drf,
    compute_item_drf,
    plot_drf,
    reliability_invariance,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel


class _InformationModel:
    def __init__(
        self,
        n_items: int,
        scale: float = 1.0,
        *,
        total_only: bool = False,
    ) -> None:
        self.n_items = n_items
        self.scale = scale
        self.total_only = total_only
        self.calls: list[int | None] = []

    def information(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        self.calls.append(item_idx)
        theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
        columns = (
            self.scale
            * (1.0 + theta_values[:, None] ** 2)
            * np.arange(1, self.n_items + 1, dtype=np.float64)
        )
        if item_idx is not None:
            return columns[:, item_idx]
        if self.total_only:
            return columns.sum(axis=1)
        return columns


class _ConstantInformationModel:
    def __init__(self, total_information: float, n_items: int = 2) -> None:
        self.n_items = n_items
        self.total_information = total_information

    def information(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        n_theta = np.asarray(theta).reshape(-1).size
        per_item = self.total_information / self.n_items
        if item_idx is not None:
            return np.full(n_theta, per_item)
        return np.full((n_theta, self.n_items), per_item)


def _fit_result(model: Any) -> SimpleNamespace:
    return SimpleNamespace(model=model)


def test_item_information_uses_matrix_result_without_per_item_calls() -> None:
    model = _InformationModel(4)
    theta = np.linspace(-2.0, 2.0, 9)

    information = _compute_item_information(model, theta)

    assert information.shape == (9, 4)
    assert model.calls == [None]


def test_item_information_recovers_columns_from_total_only_models() -> None:
    model = _InformationModel(3, total_only=True)
    theta = np.linspace(-2.0, 2.0, 7)

    information = _compute_item_information(model, theta)

    expected = (1.0 + theta[:, None] ** 2) * np.arange(
        1, model.n_items + 1, dtype=np.float64
    )
    assert_allclose(information, expected)
    assert model.calls == [None, 0, 1, 2]


@pytest.mark.parametrize("model_type", [GradedResponseModel, GeneralizedPartialCredit])
def test_item_information_supports_public_polytomous_models(
    model_type: type[GradedResponseModel] | type[GeneralizedPartialCredit],
) -> None:
    model = model_type(n_items=3, n_categories=[3, 4, 5])
    theta = np.linspace(-2.0, 2.0, 11)

    information = _compute_item_information(model, theta)

    expected = np.column_stack(
        [model.information(theta, item_idx=item) for item in range(model.n_items)]
    )
    assert_allclose(information, expected)


def test_compute_item_drf_supports_total_only_polytomous_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref_model = _InformationModel(3, scale=1.0, total_only=True)
    focal_model = _InformationModel(3, scale=1.5, total_only=True)

    def fake_fit(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert kwargs["model"] == "GRM"
        return _fit_result(ref_model), _fit_result(focal_model)

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", fake_fit)
    data = np.zeros((8, 3), dtype=np.int64)
    groups = np.array(["focal"] * 4 + ["reference"] * 4)

    result = compute_item_drf(
        data,
        groups,
        model="GRM",
        theta_range=(-1.0, 1.0),
        n_points=11,
        focal_group="focal",
    )

    theta = result["theta_grid"]
    expected_difference = (
        0.5 * (1.0 + theta[:, None] ** 2) * np.arange(1, 4, dtype=np.float64)
    )
    assert_allclose(
        result["item_drf"],
        integrate.trapezoid(expected_difference, theta, axis=0),
    )
    assert_allclose(result["info_diff_max"], expected_difference.max(axis=0))
    assert result["info_ref"].shape == (3, 11)
    assert result["info_focal"].shape == (3, 11)
    assert result["ref_group"] == "reference"
    assert result["focal_group"] == "focal"


def test_compute_item_drf_matches_scalar_integration_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(381)
    n_points = 101
    n_items = 37
    ref_values = rng.uniform(0.0, 4.0, size=(n_points, n_items))
    focal_values = rng.uniform(0.0, 4.0, size=(n_points, n_items))

    class FixedModel:
        def __init__(self, values: np.ndarray) -> None:
            self.n_items = values.shape[1]
            self.values = values

        def information(
            self, theta: np.ndarray, item_idx: int | None = None
        ) -> np.ndarray:
            if item_idx is None:
                return self.values
            return self.values[:, item_idx]

    def fake_fit(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return _fit_result(FixedModel(ref_values)), _fit_result(
            FixedModel(focal_values)
        )

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", fake_fit)
    result = compute_item_drf(
        np.zeros((4, n_items), dtype=np.int64),
        np.array([0, 0, 1, 1]),
        n_points=n_points,
    )
    theta = result["theta_grid"]
    difference = np.abs(ref_values - focal_values)
    scalar_reference = np.array(
        [integrate.trapezoid(difference[:, item], theta) for item in range(n_items)]
    )

    assert_allclose(result["item_drf"], scalar_reference, rtol=1e-14, atol=1e-14)


def test_marginal_reliability_uses_normalized_trapezoidal_quadrature() -> None:
    model = _InformationModel(1, scale=2.0)
    theta_range = (-1.5, 2.0)
    n_points = 8

    actual = _compute_marginal_reliability(model, theta_range, n_points=n_points)

    theta = np.linspace(*theta_range, n_points)
    weights = stats.norm.pdf(theta)
    information = 2.0 * (1.0 + theta**2)
    expected_error = integrate.trapezoid(
        weights / information, theta
    ) / integrate.trapezoid(weights, theta)
    assert actual == pytest.approx(1.0 - expected_error)


@pytest.mark.parametrize(
    ("information", "message"),
    [
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, -0.5]), "nonnegative"),
        (np.ones((2, 2, 1)), "shape"),
    ],
)
def test_test_information_rejects_invalid_model_output(
    information: np.ndarray, message: str
) -> None:
    class InvalidModel:
        n_items = 2

        def information(self, theta: np.ndarray) -> np.ndarray:
            return information

    with pytest.raises(ValueError, match=message):
        _compute_test_information(InvalidModel(), np.array([-1.0, 1.0]))


@pytest.mark.parametrize(
    "theta", [np.array([]), np.ones((2, 1)), np.array([0.0, np.nan])]
)
@pytest.mark.parametrize(
    "function", [_compute_test_information, _compute_item_information]
)
def test_information_helpers_validate_theta(function: Any, theta: np.ndarray) -> None:
    with pytest.raises(ValueError, match="theta"):
        function(_InformationModel(2), theta)


@pytest.mark.parametrize(
    ("theta_range", "n_points", "message"),
    [
        ((0.0, np.inf), 5, "theta_range"),
        ((1.0, 1.0), 5, "theta_range"),
        ((-1.0, 1.0), True, "n_points"),
        ((-1.0, 1.0), 1, "n_points"),
        ((1000.0, 1001.0), 5, "positive"),
    ],
)
def test_marginal_reliability_validates_quadrature(
    theta_range: tuple[float, float], n_points: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _compute_marginal_reliability(
            _ConstantInformationModel(4.0), theta_range, n_points=n_points
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "NRM"}, "model"),
        ({"n_points": 1}, "n_points"),
        ({"n_points": True}, "n_points"),
        ({"theta_range": (1.0, 1.0)}, "theta_range"),
        ({"theta_range": (-1.0, np.inf)}, "theta_range"),
    ],
)
def test_compute_drf_validates_configuration_before_fitting(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], message: str
) -> None:
    def unexpected_fit(*args: Any, **fit_kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        compute_drf(
            np.zeros((4, 2), dtype=np.int64),
            np.array([0, 0, 1, 1]),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("data", "groups", "message"),
    [
        (np.empty((0, 2)), np.array([]), "nonempty"),
        (np.zeros(4), np.array([0, 0, 1, 1]), "two-dimensional"),
        (np.zeros((4, 2)), np.array([[0], [0], [1], [1]]), "one-dimensional"),
        (np.zeros((4, 2)), np.array([0, 1]), "length"),
        (np.zeros((4, 2)), np.array([0.0, 0.0, 1.0, np.nan]), "missing"),
        (
            np.zeros((4, 2)),
            np.array(["a", "a", "b", None], dtype=object),
            "missing",
        ),
        (
            np.zeros((4, 2)),
            np.array(["a", "a", "b", np.nan], dtype=object),
            "missing",
        ),
    ],
)
def test_compute_drf_validates_data_before_fitting(
    monkeypatch: pytest.MonkeyPatch,
    data: np.ndarray,
    groups: np.ndarray,
    message: str,
) -> None:
    def unexpected_fit(*args: Any, **kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        compute_drf(data, groups)


def test_compute_item_drf_rejects_mismatched_group_model_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fit(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return _fit_result(_InformationModel(2)), _fit_result(_InformationModel(3))

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", fake_fit)
    with pytest.raises(ValueError, match="shapes must match"):
        compute_item_drf(
            np.zeros((4, 2), dtype=np.int64),
            np.array([0, 0, 1, 1]),
        )


def test_reliability_invariance_is_reproducible_and_reports_bootstrap_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_options: list[dict[str, Any]] = []

    def fake_fit(
        ref_data: np.ndarray,
        focal_data: np.ndarray,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        fit_options.append(kwargs)
        ref_information = 3.0 + float(np.mean(ref_data))
        focal_information = 3.0 + float(np.mean(focal_data))
        return _fit_result(_ConstantInformationModel(ref_information)), _fit_result(
            _ConstantInformationModel(focal_information)
        )

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", fake_fit)
    data = np.arange(48, dtype=np.int64).reshape(12, 4) % 3
    groups = np.array([0] * 6 + [1] * 6)
    options = {
        "n_bootstrap": 20,
        "seed": 991,
        "theta_range": (-2.0, 2.0),
        "n_points": 9,
        "confidence_level": 0.9,
        "max_iter": 7,
    }

    first = reliability_invariance(data, groups, **options)
    second = reliability_invariance(data, groups, **options)

    for key in (
        "reliability_ref",
        "reliability_focal",
        "reliability_diff",
        "reliability_diff_se",
        "z",
        "p_value",
    ):
        assert first[key] == pytest.approx(second[key])
    assert_allclose(first["reliability_diff_ci"], second["reliability_diff_ci"])
    assert first["n_bootstrap_successful"] == 20
    assert first["n_bootstrap_failed"] == 0
    assert first["reliability_diff_ci"].shape == (2,)
    assert 0.0 <= first["p_value"] <= 1.0
    assert all(option["max_iter"] == 7 for option in fit_options)


def test_reliability_invariance_counts_failed_bootstrap_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def sometimes_fails(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        if call_count > 1 and call_count % 3 == 0:
            raise RuntimeError("synthetic fit failure")
        return _fit_result(_ConstantInformationModel(4.0)), _fit_result(
            _ConstantInformationModel(5.0)
        )

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", sometimes_fails)
    result = reliability_invariance(
        np.zeros((8, 2), dtype=np.int64),
        np.array([0] * 4 + [1] * 4),
        n_bootstrap=8,
        seed=12,
    )

    assert result["n_bootstrap_successful"] == 5
    assert result["n_bootstrap_failed"] == 3
    assert result["n_bootstrap_successful"] + result["n_bootstrap_failed"] == 8


def test_reliability_invariance_handles_too_few_successful_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def bootstrap_fails(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("synthetic fit failure")
        return _fit_result(_ConstantInformationModel(4.0)), _fit_result(
            _ConstantInformationModel(5.0)
        )

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", bootstrap_fails)
    result = reliability_invariance(
        np.zeros((8, 2), dtype=np.int64),
        np.array([0] * 4 + [1] * 4),
        n_bootstrap=4,
        seed=12,
    )

    assert result["n_bootstrap_successful"] == 0
    assert result["n_bootstrap_failed"] == 4
    assert np.isnan(result["reliability_diff_se"])
    assert np.isnan(result["p_value"])


def test_reliability_invariance_can_skip_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_fit(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal calls
        calls += 1
        return _fit_result(_ConstantInformationModel(4.0)), _fit_result(
            _ConstantInformationModel(5.0)
        )

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", fake_fit)
    result = reliability_invariance(
        np.zeros((8, 2), dtype=np.int64),
        np.array([0] * 4 + [1] * 4),
        n_bootstrap=0,
    )

    assert calls == 1
    assert result["n_bootstrap_successful"] == 0
    assert result["n_bootstrap_failed"] == 0
    assert np.isnan(result["reliability_diff_se"])
    assert np.isnan(result["p_value"])
    assert_array_equal(np.isnan(result["reliability_diff_ci"]), [True, True])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bootstrap": -1}, "n_bootstrap"),
        ({"n_bootstrap": True}, "n_bootstrap"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": np.nan}, "confidence_level"),
    ],
)
def test_reliability_invariance_validates_bootstrap_configuration(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], message: str
) -> None:
    def unexpected_fit(*args: Any, **fit_kwargs: Any) -> None:
        pytest.fail("fit should not run for invalid inputs")

    monkeypatch.setattr("mirt.diagnostics.drf.fit_group_models", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        reliability_invariance(
            np.zeros((4, 2), dtype=np.int64),
            np.array([0, 0, 1, 1]),
            **kwargs,
        )


def test_plot_drf_applies_custom_line_options() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    result = {
        "theta_grid": np.array([-1.0, 0.0, 1.0]),
        "information_ref": np.array([1.0, 2.0, 1.0]),
        "information_focal": np.array([0.5, 1.5, 0.5]),
        "ref_group": "reference",
        "focal_group": "focal",
        "DRF": 0.5,
    }
    figure, axis = pyplot.subplots()

    axes = plot_drf(result, ax=axis, color="purple", linewidth=3.0)

    assert len(axes) == 2
    for current_axis in axes:
        assert [line.get_color() for line in current_axis.lines] == [
            "purple",
            "purple",
        ]
        assert [line.get_linewidth() for line in current_axis.lines] == [3.0, 3.0]
    pyplot.close(figure)


def test_plot_drf_creates_default_pair_of_axes() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    result = {
        "theta_grid": np.array([-1.0, 0.0, 1.0]),
        "information_ref": np.array([1.0, 2.0, 1.0]),
        "information_focal": np.array([0.5, 1.5, 0.5]),
        "ref_group": "reference",
        "focal_group": "focal",
        "DRF": 0.5,
    }

    axes = plot_drf(result)

    assert len(axes) == 2
    pyplot.close(axes[0].figure)
