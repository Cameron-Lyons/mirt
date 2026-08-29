"""Fast regression tests for Differential Test Functioning."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import integrate

from mirt.diagnostics.dtf import (
    _aggregate_dtf,
    _bootstrap_dtf_se,
    _bootstrap_dtf_statistics,
    _compute_expected_score,
    _create_integration_weights,
    compute_dtf,
    plot_dtf,
)


class BinaryModel:
    def __init__(self, probabilities: list[float] | np.ndarray):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.n_items = self.probabilities.size
        self.calls = 0

    def probability(self, theta, item_idx=None):
        self.calls += 1
        values = np.broadcast_to(
            self.probabilities, (np.asarray(theta).shape[0], self.n_items)
        ).copy()
        return values if item_idx is None else values[:, item_idx]


class PolytomousModel:
    n_items = 3

    def __init__(self):
        self.calls = 0

    def probability(self, theta, item_idx=None):
        self.calls += 1
        base = np.array(
            [
                [0.2, 0.3, 0.5, 0.0],
                [0.4, 0.6, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        values = np.broadcast_to(base, (np.asarray(theta).shape[0], 3, 4)).copy()
        return values if item_idx is None else values[:, item_idx]


def install_group_models(monkeypatch, reference, focal):
    calls: list[tuple[np.ndarray, np.ndarray, str, dict[str, Any]]] = []

    def fake_fit(ref_data, focal_data, model="2PL", **kwargs):
        calls.append((ref_data.copy(), focal_data.copy(), model, kwargs.copy()))
        return SimpleNamespace(model=reference), SimpleNamespace(model=focal)

    monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", fake_fit)
    return calls


def base_data():
    data = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    groups = np.array(["reference"] * 3 + ["focal"] * 3)
    return data, groups


class TestComputeDTF:
    def test_unsigned_statistic_is_in_score_units(self, monkeypatch):
        reference = BinaryModel([0.8, 0.7])
        focal = BinaryModel([0.2, 0.3])
        install_group_models(monkeypatch, reference, focal)
        data, groups = base_data()

        result = compute_dtf(
            data,
            groups,
            method="unsigned",
            theta_range=(-6.0, 6.0),
            n_bootstrap=0,
        )

        assert result["DTF"] == pytest.approx(1.0)
        np.testing.assert_allclose(result["expected_score_diff"], 1.0)
        assert integrate_weights(result) == pytest.approx(1.0)

    def test_signed_statistic_preserves_direction(self, monkeypatch):
        install_group_models(
            monkeypatch, BinaryModel([0.2, 0.3]), BinaryModel([0.8, 0.7])
        )
        data, groups = base_data()

        result = compute_dtf(data, groups, method="signed", n_bootstrap=0)

        assert result["DTF"] == pytest.approx(-1.0)

    def test_expected_score_method_returns_pointwise_curve(self, monkeypatch):
        install_group_models(
            monkeypatch, BinaryModel([0.8, 0.7]), BinaryModel([0.2, 0.3])
        )
        data, groups = base_data()

        result = compute_dtf(
            data, groups, method="expected_score", n_quadpts=11, n_bootstrap=0
        )

        assert result["DTF"] == pytest.approx(1.0)
        assert result["expected_score_diff"].shape == (11,)

    def test_uniform_weighting_is_normalized(self, monkeypatch):
        install_group_models(
            monkeypatch, BinaryModel([0.8, 0.7]), BinaryModel([0.2, 0.3])
        )
        data, groups = base_data()

        narrow = compute_dtf(
            data,
            groups,
            theta_range=(-1.0, 1.0),
            weighting="uniform",
            n_bootstrap=0,
        )
        wide = compute_dtf(
            data,
            groups,
            theta_range=(-10.0, 10.0),
            weighting="uniform",
            n_bootstrap=0,
        )

        assert narrow["DTF"] == pytest.approx(1.0)
        assert wide["DTF"] == pytest.approx(1.0)

    def test_custom_weighting_is_exposed(self, monkeypatch):
        install_group_models(
            monkeypatch, BinaryModel([0.8, 0.7]), BinaryModel([0.2, 0.3])
        )
        data, groups = base_data()
        custom = np.linspace(0.1, 1.0, 9)

        result = compute_dtf(
            data,
            groups,
            n_quadpts=9,
            weighting=custom,
            n_bootstrap=0,
        )

        assert result["weighting"] == "custom"
        assert integrate_weights(result) == pytest.approx(1.0)

    def test_focal_group_selection_controls_split(self, monkeypatch):
        calls = install_group_models(
            monkeypatch, BinaryModel([0.8, 0.7]), BinaryModel([0.2, 0.3])
        )
        data, groups = base_data()

        result = compute_dtf(data, groups, focal_group="reference", n_bootstrap=0)

        assert result["focal_group"] == "reference"
        assert result["ref_group"] == "focal"
        np.testing.assert_array_equal(calls[0][0], data[groups == "focal"])
        np.testing.assert_array_equal(calls[0][1], data[groups == "reference"])

    def test_result_reports_group_and_bootstrap_metadata(self, monkeypatch):
        install_group_models(
            monkeypatch, BinaryModel([0.8, 0.7]), BinaryModel([0.2, 0.3])
        )
        data, groups = base_data()

        result = compute_dtf(data, groups, n_bootstrap=0)

        assert result["n_reference"] == 3
        assert result["n_focal"] == 3
        assert result["n_bootstrap"] == 0
        assert result["n_bootstrap_successful"] == 0
        assert result["n_bootstrap_failed"] == 0
        assert np.isnan(result["DTF_SE"])
        assert np.isnan(result["p_value"])
        np.testing.assert_array_equal(
            np.isnan(result["confidence_interval"]), [True, True]
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"model": "Rasch"}, "model must be one of"),
            ({"method": "typo"}, "method must be one of"),
            ({"n_quadpts": 1}, "n_quadpts"),
            ({"n_quadpts": 2.5}, "n_quadpts"),
            ({"n_quadpts": True}, "n_quadpts"),
            ({"n_bootstrap": -1}, "n_bootstrap"),
            ({"n_bootstrap": 2.5}, "n_bootstrap"),
            ({"n_bootstrap": False}, "n_bootstrap"),
            ({"n_jobs": 0}, "n_jobs"),
            ({"n_jobs": -2}, "n_jobs"),
            ({"n_jobs": 2.5}, "n_jobs"),
            ({"n_jobs": True}, "n_jobs"),
            ({"confidence_level": 0.0}, "confidence_level"),
            ({"confidence_level": 1.0}, "confidence_level"),
            ({"confidence_level": np.nan}, "confidence_level"),
            ({"theta_range": (4.0, -4.0)}, "strictly increasing"),
            ({"theta_range": (0.0, 0.0)}, "strictly increasing"),
            ({"theta_range": (0.0,)}, "two finite"),
            ({"theta_range": (0.0, np.inf)}, "two finite"),
        ],
    )
    def test_configuration_validation_happens_before_fitting(
        self, monkeypatch, kwargs, match
    ):
        def unexpected_fit(*args, **fit_kwargs):
            raise AssertionError("fit should not run")

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", unexpected_fit)
        data, groups = base_data()
        with pytest.raises(ValueError, match=match):
            compute_dtf(data, groups, **kwargs)

    @pytest.mark.parametrize(
        ("data", "groups", "match"),
        [
            (np.ones(4), np.array([0, 0, 1, 1]), "two-dimensional"),
            (np.empty((0, 2)), np.empty(0), "nonempty"),
            (np.ones((4, 2)), np.ones((4, 1)), "one-dimensional"),
            (np.ones((4, 2)), np.array([0, 0, 1]), "length must match"),
            (np.ones((4, 2)), np.array([0.0, 0.0, 1.0, np.nan]), "non-finite"),
            (
                np.ones((4, 2)),
                np.array(["a", "a", "b", None], dtype=object),
                "missing",
            ),
        ],
    )
    def test_data_and_group_validation(self, data, groups, match):
        with pytest.raises(ValueError, match=match):
            compute_dtf(data, groups, n_bootstrap=0)

    @pytest.mark.parametrize(
        ("weighting", "match"),
        [
            ("triangular", "weighting must"),
            (np.ones(3), "custom weighting must have shape"),
            (np.array([1.0, 1.0, -1.0, 1.0]), "nonnegative"),
            (np.zeros(4), "positive integral"),
            (np.array([1.0, 1.0, np.nan, 1.0]), "finite"),
        ],
    )
    def test_weight_validation_happens_before_fitting(
        self, monkeypatch, weighting, match
    ):
        def unexpected_fit(*args, **kwargs):
            raise AssertionError("fit should not run")

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", unexpected_fit)
        data, groups = base_data()
        with pytest.raises(ValueError, match=match):
            compute_dtf(
                data,
                groups,
                n_quadpts=4,
                weighting=weighting,
                n_bootstrap=0,
            )


def integrate_weights(result):
    return float(integrate.trapezoid(result["ability_weights"], result["theta_grid"]))


class TestExpectedScore:
    def test_dichotomous_score_uses_one_probability_call(self):
        model = BinaryModel([0.8, 0.4, 0.2])
        theta = np.linspace(-2.0, 2.0, 9)

        result = _compute_expected_score(model, theta)

        np.testing.assert_allclose(result, 1.4)
        assert model.calls == 1

    def test_polytomous_score_uses_one_probability_call(self):
        model = PolytomousModel()
        theta = np.linspace(-2.0, 2.0, 9)

        result = _compute_expected_score(model, theta)

        np.testing.assert_allclose(result, 3.9)
        assert model.calls == 1

    def test_one_item_vector_output_is_supported(self):
        class OneItem:
            n_items = 1

            def probability(self, theta, item_idx=None):
                return np.linspace(0.2, 0.8, len(theta))

        result = _compute_expected_score(OneItem(), np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [0.2, 0.5, 0.8])

    @pytest.mark.parametrize(
        ("theta", "match"),
        [
            (np.array([]), "nonempty"),
            (np.ones((2, 1)), "one-dimensional"),
            (np.array([0.0, np.nan]), "finite"),
        ],
    )
    def test_theta_validation(self, theta, match):
        with pytest.raises(ValueError, match=match):
            _compute_expected_score(BinaryModel([0.5]), theta)

    @pytest.mark.parametrize(
        ("output", "n_items", "match"),
        [
            (np.array([0.2, np.nan]), 1, "finite"),
            (np.array([0.2, 1.2]), 1, "lie in"),
            (np.ones(2), 2, "one-item"),
            (np.ones((2, 3)), 2, "dichotomous probability shape"),
            (np.ones((2, 3, 4)), 2, "polytomous probability shape"),
            (np.ones((2, 2, 2, 2)), 2, "one-, two-, or three"),
        ],
    )
    def test_probability_validation(self, output, n_items, match):
        class InvalidModel:
            def __init__(self):
                self.n_items = n_items

            def probability(self, theta, item_idx=None):
                return output

        with pytest.raises(ValueError, match=match):
            _compute_expected_score(InvalidModel(), np.array([-1.0, 1.0]))


class TestWeightingAndAggregation:
    @pytest.mark.parametrize("weighting", ["normal", "uniform"])
    def test_builtin_weights_integrate_to_one(self, weighting):
        theta = np.linspace(-4.0, 4.0, 101)
        weights, name = _create_integration_weights(theta, weighting)
        assert name == weighting
        assert integrate.trapezoid(weights, theta) == pytest.approx(1.0)

    def test_signed_and_unsigned_aggregation(self):
        theta = np.linspace(-1.0, 1.0, 5)
        weights, _ = _create_integration_weights(theta, "uniform")
        difference = theta.copy()

        signed = _aggregate_dtf(difference, theta, "signed", weights)
        unsigned = _aggregate_dtf(difference, theta, "unsigned", weights)

        assert signed == pytest.approx(0.0, abs=1e-15)
        assert unsigned == pytest.approx(0.5)

    def test_aggregation_rejects_unknown_method(self):
        theta = np.linspace(-1.0, 1.0, 5)
        weights, _ = _create_integration_weights(theta, "uniform")

        with pytest.raises(ValueError, match="Unknown DTF method"):
            _aggregate_dtf(theta, theta, "invalid", weights)


class TestBootstrap:
    def install_data_driven_fit(self, monkeypatch):
        def fake_fit(ref_data, focal_data, model="2PL", **kwargs):
            ref_probability = 0.1 + 0.8 * float(np.mean(ref_data))
            focal_probability = 0.1 + 0.8 * float(np.mean(focal_data))
            return (
                SimpleNamespace(model=BinaryModel([ref_probability] * 2)),
                SimpleNamespace(model=BinaryModel([focal_probability] * 2)),
            )

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", fake_fit)

    def test_public_bootstrap_is_reproducible(self, monkeypatch):
        self.install_data_driven_fit(monkeypatch)
        data, groups = base_data()

        first = compute_dtf(data, groups, n_bootstrap=20, random_state=123, n_quadpts=9)
        second = compute_dtf(
            data, groups, n_bootstrap=20, random_state=123, n_quadpts=9
        )

        assert first["DTF_SE"] == pytest.approx(second["DTF_SE"])
        assert first["p_value"] == pytest.approx(second["p_value"])
        np.testing.assert_allclose(
            first["confidence_interval"], second["confidence_interval"]
        )
        assert first["n_bootstrap_successful"] == 20
        assert first["n_bootstrap_failed"] == 0
        assert np.isfinite(first["DTF_SE"])
        assert 0.0 <= first["p_value"] <= 1.0

    def test_parallel_bootstrap_matches_serial_results(self, monkeypatch):
        self.install_data_driven_fit(monkeypatch)

        def run_inline(function, tasks, n_jobs):
            assert len(tasks) == min(n_jobs, 24)
            return [function(task) for task in tasks]

        monkeypatch.setattr("mirt.diagnostics.dtf._run_bootstrap_tasks", run_inline)
        data, groups = base_data()
        options = {
            "n_bootstrap": 24,
            "random_state": 2718,
            "n_quadpts": 9,
            "confidence_level": 0.9,
        }

        serial = compute_dtf(data, groups, n_jobs=1, **options)
        parallel = compute_dtf(data, groups, n_jobs=3, **options)

        for key in ("DTF", "DTF_SE", "p_value"):
            assert parallel[key] == pytest.approx(serial[key])
        np.testing.assert_allclose(
            parallel["confidence_interval"], serial["confidence_interval"]
        )
        assert parallel["n_bootstrap_successful"] == 24
        assert parallel["n_bootstrap_failed"] == 0
        assert parallel["n_jobs"] == 3

    def test_failures_are_counted(self, monkeypatch):
        call_count = 0

        def sometimes_fails(ref_data, focal_data, model="2PL", **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError("synthetic failed fit")
            probability = 0.2 + 0.01 * call_count
            return (
                SimpleNamespace(model=BinaryModel([probability, 0.5])),
                SimpleNamespace(model=BinaryModel([0.4, 0.5])),
            )

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", sometimes_fails)
        data, groups = base_data()
        theta = np.linspace(-2.0, 2.0, 9)
        weights, _ = _create_integration_weights(theta, "normal")

        summary = _bootstrap_dtf_statistics(
            data=data,
            groups=groups,
            model="2PL",
            method="unsigned",
            theta_grid=theta,
            integration_weights=weights,
            observed_dtf=0.2,
            ref_group="focal",
            focal_group="reference",
            n_bootstrap=10,
            confidence_level=0.95,
            random_state=1,
            fit_kwargs={},
        )

        assert summary.n_successful == 5
        assert summary.n_failed == 5
        assert np.isfinite(summary.standard_error)

    def test_too_few_successes_return_nan(self, monkeypatch):
        def always_fails(*args, **kwargs):
            raise RuntimeError("synthetic failed fit")

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", always_fails)
        data, groups = base_data()
        theta = np.linspace(-2.0, 2.0, 9)
        weights, _ = _create_integration_weights(theta, "normal")

        summary = _bootstrap_dtf_statistics(
            data=data,
            groups=groups,
            model="2PL",
            method="unsigned",
            theta_grid=theta,
            integration_weights=weights,
            observed_dtf=0.2,
            ref_group="focal",
            focal_group="reference",
            n_bootstrap=5,
            confidence_level=0.95,
            random_state=1,
            fit_kwargs={},
        )

        assert summary.n_successful == 0
        assert summary.n_failed == 5
        assert np.isnan(summary.standard_error)
        assert np.isnan(summary.p_value)

    def test_p_value_uses_observed_statistic(self, monkeypatch):
        self.install_data_driven_fit(monkeypatch)
        data, groups = base_data()
        theta = np.linspace(-2.0, 2.0, 9)
        weights, _ = _create_integration_weights(theta, "normal")
        common = dict(
            data=data,
            groups=groups,
            model="2PL",
            method="signed",
            theta_grid=theta,
            integration_weights=weights,
            ref_group="focal",
            focal_group="reference",
            n_bootstrap=20,
            confidence_level=0.95,
            random_state=123,
            fit_kwargs={},
        )

        null = _bootstrap_dtf_statistics(observed_dtf=0.0, **common)
        distant = _bootstrap_dtf_statistics(observed_dtf=10.0, **common)

        assert null.p_value == 1.0
        assert distant.p_value < 1e-6

    @pytest.mark.parametrize(
        ("observed_dtf", "expected_p_value"), [(0.0, 1.0), (0.25, 0.0)]
    )
    def test_degenerate_bootstrap_has_exact_p_value(
        self, monkeypatch, observed_dtf, expected_p_value
    ):
        def constant_fit(ref_data, focal_data, model="2PL", **kwargs):
            return (
                SimpleNamespace(model=BinaryModel([0.6, 0.5])),
                SimpleNamespace(model=BinaryModel([0.4, 0.5])),
            )

        monkeypatch.setattr("mirt.diagnostics.dtf.fit_group_models", constant_fit)
        data, groups = base_data()
        theta = np.linspace(-2.0, 2.0, 9)
        weights, _ = _create_integration_weights(theta, "normal")

        summary = _bootstrap_dtf_statistics(
            data=data,
            groups=groups,
            model="2PL",
            method="unsigned",
            theta_grid=theta,
            integration_weights=weights,
            observed_dtf=observed_dtf,
            ref_group="focal",
            focal_group="reference",
            n_bootstrap=5,
            confidence_level=0.95,
            random_state=1,
            fit_kwargs={},
        )

        assert summary.standard_error == pytest.approx(0.0)
        assert summary.p_value == expected_p_value

    def test_compatibility_wrapper_returns_two_values(self, monkeypatch):
        self.install_data_driven_fit(monkeypatch)
        data, groups = base_data()

        standard_error, p_value = _bootstrap_dtf_se(
            data,
            groups,
            "2PL",
            "unsigned",
            (-2.0, 2.0),
            9,
            n_bootstrap=10,
            random_state=123,
        )

        assert np.isfinite(standard_error)
        assert 0.0 <= p_value <= 1.0


class FakeAxes:
    def __init__(self):
        self.plot_calls: list[dict[str, Any]] = []
        self.fill_calls: list[dict[str, Any]] = []

    def plot(self, *args, **kwargs):
        self.plot_calls.append(kwargs)

    def fill_between(self, *args, **kwargs):
        self.fill_calls.append(kwargs)

    def set_xlabel(self, *args):
        pass

    def set_ylabel(self, *args):
        pass

    def set_title(self, *args):
        pass

    def legend(self):
        pass

    def grid(self, *args, **kwargs):
        pass


def install_fake_matplotlib(monkeypatch, axes=None):
    pyplot = ModuleType("matplotlib.pyplot")
    selected_axes = axes or FakeAxes()
    pyplot.subplots = lambda **kwargs: (object(), selected_axes)
    package = ModuleType("matplotlib")
    package.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return selected_axes


def plot_result():
    return {
        "theta_grid": np.array([-1.0, 0.0, 1.0]),
        "expected_score_ref": np.array([0.5, 1.0, 1.5]),
        "expected_score_focal": np.array([0.4, 0.8, 1.2]),
        "ref_group": "reference",
        "focal_group": "focal",
        "DTF": 0.2,
    }


class TestPlotDTF:
    def test_returns_axes_and_forwards_line_style(self, monkeypatch):
        axes = install_fake_matplotlib(monkeypatch)

        returned = plot_dtf(plot_result(), ax=axes, color="red", linestyle="--")

        assert returned is axes
        assert len(axes.plot_calls) == 2
        assert [call["color"] for call in axes.plot_calls] == ["red", "red"]
        assert [call["linestyle"] for call in axes.plot_calls] == ["--", "--"]
        assert all(call["linewidth"] == 2.0 for call in axes.plot_calls)

    def test_creates_axes_when_omitted(self, monkeypatch):
        axes = install_fake_matplotlib(monkeypatch)
        assert plot_dtf(plot_result()) is axes

    def test_missing_optional_dependency_has_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises(ImportError, match="matplotlib required"):
            plot_dtf(plot_result())

    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (lambda result: result.pop("theta_grid"), "missing required keys"),
            (
                lambda result: result.__setitem__("theta_grid", np.array([0.0])),
                "length >= 2",
            ),
            (
                lambda result: result.__setitem__(
                    "expected_score_ref", np.array([1.0, 2.0])
                ),
                "must match",
            ),
            (
                lambda result: result.__setitem__(
                    "expected_score_focal", np.array([0.0, np.nan, 1.0])
                ),
                "finite",
            ),
            (lambda result: result.__setitem__("DTF", np.nan), "DTF must be finite"),
        ],
    )
    def test_result_validation(self, monkeypatch, mutator, match):
        axes = install_fake_matplotlib(monkeypatch)
        result = plot_result()
        mutator(result)
        with pytest.raises(ValueError, match=match):
            plot_dtf(result, ax=axes)
