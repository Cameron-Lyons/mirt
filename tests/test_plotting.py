"""Fast contract tests for the optional plotting surface."""

from __future__ import annotations

import builtins
import sys
from collections import defaultdict
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import mirt.plotting as plotting
from mirt._api_registry import PLOTTING_EXPORTS
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import (
    MultidimensionalModel as RealMultidimensionalModel,
)
from mirt.models.polytomous import GradedResponseModel


class RecordingAxes:
    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = (
            defaultdict(list)
        )
        self.transAxes = object()
        self.secondary: RecordingAxes | None = None

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls[name].append((args, kwargs))

    def plot(self, *args: Any, **kwargs: Any) -> None:
        self._record("plot", *args, **kwargs)

    def hist(self, *args: Any, **kwargs: Any) -> None:
        self._record("hist", *args, **kwargs)

    def bar(self, *args: Any, **kwargs: Any) -> None:
        self._record("bar", *args, **kwargs)

    def scatter(self, *args: Any, **kwargs: Any) -> None:
        self._record("scatter", *args, **kwargs)

    def axhline(self, *args: Any, **kwargs: Any) -> None:
        self._record("axhline", *args, **kwargs)

    def grid(self, *args: Any, **kwargs: Any) -> None:
        self._record("grid", *args, **kwargs)

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self._record("legend", *args, **kwargs)

    def annotate(self, *args: Any, **kwargs: Any) -> None:
        self._record("annotate", *args, **kwargs)

    def text(self, *args: Any, **kwargs: Any) -> None:
        self._record("text", *args, **kwargs)

    def twinx(self) -> RecordingAxes:
        self._record("twinx")
        self.secondary = RecordingAxes()
        return self.secondary

    def twiny(self) -> RecordingAxes:
        self._record("twiny")
        self.secondary = RecordingAxes()
        return self.secondary

    def __getattr__(self, name: str):
        if name.startswith("set_"):
            return lambda *args, **kwargs: self._record(name, *args, **kwargs)
        raise AttributeError(name)


class BinaryModel:
    n_items = 3
    n_factors = 1
    is_polytomous = False
    item_names = ["Alpha", "Beta", "Gamma"]
    parameters = {"difficulty": np.array([-1.0, 0.0, 1.0])}

    def __init__(self) -> None:
        self.probability_calls: list[int | None] = []
        self.information_calls: list[int | None] = []
        self.expected_score_calls: list[int | None] = []
        self.last_theta: np.ndarray | None = None

    def probability(self, theta, item_idx=None):
        theta = np.asarray(theta, dtype=float)
        self.last_theta = theta.copy()
        self.probability_calls.append(item_idx)
        probabilities = 1.0 / (
            1.0 + np.exp(-(theta[:, :1] - self.parameters["difficulty"]))
        )
        if item_idx is None:
            return probabilities
        return probabilities[:, item_idx]

    def information(self, theta, item_idx=None):
        self.information_calls.append(item_idx)
        probabilities = self.probability(theta, item_idx)
        return probabilities * (1.0 - probabilities)

    def expected_score(self, theta, item_idx=None):
        self.expected_score_calls.append(item_idx)
        probabilities = self.probability(theta, item_idx)
        if item_idx is None:
            return np.sum(probabilities, axis=1)
        return probabilities


class PolytomousModel:
    n_items = 2
    n_factors = 1
    is_polytomous = True
    item_names = ["Short", "Long"]
    n_categories = [3, 4]
    parameters = {"thresholds": np.array([[-1.0, 1.0, 99.0], [-2.0, 0.0, 2.0]])}

    def __init__(self) -> None:
        self.probability_calls: list[int | None] = []
        self.information_calls: list[int | None] = []
        self.expected_score_calls: list[int | None] = []
        self.last_theta: np.ndarray | None = None

    def probability(self, theta, item_idx=None):
        theta = np.asarray(theta, dtype=float)
        self.last_theta = theta.copy()
        self.probability_calls.append(item_idx)
        n_points = theta.shape[0]
        short = np.tile([0.2, 0.5, 0.3], (n_points, 1))
        long = np.tile([0.1, 0.2, 0.3, 0.4], (n_points, 1))
        if item_idx == 0:
            return short
        if item_idx == 1:
            return long
        result = np.zeros((n_points, 2, 4))
        result[:, 0, :3] = short
        result[:, 1, :] = long
        return result

    def information(self, theta, item_idx=None):
        self.information_calls.append(item_idx)
        n_points = np.asarray(theta).shape[0]
        first = np.linspace(1.0, 2.0, n_points)
        second = np.linspace(2.0, 3.0, n_points)
        if item_idx == 0:
            return first
        if item_idx == 1:
            return second
        return first + second

    def expected_score(self, theta, item_idx=None):
        self.expected_score_calls.append(item_idx)
        n_points = np.asarray(theta).shape[0]
        values = [1.1, 2.0]
        if item_idx is None:
            return np.full(n_points, sum(values))
        return np.full(n_points, values[item_idx])


class MultidimensionalModel:
    n_items = 2
    n_factors = 2
    is_polytomous = False
    item_names = ["One", "Two"]
    parameters = {
        "slopes": np.array([[1.0, 2.0], [2.0, 1.0]]),
        "intercepts": np.array([1.0, -2.0]),
    }

    def __init__(self) -> None:
        self.last_theta: np.ndarray | None = None

    def probability(self, theta, item_idx=None):
        theta = np.asarray(theta, dtype=float)
        self.last_theta = theta.copy()
        logits = theta @ self.parameters["slopes"].T + self.parameters["intercepts"]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities if item_idx is None else probabilities[:, item_idx]

    def information(self, theta, item_idx=None):
        probabilities = self.probability(theta, item_idx)
        if item_idx is None:
            scale = np.sum(self.parameters["slopes"] ** 2, axis=1)
            return probabilities * (1.0 - probabilities) * scale
        scale = np.sum(self.parameters["slopes"][item_idx] ** 2)
        return probabilities * (1.0 - probabilities) * scale

    def expected_score(self, theta, item_idx=None):
        probabilities = self.probability(theta, item_idx)
        if item_idx is None:
            return np.sum(probabilities, axis=1)
        return probabilities


class TestOptionalBackend:
    def test_supplied_axes_does_not_import_backend(self, monkeypatch):
        monkeypatch.setattr(
            plotting,
            "_check_matplotlib",
            lambda: pytest.fail("backend should not be imported"),
        )
        axes = RecordingAxes()

        result = plotting.plot_expected_score(BinaryModel(), ax=axes)

        assert result is axes

    def test_missing_backend_error_is_actionable(self, monkeypatch):
        original_import = builtins.__import__

        def unavailable(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("synthetic missing backend")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", unavailable)

        with pytest.raises(ImportError, match=r"mirt\[plot\]"):
            plotting._check_matplotlib()

    def test_available_backend_is_returned(self, monkeypatch):
        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []
        pyplot = ModuleType("matplotlib.pyplot")
        matplotlib.pyplot = pyplot
        monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

        assert plotting._check_matplotlib() is pyplot

    def test_axes_are_created_with_expected_size(self, monkeypatch):
        axes = RecordingAxes()
        subplots_calls: list[dict[str, Any]] = []

        def subplots(**kwargs):
            subplots_calls.append(kwargs)
            return object(), axes

        monkeypatch.setattr(
            plotting, "_check_matplotlib", lambda: SimpleNamespace(subplots=subplots)
        )

        result = plotting.plot_expected_score(BinaryModel())

        assert result is axes
        assert subplots_calls == [{"figsize": (8, 6)}]


def test_category_curves_are_registered_for_lazy_top_level_access():
    import mirt

    assert "plot_category_curves" in PLOTTING_EXPORTS
    assert mirt.plot_category_curves is plotting.plot_category_curves


class TestGridAndItemValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"theta_range": (1.0, -1.0)}, "strictly increasing"),
            ({"theta_range": (0.0, np.inf)}, "two finite"),
            ({"theta_range": (0.0, 1.0, 2.0)}, "two finite"),
            ({"n_points": 1}, "at least 2"),
            ({"n_points": 2.5}, "integer"),
            ({"n_points": False}, "integer"),
            ({"factor": 1}, "factor must be in"),
            ({"factor": 0.5}, "factor must be an integer"),
            ({"factor": True}, "factor must be an integer"),
        ],
    )
    def test_invalid_grid_is_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plotting.plot_expected_score(BinaryModel(), ax=RecordingAxes(), **kwargs)

    @pytest.mark.parametrize(
        ("item_idx", "error", "match"),
        [
            (True, ValueError, "integers"),
            (1.5, ValueError, "integer or a sequence"),
            ([0, 0], ValueError, "unique"),
            ([0, 1.5], ValueError, "integers"),
            (-1, IndexError, "out of range"),
            (3, IndexError, "out of range"),
        ],
    )
    def test_invalid_items_are_rejected(self, item_idx, error, match):
        with pytest.raises(error, match=match):
            plotting.plot_icc(BinaryModel(), item_idx=item_idx, ax=RecordingAxes())

    def test_invalid_model_size_is_rejected(self):
        model = BinaryModel()
        model.n_items = 0

        with pytest.raises(ValueError, match="positive integer"):
            plotting.plot_icc(model, ax=RecordingAxes())

    def test_invalid_factor_count_is_rejected(self):
        model = BinaryModel()
        model.n_factors = False

        with pytest.raises(ValueError, match="n_factors"):
            plotting.plot_icc(model, ax=RecordingAxes())

    def test_fallback_item_names_are_generated(self):
        model = BinaryModel()
        model.item_names = None
        axes = RecordingAxes()

        plotting.plot_icc(model, item_idx=0, ax=axes)

        assert axes.calls["plot"][0][1]["label"] == "Item 1"


class TestItemCurves:
    def test_binary_icc_returns_axes_and_forwards_style(self):
        axes = RecordingAxes()

        result = plotting.plot_icc(
            BinaryModel(), item_idx=[0, 2], n_points=7, ax=axes, color="purple"
        )

        assert result is axes
        assert len(axes.calls["plot"]) == 2
        assert [call[1]["label"] for call in axes.calls["plot"]] == ["Alpha", "Gamma"]
        assert all(call[1]["color"] == "purple" for call in axes.calls["plot"])
        assert len(axes.calls["legend"]) == 1

    def test_polytomous_icc_plots_every_category(self):
        axes = RecordingAxes()

        plotting.plot_icc(PolytomousModel(), ax=axes, n_points=5)

        assert len(axes.calls["plot"]) == 7
        assert axes.calls["plot"][0][1]["label"] == "Short · Category 0"
        assert axes.calls["plot"][-1][1]["label"] == "Long · Category 3"
        assert axes.calls["set_ylabel"][0][0] == ("P(X = k)",)

    def test_legend_can_be_suppressed(self):
        axes = RecordingAxes()

        plotting.plot_icc(BinaryModel(), ax=axes, show_legend=False)

        assert not axes.calls["legend"]

    def test_show_legend_must_be_boolean(self):
        with pytest.raises(ValueError, match="show_legend must be boolean"):
            plotting.plot_icc(BinaryModel(), ax=RecordingAxes(), show_legend="yes")

    def test_large_curve_set_does_not_force_legend(self):
        model = BinaryModel()
        model.n_items = 11
        model.item_names = [f"I{i}" for i in range(11)]
        model.parameters = {"difficulty": np.arange(11, dtype=float)}
        axes = RecordingAxes()

        plotting.plot_icc(model, ax=axes)

        assert len(axes.calls["plot"]) == 11
        assert not axes.calls["legend"]

    def test_empty_selection_is_rejected(self):
        with pytest.raises(ValueError, match="at least one item"):
            plotting.plot_icc(BinaryModel(), item_idx=[], ax=RecordingAxes())

    def test_factor_slice_varies_only_selected_dimension(self):
        model = MultidimensionalModel()

        plotting.plot_icc(model, item_idx=0, factor=1, n_points=5, ax=RecordingAxes())

        assert model.last_theta is not None
        np.testing.assert_allclose(model.last_theta[:, 0], 0.0)
        np.testing.assert_allclose(model.last_theta[:, 1], np.linspace(-4, 4, 5))

    def test_binary_category_curves_are_complements(self):
        model = BinaryModel()
        axes = RecordingAxes()

        result = plotting.plot_category_curves(model, 1, n_points=5, ax=axes)

        assert result is axes
        assert len(axes.calls["plot"]) == 2
        first = axes.calls["plot"][0][0][1]
        second = axes.calls["plot"][1][0][1]
        np.testing.assert_allclose(first + second, 1.0)

    def test_polytomous_category_curves_plot_all_categories(self):
        axes = RecordingAxes()

        plotting.plot_category_curves(PolytomousModel(), 1, ax=axes, linestyle="--")

        assert len(axes.calls["plot"]) == 4
        assert all(call[1]["linestyle"] == "--" for call in axes.calls["plot"])
        assert axes.calls["set_title"][0][0] == ("Category Response Curves: Long",)

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (np.ones((3, 2, 2)), "one- or two-dimensional"),
            (np.ones(4), "shape"),
            (np.ones((5, 1)), "n_categories"),
            (np.tile([0.2, 0.2], (5, 1)), "sum to one"),
            (np.tile([np.nan, np.nan], (5, 1)), "finite values"),
            (np.tile([-0.1, 1.1], (5, 1)), "lie in"),
        ],
    )
    def test_invalid_item_probability_output_is_rejected(self, output, match):
        model = BinaryModel()
        model.probability = lambda theta, item_idx=None: output

        with pytest.raises(ValueError, match=match):
            plotting.plot_category_curves(model, 0, n_points=5, ax=RecordingAxes())


class TestInformation:
    def test_binary_test_information_uses_vectorized_matrix(self):
        model = BinaryModel()
        axes = RecordingAxes()

        plotting.plot_information(model, n_points=5, ax=axes)

        assert model.information_calls == [None]
        expected = np.sum(model.information(np.linspace(-4, 4, 5)[:, None]), axis=1)
        np.testing.assert_allclose(axes.calls["plot"][0][0][1], expected)

    def test_binary_item_information_uses_one_full_evaluation(self):
        model = BinaryModel()
        axes = RecordingAxes()

        plotting.plot_information(model, test_info=False, n_points=5, ax=axes)

        assert model.information_calls == [None]
        assert len(axes.calls["plot"]) == 3

    def test_polytomous_total_information_accepts_one_dimensional_output(self):
        model = PolytomousModel()
        axes = RecordingAxes()

        plotting.plot_information(model, n_points=5, ax=axes)

        assert model.information_calls == [None]
        assert len(axes.calls["plot"]) == 1

    def test_polytomous_items_avoid_recomputing_total(self):
        model = PolytomousModel()
        axes = RecordingAxes()

        plotting.plot_information(model, test_info=False, n_points=5, ax=axes)

        assert model.information_calls == [0, 1]
        assert len(axes.calls["plot"]) == 2

    def test_polytomous_all_items_supply_total_without_extra_call(self):
        model = PolytomousModel()
        axes = RecordingAxes()

        plotting.plot_information(
            model, item_idx=[0, 1], test_info=True, n_points=5, ax=axes
        )

        assert model.information_calls == [0, 1]
        total = axes.calls["plot"][0][0][1]
        item_sum = axes.calls["plot"][1][0][1] + axes.calls["plot"][2][0][1]
        np.testing.assert_allclose(total, item_sum)

    def test_polytomous_subset_fetches_total_once(self):
        model = PolytomousModel()

        plotting.plot_information(
            model, item_idx=1, test_info=True, n_points=5, ax=RecordingAxes()
        )

        assert model.information_calls == [1, None]

    def test_style_overrides_defaults_without_collision(self):
        axes = RecordingAxes()

        plotting.plot_information(BinaryModel(), ax=axes, linewidth=7.0, alpha=0.2)

        assert axes.calls["plot"][0][1]["linewidth"] == 7.0
        assert axes.calls["plot"][0][1]["alpha"] == 0.2

    @pytest.mark.parametrize("value", ["yes", 1])
    def test_test_info_must_be_boolean(self, value):
        with pytest.raises(ValueError, match="boolean"):
            plotting.plot_information(
                BinaryModel(), test_info=value, ax=RecordingAxes()
            )

    def test_empty_item_selection_requires_test_curve(self):
        with pytest.raises(ValueError, match="select at least"):
            plotting.plot_information(
                BinaryModel(), item_idx=[], test_info=False, ax=RecordingAxes()
            )

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (np.ones((5, 2)), "full information"),
            (np.full((5, 3), -1.0), "nonnegative"),
            (np.full((5, 3), np.nan), "finite"),
            (np.ones((5, 3, 1)), "full information"),
        ],
    )
    def test_invalid_full_information_is_rejected(self, output, match):
        model = BinaryModel()
        model.information = lambda theta, item_idx=None: output

        with pytest.raises(ValueError, match=match):
            plotting.plot_information(model, n_points=5, ax=RecordingAxes())

    def test_invalid_item_information_is_rejected(self):
        model = PolytomousModel()
        model.information = lambda theta, item_idx=None: np.ones((5, 1))

        with pytest.raises(ValueError, match="shape"):
            plotting.plot_information(
                model, item_idx=0, test_info=False, n_points=5, ax=RecordingAxes()
            )

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (np.full(5, np.nan), "finite"),
            (np.full(5, -1.0), "nonnegative"),
        ],
    )
    def test_invalid_total_curve_is_rejected(self, output, match):
        model = PolytomousModel()
        model.information = lambda theta, item_idx=None: output

        with pytest.raises(ValueError, match=match):
            plotting.plot_information(model, n_points=5, ax=RecordingAxes())

    def test_generic_total_only_model_fetches_selected_items(self):
        model = PolytomousModel()
        model.is_polytomous = False

        plotting.plot_information(
            model, item_idx=0, test_info=True, n_points=5, ax=RecordingAxes()
        )

        assert model.information_calls == [None, 0]


class TestAbilityDistribution:
    def test_density_plot_forwards_histogram_style(self):
        axes = RecordingAxes()

        result = plotting.plot_ability_distribution(
            np.array([-1.0, 0.0, 1.0]), ax=axes, alpha=0.2, color="navy"
        )

        assert result is axes
        assert axes.calls["hist"][0][1]["alpha"] == 0.2
        assert axes.calls["hist"][0][1]["color"] == "navy"
        assert len(axes.calls["plot"]) == 2

    def test_constant_values_skip_singular_kde(self):
        axes = RecordingAxes()

        plotting.plot_ability_distribution(np.ones(5), ax=axes)

        assert len(axes.calls["plot"]) == 1
        assert axes.calls["plot"][0][1]["label"] == "N(0,1)"

    def test_count_histogram_can_disable_all_overlays(self):
        axes = RecordingAxes()

        plotting.plot_ability_distribution(
            np.arange(5.0),
            ax=axes,
            density=False,
            show_density=False,
            show_normal=False,
        )

        assert not axes.calls["plot"]
        assert not axes.calls["legend"]
        assert axes.calls["set_ylabel"][0][0] == ("Count",)

    def test_factor_and_standard_error_are_selected_together(self):
        axes = RecordingAxes()
        theta = np.column_stack((np.arange(4.0), np.arange(10.0, 14.0)))
        se = np.column_stack((np.ones(4), np.full(4, 0.25)))

        plotting.plot_ability_distribution(
            theta,
            se=se,
            factor=1,
            show_density=False,
            ax=axes,
        )

        np.testing.assert_allclose(axes.calls["hist"][0][0][0], theta[:, 1])
        assert axes.calls["text"][0][0][2] == "Mean SE = 0.250"

    def test_selected_factor_accepts_one_dimensional_standard_error(self):
        axes = RecordingAxes()
        theta = np.column_stack((np.arange(4.0), np.arange(10.0, 14.0)))

        plotting.plot_ability_distribution(
            theta,
            se=np.full(4, 0.5),
            factor=1,
            show_density=False,
            ax=axes,
        )

        assert axes.calls["text"][0][0][2] == "Mean SE = 0.500"

    @pytest.mark.parametrize(
        ("theta", "kwargs", "match"),
        [
            (np.array([]), {}, "finite values"),
            (np.array([0.0, np.nan]), {}, "finite values"),
            (np.ones((2, 2, 1)), {}, "one- or two-dimensional"),
            (np.ones(3), {"factor": 1}, "factor must be 0"),
            (np.ones((3, 2)), {"factor": 2}, "factor must be in"),
            (np.ones(3), {"factor": True}, "factor must be an integer"),
            (np.ones(3), {"bins": 0}, "valid histogram"),
            (np.ones(3), {"density": "yes"}, "density must be boolean"),
            (
                np.arange(3.0),
                {"density": False, "show_density": True},
                "density overlays",
            ),
            (np.ones(3), {"show_density": "yes"}, "show_density must be boolean"),
            (np.ones(3), {"show_normal": 1}, "show_normal must be boolean"),
        ],
    )
    def test_invalid_distribution_inputs(self, theta, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plotting.plot_ability_distribution(theta, ax=RecordingAxes(), **kwargs)

    @pytest.mark.parametrize(
        ("se", "match"),
        [
            (np.ones(2), "match theta"),
            (np.array([0.1, -0.1, 0.1]), "nonnegative"),
            (np.array([0.1, np.nan, 0.1]), "finite values"),
            (np.ones((3, 1, 1)), "one- or two-dimensional"),
        ],
    )
    def test_invalid_standard_errors(self, se, match):
        with pytest.raises(ValueError, match=match):
            plotting.plot_ability_distribution(
                np.arange(3.0), se=se, ax=RecordingAxes()
            )


class TestItemFit:
    def test_itemfit_colors_misfit_and_forwards_style(self):
        axes = RecordingAxes()

        result = plotting.plot_itemfit(
            {"outfit": np.array([0.8, 1.5])},
            statistic="outfit",
            item_names=["A", "B"],
            ax=axes,
            width=0.5,
        )

        assert result is axes
        assert axes.calls["bar"][0][1]["color"] == ["steelblue", "tomato"]
        assert axes.calls["bar"][0][1]["width"] == 0.5
        assert len(axes.calls["scatter"]) == 2

    def test_custom_color_does_not_add_misleading_color_legend(self):
        axes = RecordingAxes()

        plotting.plot_itemfit({"infit": np.array([1.0])}, ax=axes, color="black")

        assert axes.calls["bar"][0][1]["color"] == "black"
        assert not axes.calls["scatter"]
        assert not axes.calls["legend"]

    @pytest.mark.parametrize(
        ("fit_stats", "kwargs", "match"),
        [
            ([], {}, "mapping"),
            ({"infit": np.array([1.0])}, {"statistic": "outfit"}, "do not contain"),
            ({"infit": np.array([])}, {}, "nonempty"),
            ({"infit": np.ones((1, 1))}, {}, "one-dimensional"),
            ({"infit": np.array([np.nan])}, {}, "finite"),
            ({"infit": np.ones(2)}, {"criterion": (1.3, 0.7)}, "increasing"),
            ({"infit": np.ones(2)}, {"item_names": ["one"]}, "2 entries"),
        ],
    )
    def test_invalid_itemfit_inputs(self, fit_stats, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plotting.plot_itemfit(fit_stats, ax=RecordingAxes(), **kwargs)


class TestPersonItemMap:
    def test_difficulty_map_batches_item_markers_and_forwards_styles(self):
        axes = RecordingAxes()

        result = plotting.plot_person_item_map(
            BinaryModel(),
            np.array([-1.0, 0.0, 1.0]),
            ax=axes,
            color="gray",
            item_kwargs={"color": "purple", "s": 10},
        )

        assert result is axes
        assert axes.calls["hist"][0][1]["color"] == "gray"
        assert len(axes.calls["twiny"]) == 1
        assert not axes.calls["twinx"]
        assert axes.secondary is not None
        assert len(axes.secondary.calls["scatter"]) == 1
        np.testing.assert_allclose(
            axes.secondary.calls["scatter"][0][0][1], [-1.0, 0.0, 1.0]
        )
        assert axes.secondary.calls["scatter"][0][1]["color"] == "purple"
        assert len(axes.secondary.calls["annotate"]) == 3

    def test_threshold_map_ignores_padded_thresholds(self):
        axes = RecordingAxes()

        plotting.plot_person_item_map(PolytomousModel(), np.arange(5.0), ax=axes)

        assert axes.secondary is not None
        locations = axes.secondary.calls["scatter"][0][0][1]
        np.testing.assert_allclose(locations, [0.0, 0.0])

    def test_multidimensional_locations_use_selected_slope(self):
        axes = RecordingAxes()
        theta = np.column_stack((np.arange(3.0), np.arange(5.0, 8.0)))

        plotting.plot_person_item_map(MultidimensionalModel(), theta, factor=1, ax=axes)

        np.testing.assert_allclose(axes.calls["hist"][0][0][0], theta[:, 1])
        assert axes.secondary is not None
        np.testing.assert_allclose(
            axes.secondary.calls["scatter"][0][0][1], [-0.5, 2.0]
        )

    def test_multidimensional_difficulty_uses_selected_factor(self):
        model = BinaryModel()
        model.n_factors = 2
        model.parameters = {
            "difficulty": np.array([[-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
        }
        theta = np.column_stack((np.arange(3.0), np.arange(4.0, 7.0)))
        axes = RecordingAxes()

        plotting.plot_person_item_map(model, theta, factor=1, ax=axes)

        assert axes.secondary is not None
        np.testing.assert_allclose(
            axes.secondary.calls["scatter"][0][0][1], [1.0, 2.0, 3.0]
        )

    def test_intercepts_without_slopes_are_supported(self):
        model = BinaryModel()
        model.parameters = {"intercepts": np.array([1.0, 0.0, -1.0])}
        axes = RecordingAxes()

        plotting.plot_person_item_map(model, np.arange(3.0), ax=axes)

        assert axes.secondary is not None
        np.testing.assert_allclose(
            axes.secondary.calls["scatter"][0][0][1], [-1.0, 0.0, 1.0]
        )

    def test_thresholds_without_category_counts_use_row_means(self):
        model = PolytomousModel()
        model.n_categories = None
        model.parameters = {"thresholds": np.array([[-1.0, 1.0], [2.0, 4.0]])}
        axes = RecordingAxes()

        plotting.plot_person_item_map(model, np.arange(3.0), ax=axes)

        assert axes.secondary is not None
        np.testing.assert_allclose(axes.secondary.calls["scatter"][0][0][1], [0.0, 3.0])

    @pytest.mark.parametrize(
        ("parameters", "match"),
        [
            ({}, "does not expose"),
            ({"difficulty": np.ones((3, 2))}, "difficulty"),
            ({"difficulty": np.array([0.0, np.nan, 1.0])}, "finite"),
            ({"thresholds": np.ones(3)}, "threshold"),
            ({"intercepts": np.ones((3, 1))}, "intercept"),
            (
                {"intercepts": np.ones(3), "slopes": np.ones((3, 2))},
                "slope",
            ),
            ({"intercepts": np.ones(3), "slopes": np.zeros(3)}, "zero slopes"),
        ],
    )
    def test_invalid_item_locations(self, parameters, match):
        model = BinaryModel()
        model.parameters = parameters

        with pytest.raises(ValueError, match=match):
            plotting.plot_person_item_map(model, np.arange(3.0), ax=RecordingAxes())

    def test_invalid_bins_are_rejected(self):
        with pytest.raises(ValueError, match="valid histogram"):
            plotting.plot_person_item_map(
                BinaryModel(), np.arange(3.0), bins=0, ax=RecordingAxes()
            )

    def test_category_counts_must_match_threshold_width(self):
        model = PolytomousModel()
        model.parameters = {"thresholds": np.ones((2, 1))}

        with pytest.raises(ValueError, match="n_categories is inconsistent"):
            plotting.plot_person_item_map(model, np.arange(3.0), ax=RecordingAxes())

    def test_model_and_theta_factor_counts_must_agree(self):
        with pytest.raises(ValueError, match="factor must be in"):
            plotting.plot_person_item_map(
                BinaryModel(), np.ones((3, 2)), factor=1, ax=RecordingAxes()
            )

    def test_item_style_must_be_a_mapping(self):
        with pytest.raises(ValueError, match="item_kwargs must be a mapping"):
            plotting.plot_person_item_map(
                BinaryModel(), np.arange(3.0), item_kwargs=1, ax=RecordingAxes()
            )

    def test_model_parameters_must_be_a_mapping(self):
        model = BinaryModel()
        model.parameters = []

        with pytest.raises(ValueError, match="parameters must be a mapping"):
            plotting.plot_person_item_map(model, np.arange(3.0), ax=RecordingAxes())


class TestDifPlot:
    def test_dif_uses_absolute_effects_and_classification_colors(self):
        axes = RecordingAxes()

        result = plotting.plot_dif(
            {
                "effect_size": np.array([-0.1, 0.5, -0.8]),
                "classification": np.array(["a", "B", "C"]),
            },
            item_names=["A", "B", "C"],
            ax=axes,
            width=0.6,
        )

        assert result is axes
        np.testing.assert_allclose(axes.calls["bar"][0][0][1], [0.1, 0.5, 0.8])
        assert axes.calls["bar"][0][1]["color"] == ["green", "gold", "red"]
        assert axes.calls["bar"][0][1]["width"] == 0.6
        assert len(axes.calls["scatter"]) == 3

    def test_custom_color_skips_classification_handles(self):
        axes = RecordingAxes()

        plotting.plot_dif(
            {"effect_size": np.array([0.1]), "classification": np.array(["A"])},
            ax=axes,
            color="black",
        )

        assert not axes.calls["scatter"]
        assert len(axes.calls["legend"]) == 1

    @pytest.mark.parametrize(
        ("results", "kwargs", "match"),
        [
            ([], {}, "mapping"),
            ({"classification": np.array(["A"])}, {}, "effect_size"),
            ({"effect_size": np.array([0.1])}, {}, "classification"),
            (
                {
                    "effect_size": np.array([0.1, 0.2]),
                    "classification": np.array(["A"]),
                },
                {},
                "match effect-size",
            ),
            (
                {"effect_size": np.array([0.1]), "classification": np.array(["D"])},
                {},
                "only A, B, or C",
            ),
            (
                {"effect_size": np.array([0.1]), "classification": np.array(["A"])},
                {"item_names": ["one", "two"]},
                "1 entries",
            ),
        ],
    )
    def test_invalid_dif_inputs(self, results, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plotting.plot_dif(results, ax=RecordingAxes(), **kwargs)


class TestExpectedScoreAndStandardError:
    def test_binary_expected_score_uses_one_probability_evaluation(self):
        model = BinaryModel()
        axes = RecordingAxes()

        plotting.plot_expected_score(model, n_points=5, ax=axes, color="navy")

        assert model.expected_score_calls == [None]
        assert model.probability_calls == [None]
        assert axes.calls["plot"][0][1]["color"] == "navy"
        assert axes.calls["plot"][0][0][1].shape == (5,)

    def test_polytomous_expected_score_is_vectorized_and_score_scaled(self):
        model = PolytomousModel()
        axes = RecordingAxes()

        plotting.plot_expected_score(model, n_points=5, ax=axes)

        assert model.expected_score_calls == [None]
        assert model.probability_calls == []
        np.testing.assert_allclose(axes.calls["plot"][0][0][1], 3.1)
        assert axes.calls["set_ylim"][0][0] == (0, 5.1)

    def test_one_item_probability_vector_is_supported(self):
        model = BinaryModel()
        model.n_items = 1
        model.item_names = ["Only"]
        model.parameters = {"difficulty": np.array([0.0])}
        axes = RecordingAxes()

        plotting.plot_expected_score(model, n_points=5, ax=axes)

        assert axes.calls["plot"][0][0][1].shape == (5,)

    def test_one_item_expected_score_vector_is_supported(self):
        model = BinaryModel()
        model.n_items = 1
        model.item_names = ["Only"]
        model.parameters = {"difficulty": np.array([0.0])}
        model.expected_score = lambda theta, item_idx=None: np.linspace(0.1, 0.9, 5)
        axes = RecordingAxes()

        plotting.plot_expected_score(model, n_points=5, ax=axes)

        np.testing.assert_allclose(
            axes.calls["plot"][0][0][1], np.linspace(0.1, 0.9, 5)
        )

    def test_expected_score_uses_selected_factor(self):
        model = MultidimensionalModel()

        plotting.plot_expected_score(model, factor=1, n_points=5, ax=RecordingAxes())

        assert model.last_theta is not None
        np.testing.assert_allclose(model.last_theta[:, 0], 0.0)
        np.testing.assert_allclose(model.last_theta[:, 1], np.linspace(-4, 4, 5))

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (np.ones(4), "shape"),
            (np.ones((5, 1)), "shape"),
            (np.full(5, np.nan), "finite"),
            (np.full(5, -0.1), "nonnegative"),
            (np.full(5, 4.0), "model maximum"),
        ],
    )
    def test_invalid_expected_scores(self, output, match):
        model = BinaryModel()
        model.expected_score = lambda theta, item_idx=None: output

        with pytest.raises(ValueError, match=match):
            plotting.plot_expected_score(model, n_points=5, ax=RecordingAxes())

    def test_binary_standard_error_sums_item_information(self):
        model = BinaryModel()
        axes = RecordingAxes()

        plotting.plot_se(model, n_points=5, ax=axes, linewidth=4)

        assert model.information_calls == [None]
        assert axes.calls["plot"][0][1]["linewidth"] == 4
        assert axes.calls["plot"][0][0][1].shape == (5,)

    def test_polytomous_standard_error_accepts_total_curve(self):
        model = PolytomousModel()
        axes = RecordingAxes()

        plotting.plot_se(model, n_points=5, ax=axes)

        assert model.information_calls == [None]
        assert axes.calls["plot"][0][0][1].shape == (5,)

    def test_zero_information_produces_finite_standard_error(self):
        model = BinaryModel()
        model.information = lambda theta, item_idx=None: np.zeros(len(theta))
        axes = RecordingAxes()

        plotting.plot_se(model, n_points=5, ax=axes)

        assert np.all(np.isfinite(axes.calls["plot"][0][0][1]))


class TestRealModels:
    def test_dichotomous_model_supports_all_curve_types(self):
        model = TwoParameterLogistic(n_items=3)

        assert plotting.plot_icc(model, ax=RecordingAxes())
        assert plotting.plot_category_curves(model, 0, ax=RecordingAxes())
        assert plotting.plot_information(model, ax=RecordingAxes())
        assert plotting.plot_expected_score(model, ax=RecordingAxes())
        assert plotting.plot_se(model, ax=RecordingAxes())
        assert plotting.plot_person_item_map(model, np.arange(5.0), ax=RecordingAxes())

    def test_graded_response_model_supports_all_curve_types(self):
        model = GradedResponseModel(n_items=2, n_categories=[3, 4])

        assert plotting.plot_icc(model, ax=RecordingAxes())
        assert plotting.plot_category_curves(model, 1, ax=RecordingAxes())
        assert plotting.plot_information(model, item_idx=[0, 1], ax=RecordingAxes())
        assert plotting.plot_expected_score(model, ax=RecordingAxes())
        assert plotting.plot_se(model, ax=RecordingAxes())
        assert plotting.plot_person_item_map(model, np.arange(5.0), ax=RecordingAxes())

    def test_multidimensional_model_supports_factor_slices(self):
        model = RealMultidimensionalModel(n_items=2, n_factors=2)
        theta = np.column_stack((np.arange(5.0), np.arange(5.0)))

        assert plotting.plot_icc(model, factor=1, ax=RecordingAxes())
        assert plotting.plot_information(model, factor=1, ax=RecordingAxes())
        assert plotting.plot_expected_score(model, factor=1, ax=RecordingAxes())
        assert plotting.plot_se(model, factor=1, ax=RecordingAxes())
        assert plotting.plot_person_item_map(model, theta, factor=1, ax=RecordingAxes())
