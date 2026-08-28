"""Regression tests for call-specific plotting dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_importing_plotting_defers_optional_dependencies() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.plotting

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
        }))
        """
    )

    assert result == {
        "numpy_loaded": True,
        "scipy_loaded": False,
        "matplotlib_loaded": False,
    }


def test_resolving_top_level_plotter_keeps_optional_dependencies_deferred() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt

        plotter = mirt.plot_ability_distribution
        print(json.dumps({
            "callable": callable(plotter),
            "cached": mirt.__dict__["plot_ability_distribution"] is plotter,
            "scipy_loaded": "scipy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
        }))
        """
    )

    assert result == {
        "callable": True,
        "cached": True,
        "scipy_loaded": False,
        "matplotlib_loaded": False,
    }


def test_scipy_is_loaded_only_for_a_requested_kde() -> None:
    result = _run_probe(
        """
        import json
        import sys

        from mirt.plotting import plot_ability_distribution

        class Axes:
            def __init__(self):
                self.labels = []

            def hist(self, *args, **kwargs):
                pass

            def plot(self, *args, **kwargs):
                self.labels.append(kwargs.get("label"))

            def set_xlabel(self, *args, **kwargs):
                pass

            def set_ylabel(self, *args, **kwargs):
                pass

            def set_title(self, *args, **kwargs):
                pass

            def legend(self, *args, **kwargs):
                pass

            def grid(self, *args, **kwargs):
                pass

        normal_axes = Axes()
        plot_ability_distribution(
            [-1.0, 0.0, 1.0],
            ax=normal_axes,
            show_density=False,
        )
        scipy_after_normal = "scipy" in sys.modules

        kde_axes = Axes()
        plot_ability_distribution(
            [-1.0, 0.0, 1.0],
            ax=kde_axes,
            show_density=True,
            show_normal=False,
        )
        print(json.dumps({
            "normal_labels": normal_axes.labels,
            "kde_labels": kde_axes.labels,
            "scipy_after_normal": scipy_after_normal,
            "scipy_after_kde": "scipy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
        }))
        """
    )

    assert result == {
        "normal_labels": ["N(0,1)"],
        "kde_labels": ["KDE"],
        "scipy_after_normal": False,
        "scipy_after_kde": True,
        "matplotlib_loaded": False,
    }
