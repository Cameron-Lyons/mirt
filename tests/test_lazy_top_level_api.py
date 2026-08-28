"""Regression tests for the lightweight top-level package API."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from mirt._api_registry import BASE_EXPORTS, build_lazy_imports


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_plain_import_defers_heavy_dependencies_and_subpackages() -> None:
    loaded = _run_probe(
        """
        import importlib
        import json
        import sys

        importlib.__dict__.pop("util", None)
        import mirt

        deferred = (
            "numpy",
            "scipy",
            "scipy.optimize",
            "scipy.stats",
            "torch",
            "matplotlib",
            "mirt.mirt_rs",
            "mirt.cat",
            "mirt.estimation",
            "mirt.models",
            "mirt.utils",
        )
        print(json.dumps({name: name in sys.modules for name in deferred}))
        """
    )

    assert not any(loaded.values())


def test_optional_plotting_exports_are_stable_without_package_probe() -> None:
    result = _run_probe(
        """
        import importlib.util
        import json
        import sys

        original_find_spec = importlib.util.find_spec

        def guarded_find_spec(name, *args, **kwargs):
            if name == "matplotlib":
                raise AssertionError("optional plotting package was probed")
            return original_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = guarded_find_spec

        import mirt

        listed = "plot_icc" in mirt.__all__
        deferred_before_access = "plot_icc" not in mirt.__dict__
        plotter = mirt.plot_icc
        print(json.dumps({
            "listed": listed,
            "deferred_before_access": deferred_before_access,
            "callable": callable(plotter),
            "cached": mirt.__dict__["plot_icc"] is plotter,
            "matplotlib_loaded": "matplotlib" in sys.modules,
        }))
        """
    )

    assert result == {
        "listed": True,
        "deferred_before_access": True,
        "callable": True,
        "cached": True,
        "matplotlib_loaded": False,
    }


def test_lazy_symbol_resolution_is_cached_and_preserves_identity() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt

        models_loaded_before = "mirt.models" in sys.modules
        symbol = mirt.TwoParameterLogistic
        cached = mirt.__dict__["TwoParameterLogistic"] is symbol

        from mirt.models.dichotomous import TwoParameterLogistic

        print(json.dumps({
            "models_loaded_before": models_loaded_before,
            "models_loaded_after": "mirt.models.dichotomous" in sys.modules,
            "cached": cached,
            "same_symbol": symbol is TwoParameterLogistic,
        }))
        """
    )

    assert result == {
        "models_loaded_before": False,
        "models_loaded_after": True,
        "cached": True,
        "same_symbol": True,
    }


def test_lazy_registry_covers_every_nonlocal_base_export() -> None:
    local_exports = {"__version__", "fit_mirt", "itemfit", "personfit", "dif"}
    missing = set(BASE_EXPORTS) - set(build_lazy_imports()) - local_exports

    assert not missing
