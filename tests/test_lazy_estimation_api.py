"""Regression tests for the lightweight public estimation namespace."""

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


def test_plain_estimation_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.estimation as estimation

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "estimation_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.estimation.")
            ),
            "export_count": len(estimation.__all__),
            "exports_visible": all(
                name in dir(estimation) for name in estimation.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "estimation_submodules": [],
        "export_count": 53,
        "exports_visible": True,
    }


def test_estimation_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.estimation as estimation

        deferred_before_access = "ParameterConstraint" not in estimation.__dict__
        symbol = estimation.ParameterConstraint

        from mirt.estimation.constraints import ParameterConstraint

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": estimation.__dict__["ParameterConstraint"] is symbol,
            "same_symbol": symbol is ParameterConstraint,
            "estimation_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.estimation.")
            ),
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "estimation_submodules": ["mirt.estimation.constraints"],
        "numpy_loaded": True,
        "scipy_loaded": False,
    }


def test_top_level_and_estimation_exports_preserve_identity() -> None:
    result = _run_probe(
        """
        import json

        import mirt
        import mirt.estimation as estimation

        top_level_symbol = mirt.EMEstimator
        estimation_symbol = estimation.EMEstimator

        print(json.dumps({
            "same_symbol": top_level_symbol is estimation_symbol,
            "top_level_cached": mirt.__dict__["EMEstimator"]
                is top_level_symbol,
            "estimation_cached": estimation.__dict__["EMEstimator"]
                is estimation_symbol,
        }))
        """
    )

    assert result == {
        "same_symbol": True,
        "top_level_cached": True,
        "estimation_cached": True,
    }


def test_star_import_resolves_every_public_estimation_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.estimation as estimation

        namespace = {}
        exec("from mirt.estimation import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(estimation.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in estimation.__dict__ for name in estimation.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 53,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.estimation as estimation
        from mirt.estimation import regularized

        try:
            estimation.NotAnEstimator
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": regularized.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.estimation.regularized",
        "error": "module 'mirt.estimation' has no attribute 'NotAnEstimator'",
    }
