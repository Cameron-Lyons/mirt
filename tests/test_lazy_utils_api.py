"""Regression tests for the lightweight public utility namespace."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_plain_utils_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.utils as utils

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "utils_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.utils.")
            ),
            "export_count": len(utils.__all__),
            "exports_visible": all(name in dir(utils) for name in utils.__all__),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "utils_submodules": [],
        "export_count": 120,
        "exports_visible": True,
    }


def test_utility_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.utils as utils

        deferred_before_access = "validate_responses" not in utils.__dict__
        symbol = utils.validate_responses

        from mirt.utils.data import validate_responses

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": utils.__dict__["validate_responses"] is symbol,
            "same_symbol": symbol is validate_responses,
            "utils_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.utils.")
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
        "utils_submodules": ["mirt.utils.data"],
        "numpy_loaded": True,
        "scipy_loaded": False,
    }


def test_top_level_and_utils_exports_preserve_identity() -> None:
    result = _run_probe(
        """
        import json

        import mirt
        import mirt.utils as utils

        top_level_symbol = mirt.validate_responses
        utils_symbol = utils.validate_responses

        print(json.dumps({
            "same_symbol": top_level_symbol is utils_symbol,
            "top_level_cached": mirt.__dict__["validate_responses"]
                is top_level_symbol,
            "utils_cached": utils.__dict__["validate_responses"] is utils_symbol,
        }))
        """
    )

    assert result == {
        "same_symbol": True,
        "top_level_cached": True,
        "utils_cached": True,
    }


@pytest.mark.parametrize("residuals_last", [False, True])
def test_star_import_resolves_every_public_utility_export(residuals_last: bool) -> None:
    result = _run_probe(
        f"""
        import importlib
        import json

        import mirt.utils as utils

        if {residuals_last!r}:
            utils.__all__ = sorted(utils.__all__, key=lambda name: name == "residuals")

        namespace = {{}}
        exec("from mirt.utils import *", namespace)
        exported = {{name for name in namespace if not name.startswith("__")}}
        residuals_module = importlib.import_module("mirt.utils.residuals")

        print(json.dumps({{
            "matches_all": exported == set(utils.__all__),
            "export_count": len(exported),
            "all_cached": all(name in utils.__dict__ for name in utils.__all__),
            "residuals_is_function": namespace["residuals"]
                is residuals_module.residuals,
        }}))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 120,
        "all_cached": True,
        "residuals_is_function": True,
    }


@pytest.mark.parametrize(
    "first_import",
    [
        "from mirt.utils import Q3",
        "from mirt.utils.residuals import Q3",
        "importlib.import_module('mirt.utils.residuals')",
        "from mirt.utils import residuals",
        "from mirt import Q3",
    ],
    ids=["utils-q3", "submodule-q3", "submodule", "function", "top-level-q3"],
)
def test_residuals_function_survives_submodule_import_order(first_import: str) -> None:
    result = _run_probe(
        f"""
        import importlib
        import json
        import sys
        import types

        import mirt.utils as utils

        {first_import}
        from mirt.utils import residuals

        cached = utils.__dict__["residuals"] is residuals
        residuals_module = importlib.import_module("mirt.utils.residuals")
        from mirt.utils.residuals import residuals as implementation
        from mirt import residuals as top_level

        print(json.dumps({{
            "callable": callable(residuals),
            "cached": cached and utils.__dict__["residuals"] is residuals,
            "same_symbol": residuals is implementation is top_level,
            "submodule_preserved": isinstance(residuals_module, types.ModuleType)
                and sys.modules["mirt.utils.residuals"] is residuals_module,
            "utils_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.utils.")
            ),
            "scipy_loaded": "scipy" in sys.modules,
        }}))
        """
    )

    assert result == {
        "callable": True,
        "cached": True,
        "same_symbol": True,
        "submodule_preserved": True,
        "utils_submodules": ["mirt.utils.data", "mirt.utils.residuals"],
        "scipy_loaded": False,
    }


def test_utils_module_assignments_keep_normal_behavior() -> None:
    result = _run_probe(
        """
        import json
        import types

        import mirt.utils as utils

        sentinel = object()
        utils.residuals = sentinel
        normal_assignment = utils.residuals is sentinel
        unrelated_module = types.ModuleType("unrelated")
        utils.residuals = unrelated_module
        unrelated_assignment = utils.residuals is unrelated_module
        incomplete_module = types.ModuleType("mirt.utils.residuals")
        utils.residuals = incomplete_module

        print(json.dumps({
            "normal_assignment": normal_assignment,
            "unrelated_assignment": unrelated_assignment,
            "incomplete_module_assignment": utils.residuals is incomplete_module,
        }))
        """
    )

    assert result == {
        "normal_assignment": True,
        "unrelated_assignment": True,
        "incomplete_module_assignment": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.utils as utils
        from mirt.utils import calibration

        try:
            utils.NotAUtility
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": calibration.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.utils.calibration",
        "error": "module 'mirt.utils' has no attribute 'NotAUtility'",
    }
