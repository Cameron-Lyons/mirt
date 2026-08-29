"""Regression tests for the lightweight public utility namespace."""

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
        "export_count": 119,
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


def test_star_import_resolves_every_public_utility_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.utils as utils

        namespace = {}
        exec("from mirt.utils import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(utils.__all__),
            "export_count": len(exported),
            "all_cached": all(name in utils.__dict__ for name in utils.__all__),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 119,
        "all_cached": True,
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
