"""Regression tests for the lightweight public CAT namespace."""

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


def test_plain_cat_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.cat as cat

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "cat_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.cat.")
            ),
            "export_count": len(cat.__all__),
            "exports_visible": all(name in dir(cat) for name in cat.__all__),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "cat_submodules": [],
        "export_count": 56,
        "exports_visible": True,
    }


def test_cat_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.cat as cat

        deferred_before_access = "ContentArea" not in cat.__dict__
        symbol = cat.ContentArea

        from mirt.cat.content import ContentArea

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": cat.__dict__["ContentArea"] is symbol,
            "same_symbol": symbol is ContentArea,
            "cat_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.cat.")
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
        "cat_submodules": ["mirt.cat.content"],
        "numpy_loaded": False,
        "scipy_loaded": False,
    }


def test_top_level_and_cat_exports_preserve_identity() -> None:
    result = _run_probe(
        """
        import json

        import mirt
        import mirt.cat as cat

        top_level_symbol = mirt.CATEngine
        cat_symbol = cat.CATEngine

        print(json.dumps({
            "same_symbol": top_level_symbol is cat_symbol,
            "top_level_cached": mirt.__dict__["CATEngine"] is top_level_symbol,
            "cat_cached": cat.__dict__["CATEngine"] is cat_symbol,
        }))
        """
    )

    assert result == {
        "same_symbol": True,
        "top_level_cached": True,
        "cat_cached": True,
    }


def test_star_import_resolves_every_public_cat_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.cat as cat

        namespace = {}
        exec("from mirt.cat import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(cat.__all__),
            "export_count": len(exported),
            "all_cached": all(name in cat.__dict__ for name in cat.__all__),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 56,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.cat as cat
        from mirt.cat import selection

        try:
            cat.NotACATComponent
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": selection.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.cat.selection",
        "error": "module 'mirt.cat' has no attribute 'NotACATComponent'",
    }
