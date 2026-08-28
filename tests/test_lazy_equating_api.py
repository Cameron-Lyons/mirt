"""Regression tests for the lightweight public equating namespace."""

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


def test_plain_equating_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.equating as equating

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "equating_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.equating.")
            ),
            "export_count": len(equating.__all__),
            "exports_visible": all(
                name in dir(equating) for name in equating.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "equating_submodules": [],
        "export_count": 57,
        "exports_visible": True,
    }


def test_equating_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.equating as equating

        deferred_before_access = "GradeData" not in equating.__dict__
        symbol = equating.GradeData

        from mirt.equating.vertical import GradeData

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": equating.__dict__["GradeData"] is symbol,
            "same_symbol": symbol is GradeData,
            "equating_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.equating.")
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
        "equating_submodules": ["mirt.equating.vertical"],
        "numpy_loaded": True,
        "scipy_loaded": False,
    }


def test_top_level_and_equating_exports_preserve_identity() -> None:
    result = _run_probe(
        """
        import json

        import mirt
        import mirt.equating as equating

        top_level_symbol = mirt.LinkingConstants
        equating_symbol = equating.LinkingConstants

        print(json.dumps({
            "same_symbol": top_level_symbol is equating_symbol,
            "top_level_cached": mirt.__dict__["LinkingConstants"]
                is top_level_symbol,
            "equating_cached": equating.__dict__["LinkingConstants"]
                is equating_symbol,
        }))
        """
    )

    assert result == {
        "same_symbol": True,
        "top_level_cached": True,
        "equating_cached": True,
    }


def test_star_import_resolves_every_public_equating_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.equating as equating

        namespace = {}
        exec("from mirt.equating import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(equating.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in equating.__dict__ for name in equating.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 57,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.equating as equating
        from mirt.equating import score_equating

        try:
            equating.NotAnEquatingMethod
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": score_equating.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.equating.score_equating",
        "error": "module 'mirt.equating' has no attribute 'NotAnEquatingMethod'",
    }
