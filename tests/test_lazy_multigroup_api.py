"""Regression tests for the lightweight public multigroup namespace."""

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


def test_plain_multigroup_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.multigroup as multigroup

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "multigroup_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.multigroup.")
            ),
            "export_count": len(multigroup.__all__),
            "exports_visible": all(
                name in dir(multigroup) for name in multigroup.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "multigroup_submodules": [],
        "export_count": 13,
        "exports_visible": True,
    }


def test_local_multigroup_function_stays_lightweight_and_shared() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt
        import mirt.multigroup as multigroup

        top_level_function = mirt.fit_multigroup

        print(json.dumps({
            "same_function": top_level_function is multigroup.fit_multigroup,
            "top_level_cached": mirt.__dict__["fit_multigroup"]
                is top_level_function,
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "multigroup_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.multigroup.")
            ),
        }))
        """
    )

    assert result == {
        "same_function": True,
        "top_level_cached": True,
        "numpy_loaded": False,
        "scipy_loaded": False,
        "multigroup_submodules": [],
    }


def test_multigroup_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.multigroup as multigroup

        deferred_before_access = "InvarianceSpec" not in multigroup.__dict__
        symbol = multigroup.InvarianceSpec

        from mirt.multigroup.invariance import InvarianceSpec

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": multigroup.__dict__["InvarianceSpec"] is symbol,
            "same_symbol": symbol is InvarianceSpec,
            "multigroup_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.multigroup.")
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
        "multigroup_submodules": ["mirt.multigroup.invariance"],
        "numpy_loaded": True,
        "scipy_loaded": False,
    }


def test_star_import_resolves_every_public_multigroup_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.multigroup as multigroup

        namespace = {}
        exec("from mirt.multigroup import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(multigroup.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in multigroup.__dict__ for name in multigroup.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 13,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.multigroup as multigroup
        from mirt.multigroup import latent

        try:
            multigroup.NotAMultigroupComponent
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": latent.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.multigroup.latent",
        "error": "module 'mirt.multigroup' has no attribute 'NotAMultigroupComponent'",
    }
