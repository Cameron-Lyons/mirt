"""Regression tests for the lightweight public model namespace."""

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


def test_plain_models_import_defers_numerical_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.models as models

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "model_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.models.")
            ),
            "export_count": len(models.__all__),
            "exports_visible": all(name in dir(models) for name in models.__all__),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "model_submodules": [],
        "export_count": 100,
        "exports_visible": True,
    }


def test_model_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.models as models

        deferred_before_access = "TwoParameterLogistic" not in models.__dict__
        symbol = models.TwoParameterLogistic

        from mirt.models.dichotomous import TwoParameterLogistic

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": models.__dict__["TwoParameterLogistic"] is symbol,
            "same_symbol": symbol is TwoParameterLogistic,
            "model_submodules": sorted(
                name for name in sys.modules if name.startswith("mirt.models.")
            ),
            "scipy_loaded": "scipy" in sys.modules,
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "model_submodules": ["mirt.models.base", "mirt.models.dichotomous"],
        "scipy_loaded": False,
    }


def test_top_level_and_models_exports_preserve_identity() -> None:
    result = _run_probe(
        """
        import json

        import mirt
        import mirt.models as models

        top_level_symbol = mirt.TwoParameterLogistic
        models_symbol = models.TwoParameterLogistic

        print(json.dumps({
            "same_symbol": top_level_symbol is models_symbol,
            "top_level_cached": mirt.__dict__["TwoParameterLogistic"]
                is top_level_symbol,
            "models_cached": models.__dict__["TwoParameterLogistic"]
                is models_symbol,
        }))
        """
    )

    assert result == {
        "same_symbol": True,
        "top_level_cached": True,
        "models_cached": True,
    }


def test_star_import_resolves_every_public_model_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.models as models

        namespace = {}
        exec("from mirt.models import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(models.__all__),
            "export_count": len(exported),
            "all_cached": all(name in models.__dict__ for name in models.__all__),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 100,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.models as models
        from mirt.models import state_space

        try:
            models.NotAModel
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": state_space.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.models.state_space",
        "error": "module 'mirt.models' has no attribute 'NotAModel'",
    }
