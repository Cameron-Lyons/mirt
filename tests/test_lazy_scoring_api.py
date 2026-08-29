"""Regression tests for the lightweight public scoring namespace."""

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


def test_plain_scoring_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.scoring as scoring

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "scoring_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.scoring.")
            ),
            "export_count": len(scoring.__all__),
            "exports_visible": all(
                name in dir(scoring) for name in scoring.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "scoring_submodules": [],
        "export_count": 9,
        "exports_visible": True,
    }


def test_local_scoring_function_stays_lightweight_and_shared() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt
        import mirt.scoring as scoring

        top_level_function = mirt.fscores

        print(json.dumps({
            "same_function": top_level_function is scoring.fscores,
            "top_level_cached": mirt.__dict__["fscores"]
                is top_level_function,
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "scoring_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.scoring.")
            ),
        }))
        """
    )

    assert result == {
        "same_function": True,
        "top_level_cached": True,
        "numpy_loaded": False,
        "scipy_loaded": False,
        "scoring_submodules": [],
    }


def test_scoring_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.scoring as scoring

        deferred_before_access = "EAPScorer" not in scoring.__dict__
        symbol = scoring.EAPScorer

        from mirt.scoring.eap import EAPScorer

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": scoring.__dict__["EAPScorer"] is symbol,
            "same_symbol": symbol is EAPScorer,
            "scoring_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.scoring.")
            ),
            "unrelated_scorers_loaded": any(
                name in sys.modules for name in (
                    "mirt.scoring.eapsum",
                    "mirt.scoring.map",
                    "mirt.scoring.ml",
                    "mirt.scoring.wle",
                )
            ),
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "scoring_submodules": ["mirt.scoring._common", "mirt.scoring.eap"],
        "unrelated_scorers_loaded": False,
    }


def test_fscores_loads_only_the_selected_implementation() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import numpy as np

        from mirt.models.dichotomous import TwoParameterLogistic
        from mirt.scoring import fscores

        model = TwoParameterLogistic(n_items=2)
        model.set_parameters(
            discrimination=np.ones(2),
            difficulty=np.zeros(2),
        )
        model._is_fitted = True

        score_result = fscores(
            model,
            np.array([[1, 0]], dtype=int),
            method="EAP",
            n_quadpts=5,
        )
        implementation_modules = (
            "mirt.scoring.eap",
            "mirt.scoring.eapsum",
            "mirt.scoring.map",
            "mirt.scoring.ml",
            "mirt.scoring.wle",
        )

        print(json.dumps({
            "method": score_result.method,
            "loaded_implementations": [
                name for name in implementation_modules if name in sys.modules
            ],
        }))
        """
    )

    assert result == {
        "method": "EAP",
        "loaded_implementations": ["mirt.scoring.eap"],
    }


def test_star_import_resolves_every_public_scoring_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.scoring as scoring

        namespace = {}
        exec("from mirt.scoring import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(scoring.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in scoring.__dict__ for name in scoring.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 9,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.scoring as scoring
        from mirt.scoring import ml

        try:
            scoring.NotAScoringMethod
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": ml.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.scoring.ml",
        "error": "module 'mirt.scoring' has no attribute 'NotAScoringMethod'",
    }
