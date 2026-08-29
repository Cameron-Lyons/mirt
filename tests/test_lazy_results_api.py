"""Regression tests for the lightweight public result namespace."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

EXPECTED_EXPORTS = ("AbilityPosteriorResult", "FitResult", "ScoreResult")


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_public_result_export_order_is_explicit() -> None:
    import mirt.results as results

    assert results.__all__ == list(EXPECTED_EXPORTS)


def test_plain_result_import_defers_numpy_and_implementations() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.results as results

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "result_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.results.")
            ),
            "exports_visible": all(
                name in dir(results) for name in results.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "result_submodules": [],
        "exports_visible": True,
    }


def test_result_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.results as results

        deferred_before_access = "ScoreResult" not in results.__dict__
        symbol = results.ScoreResult

        from mirt.results.score_result import ScoreResult

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": results.__dict__["ScoreResult"] is symbol,
            "same_symbol": symbol is ScoreResult,
            "fit_result_loaded": "mirt.results.fit_result" in sys.modules,
            "result_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.results.")
            ),
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "fit_result_loaded": False,
        "result_submodules": [
            "mirt.results._common",
            "mirt.results.score_result",
        ],
    }


def test_star_import_resolves_every_public_result() -> None:
    result = _run_probe(
        """
        import json

        import mirt.results as results

        namespace = {}
        exec("from mirt.results import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(results.__all__),
            "all_cached": all(
                name in results.__dict__ for name in results.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "all_cached": True,
    }


def test_result_submodule_import_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.results as results
        from mirt.results import fit_result

        try:
            results.NotAResult
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": fit_result.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.results.fit_result",
        "error": "module 'mirt.results' has no attribute 'NotAResult'",
    }
