"""Regression tests for the lightweight public reports namespace."""

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


def test_plain_reports_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.reports as reports

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
            "report_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.reports.")
            ),
            "export_count": len(reports.__all__),
            "exports_visible": all(
                name in dir(reports) for name in reports.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "scipy_loaded": False,
        "matplotlib_loaded": False,
        "report_submodules": [],
        "export_count": 5,
        "exports_visible": True,
    }


def test_local_report_function_stays_lightweight_and_shared() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt
        import mirt.reports as reports

        top_level_function = mirt.generate_report

        print(json.dumps({
            "same_function": top_level_function is reports.generate_report,
            "top_level_cached": mirt.__dict__["generate_report"]
                is top_level_function,
            "numpy_loaded": "numpy" in sys.modules,
            "scipy_loaded": "scipy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
            "report_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.reports.")
            ),
        }))
        """
    )

    assert result == {
        "same_function": True,
        "top_level_cached": True,
        "numpy_loaded": False,
        "scipy_loaded": False,
        "matplotlib_loaded": False,
        "report_submodules": [],
    }


def test_report_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.reports as reports

        deferred_before_access = "ItemAnalysisReport" not in reports.__dict__
        symbol = reports.ItemAnalysisReport

        from mirt.reports.item_analysis import ItemAnalysisReport

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": reports.__dict__["ItemAnalysisReport"] is symbol,
            "same_symbol": symbol is ItemAnalysisReport,
            "report_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.reports.")
            ),
            "numpy_loaded": "numpy" in sys.modules,
            "matplotlib_loaded": "matplotlib" in sys.modules,
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "report_submodules": [
            "mirt.reports._base",
            "mirt.reports.item_analysis",
        ],
        "numpy_loaded": True,
        "matplotlib_loaded": False,
    }


def test_generate_report_loads_only_the_selected_builder() -> None:
    result = _run_probe(
        """
        import json
        import sys
        import types

        import mirt.reports as reports

        module = types.ModuleType("mirt.reports.model_fit")

        class ModelFitReport:
            def __init__(self, fit_result, responses, **kwargs):
                self.fit_result = fit_result
                self.responses = responses

            def generate(self):
                return "<html>selected builder</html>"

        module.ModelFitReport = ModelFitReport
        sys.modules[module.__name__] = module

        html = reports.generate_report(
            object(),
            object(),
            report_type="model_fit",
        )
        implementation_modules = (
            "mirt.reports.item_analysis",
            "mirt.reports.model_fit",
            "mirt.reports.full_diagnostic",
            "mirt.reports.dif_analysis",
        )

        print(json.dumps({
            "html": html,
            "loaded_implementations": [
                name for name in implementation_modules if name in sys.modules
            ],
        }))
        """
    )

    assert result == {
        "html": "<html>selected builder</html>",
        "loaded_implementations": ["mirt.reports.model_fit"],
    }


def test_star_import_resolves_every_public_report_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.reports as reports

        namespace = {}
        exec("from mirt.reports import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(reports.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in reports.__dict__ for name in reports.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 5,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.reports as reports
        from mirt.reports import model_fit

        try:
            reports.NotAReportBuilder
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": model_fit.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.reports.model_fit",
        "error": "module 'mirt.reports' has no attribute 'NotAReportBuilder'",
    }
