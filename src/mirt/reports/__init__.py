"""HTML report generation for IRT analysis.

This module provides automated HTML report generation for IRT models.
Reports are static HTML files with embedded plots (no JavaScript required).

Report Types
------------
ItemAnalysisReport
    Item parameters, fit statistics, ICC plots, Wright map
ModelFitReport
    Global fit indices (M2, RMSEA, CFI, TLI, SRMSR), information functions
DIFAnalysisReport
    DIF statistics with ETS A/B/C classification and visualizations
FullDiagnosticReport
    Comprehensive report combining all diagnostics

Examples
--------
>>> from mirt import fit_mirt, fscores
>>> from mirt.reports import ItemAnalysisReport, generate_report

Generate item analysis report:

>>> result = fit_mirt(data, model="2PL")
>>> scores = fscores(result, data)
>>> report = ItemAnalysisReport(result, data, theta=scores.theta)
>>> report.save("item_analysis.html")

Use convenience function:

>>> generate_report(result, data, report_type="full_diagnostic",
...                 theta=scores.theta, output_path="report.html")

Generate DIF report:

>>> from mirt.diagnostics import compute_dif
>>> from mirt.reports import DIFAnalysisReport
>>> import numpy as np
>>> groups = np.array([0] * 250 + [1] * 250)
>>> dif_results = compute_dif(data, groups, model="2PL")
>>> report = DIFAnalysisReport(dif_results, data, groups)
>>> report.save("dif_analysis.html")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from mirt.results.fit_result import FitResult


_LAZY_IMPORTS = {
    "ItemAnalysisReport": ("mirt.reports.item_analysis", "ItemAnalysisReport"),
    "ModelFitReport": ("mirt.reports.model_fit", "ModelFitReport"),
    "DIFAnalysisReport": ("mirt.reports.dif_analysis", "DIFAnalysisReport"),
    "FullDiagnosticReport": (
        "mirt.reports.full_diagnostic",
        "FullDiagnosticReport",
    ),
}

_REPORT_BUILDERS = {
    "item_analysis": _LAZY_IMPORTS["ItemAnalysisReport"],
    "model_fit": _LAZY_IMPORTS["ModelFitReport"],
    "full_diagnostic": _LAZY_IMPORTS["FullDiagnosticReport"],
}

__all__ = [
    "ItemAnalysisReport",
    "ModelFitReport",
    "DIFAnalysisReport",
    "FullDiagnosticReport",
    "generate_report",
]


def generate_report(
    fit_result: FitResult,
    responses: NDArray[np.int_],
    report_type: Literal[
        "item_analysis", "model_fit", "full_diagnostic"
    ] = "item_analysis",
    theta: NDArray[np.float64] | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """Generate an IRT report.

    Convenience function for generating reports without explicitly
    instantiating report builder classes.

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result.
    responses : ndarray
        Response matrix (n_persons x n_items).
    report_type : str
        Type of report to generate:
        - 'item_analysis': Item parameters, fit statistics, ICC plots
        - 'model_fit': Global fit indices and interpretation
        - 'full_diagnostic': Comprehensive diagnostic report
    theta : ndarray, optional
        Ability estimates. Required for Wright map in item_analysis
        and full_diagnostic reports.
    output_path : str or Path, optional
        If provided, save report to this path.
    **kwargs
        Additional arguments passed to the report builder. Use
        ``include_plots=False`` to generate a lightweight report without
        importing matplotlib.

    Returns
    -------
    str
        HTML report content.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> result = fit_mirt(data, model="2PL")
    >>> html = generate_report(result, data, report_type="model_fit")
    >>> generate_report(result, data, report_type="full_diagnostic",
    ...                 output_path="report.html")
    """
    if report_type not in _REPORT_BUILDERS:
        raise ValueError(
            f"Unknown report type: {report_type}. Available: {list(_REPORT_BUILDERS)}"
        )

    module_name, class_name = _REPORT_BUILDERS[report_type]
    module = importlib.import_module(module_name)
    builder_cls = getattr(module, class_name)

    if report_type in ("item_analysis", "full_diagnostic"):
        report = builder_cls(fit_result, responses, theta=theta, **kwargs)
    else:
        report = builder_cls(fit_result, responses, **kwargs)

    html = report.generate()
    if output_path is not None:
        report._write_html(output_path, html)

    return html


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.reports' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
