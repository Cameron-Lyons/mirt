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

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mirt.reports.dif_analysis import DIFAnalysisReport
from mirt.reports.full_diagnostic import FullDiagnosticReport
from mirt.reports.item_analysis import ItemAnalysisReport
from mirt.reports.model_fit import ModelFitReport

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.results.fit_result import FitResult

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
    **kwargs: str,
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
        Additional arguments passed to report builder.

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
    builders = {
        "item_analysis": ItemAnalysisReport,
        "model_fit": ModelFitReport,
        "full_diagnostic": FullDiagnosticReport,
    }

    if report_type not in builders:
        raise ValueError(
            f"Unknown report type: {report_type}. Available: {list(builders.keys())}"
        )

    builder_cls = builders[report_type]

    if report_type in ("item_analysis", "full_diagnostic"):
        report = builder_cls(fit_result, responses, theta=theta, **kwargs)
    else:
        report = builder_cls(fit_result, responses, **kwargs)

    if output_path:
        report.save(output_path)

    return report.generate()
