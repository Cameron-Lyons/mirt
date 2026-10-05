"""Full diagnostic report builder."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from mirt.reports._base import ReportBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mirt.diagnostics.ld import LDResult
    from mirt.results.fit_result import FitResult


class FullDiagnosticReport(ReportBuilder):
    """Generate comprehensive HTML diagnostic report.

    This report includes all available diagnostics:
    - Model summary and parameters with confidence intervals
    - Item fit statistics (infit/outfit)
    - Model fit indices (M2, RMSEA, CFI, TLI, SRMSR)
    - Local dependence analysis (Q3 matrix, flagged pairs)
    - Ability distribution (if theta provided)
    - All relevant plots (ICC, information, Wright map, fit plots)

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result.
    responses : ndarray
        Response matrix (n_persons x n_items).
    theta : ndarray, optional
        Ability estimates. If None, some plots are omitted.
    include_ld : bool
        Whether to include local dependence analysis. Default True.
    title : str, optional
        Report title.
    include_plots : bool
        Embed visualizations in the report. Set to False for a lightweight
        report that does not require matplotlib. Default True.

    Examples
    --------
    >>> from mirt import fit_mirt, fscores
    >>> from mirt.reports import FullDiagnosticReport
    >>> result = fit_mirt(data, model="2PL")
    >>> scores = fscores(result, data)
    >>> report = FullDiagnosticReport(result, data, theta=scores.theta)
    >>> report.save("full_diagnostic.html")
    """

    default_title = "Full IRT Diagnostic Report"

    def __init__(
        self,
        fit_result: FitResult,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64] | None = None,
        include_ld: bool = True,
        title: str | None = None,
        include_plots: bool = True,
    ) -> None:
        super().__init__(fit_result, title)
        self.responses = np.asarray(responses)
        self.theta = theta
        self.include_ld = include_ld
        self.include_plots = include_plots

    def _build_content(self) -> str:
        from mirt.diagnostics.itemfit import compute_itemfit
        from mirt.diagnostics.modelfit import compute_fit_indices
        from mirt.reports._templates import section

        sections = []

        sections.append(self._model_summary_section("Model Type"))
        sections.append(
            section("Item Parameters", self._parameter_table(include_tests=False))
        )

        fit_stats = compute_itemfit(self.fit_result, self.responses)
        sections.append(
            section(
                "Item Fit Statistics",
                self._itemfit_table(fit_stats, include_guide=False),
            )
        )

        fit_indices = compute_fit_indices(self.fit_result, self.responses)
        sections.append(self._build_modelfit_section(fit_indices))

        if self.include_ld:
            try:
                from mirt.diagnostics.ld import compute_ld_statistics

                ld_results = compute_ld_statistics(
                    self.fit_result, self.responses, self.theta
                )
                sections.append(self._build_ld_section(ld_results))
            except (
                ImportError,
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                warnings.warn(
                    f"Skipping local dependence section due to diagnostic error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.include_plots:
            from mirt.reports._plots import (
                create_ability_distribution_base64,
                create_icc_plot_base64,
                create_information_plot_base64,
                create_itemfit_plot_base64,
                create_se_plot_base64,
                create_wright_map_base64,
            )
            from mirt.reports._templates import embedded_plot

            sections.append(section("Visualizations", ""))

            icc_base64 = create_icc_plot_base64(self.fit_result.model)
            sections.append(
                section(
                    "Item Characteristic Curves",
                    embedded_plot(icc_base64, "ICC"),
                    level=3,
                )
            )

            info_base64 = create_information_plot_base64(self.fit_result.model)
            sections.append(
                section(
                    "Test Information",
                    embedded_plot(info_base64, "Information"),
                    level=3,
                )
            )

            se_base64 = create_se_plot_base64(self.fit_result.model)
            sections.append(
                section("Standard Error", embedded_plot(se_base64, "SE"), level=3)
            )

            itemfit_base64 = create_itemfit_plot_base64(
                fit_stats, item_names=self.fit_result.model.item_names
            )
            sections.append(
                section("Item Fit", embedded_plot(itemfit_base64, "Item Fit"), level=3)
            )

            if self.theta is not None:
                ability_base64 = create_ability_distribution_base64(self.theta)
                sections.append(
                    section(
                        "Ability Distribution",
                        embedded_plot(ability_base64, "Ability"),
                        level=3,
                    )
                )

                wright_base64 = create_wright_map_base64(
                    self.fit_result.model, self.theta
                )
                sections.append(
                    section(
                        "Wright Map",
                        embedded_plot(wright_base64, "Wright Map"),
                        level=3,
                    )
                )

        return "\n".join(sections)

    def _build_modelfit_section(self, fit_indices: dict[str, float]) -> str:
        from mirt.reports._templates import format_value, section, table_from_data

        headers = ["Index", "Value"]
        rows = [
            [
                "M2",
                f"{fit_indices['M2']:.2f} (df = {fit_indices['M2_df']:.0f}, p = {fit_indices['M2_p']:.4f})",
            ],
            [
                "RMSEA",
                f"{fit_indices['RMSEA']:.4f} [{fit_indices['RMSEA_CI_lower']:.4f}, {fit_indices['RMSEA_CI_upper']:.4f}]",
            ],
            ["CFI", format_value(fit_indices["CFI"], ".4f")],
            ["TLI", format_value(fit_indices["TLI"], ".4f")],
            ["SRMSR", format_value(fit_indices["SRMSR"], ".4f")],
        ]
        return section("Model Fit Indices", table_from_data(headers, rows))

    def _build_ld_section(self, ld_results: LDResult) -> str:
        from mirt.reports._templates import section, summary_box, table_from_data

        q3_upper = ld_results.q3_matrix[np.triu_indices_from(ld_results.q3_matrix, k=1)]
        mean_q3 = float(np.mean(q3_upper)) if q3_upper.size else np.nan
        max_q3 = float(np.max(np.abs(q3_upper))) if q3_upper.size else np.nan
        n_flagged = len(ld_results.q3_flagged)

        summary_html = f"""
        <p><strong>Mean Q3:</strong> {mean_q3:.4f}</p>
        <p><strong>Max |Q3|:</strong> {max_q3:.4f}</p>
        <p><strong>Pairs with |Q3| &gt; 0.2:</strong> {n_flagged}</p>
        """
        summary = section(
            "Local Dependence Summary", summary_box(summary_html), level=3
        )

        if ld_results.q3_flagged:
            headers = ["Item 1", "Item 2", "Q3"]
            rows: list[list[str]] = []
            sorted_flagged = sorted(ld_results.q3_flagged, key=lambda x: -abs(x[2]))[
                :10
            ]
            for i, j, q3 in sorted_flagged:
                name_i = (
                    ld_results.item_names[i]
                    if ld_results.item_names
                    else f"Item {i + 1}"
                )
                name_j = (
                    ld_results.item_names[j]
                    if ld_results.item_names
                    else f"Item {j + 1}"
                )
                rows.append([name_i, name_j, f"{q3:.4f}"])
            flagged_table = section(
                "Flagged Item Pairs (|Q3| > 0.2, top 10)",
                table_from_data(headers, rows),
                level=3,
            )
        else:
            flagged_table = section(
                "Flagged Item Pairs",
                "<p>No pairs flagged for local dependence.</p>",
                level=3,
            )

        return section("Local Dependence Analysis", summary + flagged_table)
