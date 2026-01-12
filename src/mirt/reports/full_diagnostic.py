"""Full diagnostic report builder."""

from __future__ import annotations

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
    ) -> None:
        super().__init__(fit_result, title)
        self.responses = np.asarray(responses)
        self.theta = theta
        self.include_ld = include_ld

    def _build_content(self) -> str:
        from mirt.diagnostics.itemfit import compute_itemfit
        from mirt.diagnostics.modelfit import compute_fit_indices
        from mirt.reports._plots import (
            create_ability_distribution_base64,
            create_icc_plot_base64,
            create_information_plot_base64,
            create_itemfit_plot_base64,
            create_se_plot_base64,
            create_wright_map_base64,
        )
        from mirt.reports._templates import embedded_plot, section

        sections = []

        sections.append(self._build_model_summary_section())
        sections.append(self._build_parameter_section())

        fit_stats = compute_itemfit(self.fit_result.model, self.responses)
        sections.append(self._build_itemfit_section(fit_stats))

        fit_indices = compute_fit_indices(self.fit_result.model, self.responses)
        sections.append(self._build_modelfit_section(fit_indices))

        if self.include_ld:
            try:
                from mirt.diagnostics.ld import compute_ld_statistics

                ld_results = compute_ld_statistics(
                    self.fit_result.model, self.responses, self.theta
                )
                sections.append(self._build_ld_section(ld_results))
            except Exception:
                pass

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
                "Test Information", embedded_plot(info_base64, "Information"), level=3
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

            wright_base64 = create_wright_map_base64(self.fit_result.model, self.theta)
            sections.append(
                section(
                    "Wright Map", embedded_plot(wright_base64, "Wright Map"), level=3
                )
            )

        return "\n".join(sections)

    def _build_model_summary_section(self) -> str:
        from mirt.reports._templates import format_value, section, summary_box

        model = self.fit_result.model
        stats = self.fit_result.fit_statistics()
        html = f"""
        <p><strong>Model Type:</strong> {model.model_name}</p>
        <p><strong>Items:</strong> {model.n_items} | <strong>Factors:</strong> {model.n_factors}</p>
        <p><strong>Persons:</strong> {stats["n_observations"]} | <strong>Parameters:</strong> {stats["n_parameters"]}</p>
        <p><strong>Log-Likelihood:</strong> {format_value(stats["log_likelihood"], ".2f")}</p>
        <p><strong>AIC:</strong> {format_value(stats["aic"], ".2f")} | <strong>BIC:</strong> {format_value(stats["bic"], ".2f")}</p>
        <p><strong>Converged:</strong> {stats["converged"]} ({stats["n_iterations"]} iterations)</p>
        """
        return section("Model Summary", summary_box(html))

    def _build_parameter_section(self) -> str:
        from scipy import stats as scipy_stats

        from mirt.reports._templates import format_value, section, table_from_data

        model = self.fit_result.model
        se = self.fit_result.standard_errors
        z_crit = scipy_stats.norm.ppf(0.975)

        headers = ["Item", "Parameter", "Estimate", "SE", "95% CI"]
        rows: list[list[str]] = []

        for param_name, values in model.parameters.items():
            param_se = se.get(param_name, np.zeros_like(values))
            if values.ndim == 1:
                for i in range(len(values)):
                    item_name = (
                        model.item_names[i]
                        if model.item_names and i < len(model.item_names)
                        else f"Item_{i + 1}"
                    )
                    est = values[i]
                    err = param_se[i] if i < len(param_se) else np.nan
                    if err > 0 and not np.isnan(err):
                        ci_lo, ci_hi = est - z_crit * err, est + z_crit * err
                        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
                    else:
                        ci_str = "NA"
                    rows.append(
                        [
                            item_name,
                            param_name,
                            format_value(est, ".4f"),
                            format_value(err, ".4f"),
                            ci_str,
                        ]
                    )

        return section("Item Parameters", table_from_data(headers, rows))

    def _build_itemfit_section(self, fit_stats: dict[str, NDArray[np.float64]]) -> str:
        from mirt.reports._templates import format_value, section, table_from_data

        model = self.fit_result.model
        headers = ["Item", "Infit", "Outfit", "Flag"]
        rows: list[list[str]] = []

        infit = fit_stats.get("infit", np.ones(model.n_items))
        outfit = fit_stats.get("outfit", np.ones(model.n_items))

        for i in range(model.n_items):
            item_name = (
                model.item_names[i]
                if model.item_names and i < len(model.item_names)
                else f"Item_{i + 1}"
            )
            flag = ""
            quality = None
            if infit[i] < 0.7 or infit[i] > 1.3 or outfit[i] < 0.7 or outfit[i] > 1.3:
                flag = "Misfit"
                quality = "poor"
            rows.append(
                [
                    item_name,
                    format_value(infit[i], ".3f", quality),
                    format_value(outfit[i], ".3f", quality),
                    flag,
                ]
            )

        return section("Item Fit Statistics", table_from_data(headers, rows))

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
        mean_q3 = np.mean(q3_upper)
        max_q3 = np.max(np.abs(q3_upper))
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
