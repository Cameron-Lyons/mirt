"""Item analysis report builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mirt.reports._base import ReportBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mirt.results.fit_result import FitResult


class ItemAnalysisReport(ReportBuilder):
    """Generate HTML report for item analysis.

    This report includes:
    - Model summary (type, items, factors, fit statistics)
    - Item parameter table with estimates, standard errors, and confidence intervals
    - Item fit statistics (infit/outfit) with flagging for misfit
    - Item Characteristic Curves plot
    - Test Information Function plot
    - Wright map (person-item map) if theta estimates provided

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result.
    responses : ndarray
        Response matrix (n_persons x n_items) for computing fit statistics.
    theta : ndarray, optional
        Ability estimates for Wright map. If None, Wright map is omitted.
    title : str, optional
        Report title.

    Examples
    --------
    >>> from mirt import fit_mirt, fscores
    >>> from mirt.reports import ItemAnalysisReport
    >>> result = fit_mirt(data, model="2PL")
    >>> scores = fscores(result, data)
    >>> report = ItemAnalysisReport(result, data, theta=scores.theta)
    >>> report.save("item_analysis.html")
    """

    default_title = "Item Analysis Report"

    def __init__(
        self,
        fit_result: FitResult,
        responses: NDArray[np.int_],
        theta: NDArray[np.float64] | None = None,
        title: str | None = None,
    ) -> None:
        super().__init__(fit_result, title)
        self.responses = np.asarray(responses)
        self.theta = theta

    def _build_content(self) -> str:
        from mirt.diagnostics.itemfit import compute_itemfit
        from mirt.reports._plots import (
            create_icc_plot_base64,
            create_information_plot_base64,
            create_itemfit_plot_base64,
            create_wright_map_base64,
        )
        from mirt.reports._templates import (
            embedded_plot,
            section,
        )

        sections = []

        sections.append(self._build_model_summary())

        sections.append(section("Item Parameters", self._build_parameter_table()))

        fit_stats = compute_itemfit(self.fit_result.model, self.responses)
        sections.append(
            section("Item Fit Statistics", self._build_fit_table(fit_stats))
        )

        plot_base64 = create_itemfit_plot_base64(
            fit_stats, item_names=self.fit_result.model.item_names
        )
        sections.append(
            section("Item Fit Plot", embedded_plot(plot_base64, "Item Fit Statistics"))
        )

        icc_base64 = create_icc_plot_base64(self.fit_result.model)
        sections.append(
            section("Item Characteristic Curves", embedded_plot(icc_base64, "ICC Plot"))
        )

        info_base64 = create_information_plot_base64(self.fit_result.model)
        sections.append(
            section(
                "Test Information Function",
                embedded_plot(info_base64, "Information Function"),
            )
        )

        if self.theta is not None:
            wright_base64 = create_wright_map_base64(self.fit_result.model, self.theta)
            sections.append(
                section(
                    "Person-Item Map (Wright Map)",
                    embedded_plot(wright_base64, "Wright Map"),
                )
            )

        return "\n".join(sections)

    def _build_model_summary(self) -> str:
        from mirt.reports._templates import format_value, section, summary_box

        model = self.fit_result.model
        stats = self.fit_result.fit_statistics()
        summary_html = f"""
        <p><strong>Model:</strong> {model.model_name}</p>
        <p><strong>Items:</strong> {model.n_items} | <strong>Factors:</strong> {model.n_factors}</p>
        <p><strong>Persons:</strong> {stats["n_observations"]} | <strong>Parameters:</strong> {stats["n_parameters"]}</p>
        <p><strong>Log-Likelihood:</strong> {format_value(stats["log_likelihood"], ".2f")}</p>
        <p><strong>AIC:</strong> {format_value(stats["aic"], ".2f")} | <strong>BIC:</strong> {format_value(stats["bic"], ".2f")}</p>
        <p><strong>Converged:</strong> {stats["converged"]} ({stats["n_iterations"]} iterations)</p>
        """
        return section("Model Summary", summary_box(summary_html))

    def _build_parameter_table(self) -> str:
        from scipy import stats as scipy_stats

        from mirt.reports._templates import format_value, table_from_data

        model = self.fit_result.model
        se = self.fit_result.standard_errors
        z_crit = scipy_stats.norm.ppf(0.975)

        headers = ["Item", "Parameter", "Estimate", "SE", "z", "p", "95% CI"]
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
                        z = est / err
                        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
                        ci_lo, ci_hi = est - z_crit * err, est + z_crit * err
                    else:
                        z = p = ci_lo = ci_hi = np.nan
                    ci_str = (
                        f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "NA"
                    )
                    rows.append(
                        [
                            item_name,
                            param_name,
                            format_value(est, ".4f"),
                            format_value(err, ".4f"),
                            format_value(z, ".3f"),
                            format_value(p, ".4f"),
                            ci_str,
                        ]
                    )

        return table_from_data(headers, rows)

    def _build_fit_table(self, fit_stats: dict[str, NDArray[np.float64]]) -> str:
        from mirt.reports._templates import format_value, summary_box, table_from_data

        model = self.fit_result.model
        headers = ["Item", "Infit", "Outfit", "Flag"]
        rows: list[list[str]] = []

        infit = fit_stats.get("infit", np.ones(model.n_items))
        outfit = fit_stats.get("outfit", np.ones(model.n_items))

        n_misfit = 0
        for i in range(model.n_items):
            item_name = (
                model.item_names[i]
                if model.item_names and i < len(model.item_names)
                else f"Item_{i + 1}"
            )
            infit_val = infit[i]
            outfit_val = outfit[i]
            flag = ""
            quality = None
            if (
                infit_val < 0.7
                or infit_val > 1.3
                or outfit_val < 0.7
                or outfit_val > 1.3
            ):
                flag = "Misfit"
                quality = "poor"
                n_misfit += 1
            elif (
                infit_val < 0.8
                or infit_val > 1.2
                or outfit_val < 0.8
                or outfit_val > 1.2
            ):
                flag = "Check"
                quality = "warning"
            rows.append(
                [
                    item_name,
                    format_value(infit_val, ".3f", quality),
                    format_value(outfit_val, ".3f", quality),
                    flag,
                ]
            )

        table = table_from_data(headers, rows)
        interpretation = summary_box(
            f"""
            <p><strong>Interpretation Guide:</strong></p>
            <ul>
                <li>Acceptable fit: 0.7 - 1.3 (lenient), 0.8 - 1.2 (strict)</li>
                <li>Items flagged for misfit: {n_misfit} / {model.n_items}</li>
            </ul>
            """
        )
        return table + interpretation
