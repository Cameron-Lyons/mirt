"""Model fit report builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mirt.reports._base import ReportBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mirt.results.fit_result import FitResult


class ModelFitReport(ReportBuilder):
    """Generate HTML report for model fit assessment.

    This report includes:
    - Model summary (type, parameters, log-likelihood, AIC/BIC)
    - Global fit indices table (M2, RMSEA, CFI, TLI, SRMSR)
    - Fit interpretation guidelines
    - Test Information Function plot
    - Standard Error of Measurement plot

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result.
    responses : ndarray
        Response matrix (n_persons x n_items).
    title : str, optional
        Report title.

    Examples
    --------
    >>> from mirt import fit_mirt
    >>> from mirt.reports import ModelFitReport
    >>> result = fit_mirt(data, model="2PL")
    >>> report = ModelFitReport(result, data)
    >>> report.save("model_fit.html")
    """

    default_title = "Model Fit Report"

    def __init__(
        self,
        fit_result: FitResult,
        responses: NDArray[np.int_],
        title: str | None = None,
    ) -> None:
        super().__init__(fit_result, title)
        self.responses = np.asarray(responses)

    def _build_content(self) -> str:
        from mirt.diagnostics.modelfit import compute_fit_indices
        from mirt.reports._plots import (
            create_information_plot_base64,
            create_se_plot_base64,
        )
        from mirt.reports._templates import embedded_plot, section

        sections = []

        sections.append(self._build_model_summary())

        fit_indices = compute_fit_indices(self.fit_result.model, self.responses)
        sections.append(
            section("Fit Indices", self._build_fit_indices_table(fit_indices))
        )

        sections.append(
            section("Interpretation Guidelines", self._interpretation_guide())
        )

        info_base64 = create_information_plot_base64(self.fit_result.model)
        sections.append(
            section(
                "Test Information Function", embedded_plot(info_base64, "Information")
            )
        )

        se_base64 = create_se_plot_base64(self.fit_result.model)
        sections.append(
            section("Standard Error of Measurement", embedded_plot(se_base64, "SEM"))
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

    def _build_fit_indices_table(self, fit_indices: dict[str, float]) -> str:
        from mirt.reports._templates import format_value, table_from_data

        def quality_for_rmsea(val: float) -> str:
            if val < 0.05:
                return "good"
            if val < 0.08:
                return "warning"
            return "poor"

        def quality_for_cfi_tli(val: float) -> str:
            if val > 0.95:
                return "good"
            if val > 0.90:
                return "warning"
            return "poor"

        def quality_for_srmsr(val: float) -> str:
            if val < 0.05:
                return "good"
            if val < 0.08:
                return "warning"
            return "poor"

        headers = ["Index", "Value", "Interpretation"]
        rows = [
            [
                "M2",
                format_value(fit_indices["M2"], ".2f"),
                f"df = {fit_indices['M2_df']:.0f}, p = {fit_indices['M2_p']:.4f}",
            ],
            [
                "RMSEA",
                format_value(
                    fit_indices["RMSEA"], ".4f", quality_for_rmsea(fit_indices["RMSEA"])
                ),
                f"90% CI: [{fit_indices['RMSEA_CI_lower']:.4f}, {fit_indices['RMSEA_CI_upper']:.4f}]",
            ],
            [
                "CFI",
                format_value(
                    fit_indices["CFI"], ".4f", quality_for_cfi_tli(fit_indices["CFI"])
                ),
                "Comparative Fit Index",
            ],
            [
                "TLI",
                format_value(
                    fit_indices["TLI"], ".4f", quality_for_cfi_tli(fit_indices["TLI"])
                ),
                "Tucker-Lewis Index (NNFI)",
            ],
            [
                "SRMSR",
                format_value(
                    fit_indices["SRMSR"],
                    ".4f",
                    quality_for_srmsr(fit_indices["SRMSR"]),
                ),
                "Standardized Root Mean Square Residual",
            ],
        ]
        return table_from_data(headers, rows)

    def _interpretation_guide(self) -> str:
        from mirt.reports._templates import summary_box

        return summary_box(
            """
        <h4>Fit Index Guidelines</h4>
        <ul>
            <li><strong>RMSEA:</strong> &lt; 0.05 good, &lt; 0.08 acceptable, &gt; 0.10 poor</li>
            <li><strong>CFI/TLI:</strong> &gt; 0.95 good, &gt; 0.90 acceptable</li>
            <li><strong>SRMSR:</strong> &lt; 0.05 good, &lt; 0.08 acceptable</li>
            <li><strong>M2 p-value:</strong> Non-significant (p &gt; 0.05) indicates adequate fit, but sensitive to sample size</li>
        </ul>
        <p><em>Note:</em> Multiple indices should be considered together. Model fit evaluation
        should also consider substantive interpretability and parameter reasonableness.</p>
        """
        )
