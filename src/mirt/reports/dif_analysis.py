"""DIF analysis report builder."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DIFAnalysisReport:
    """Generate HTML report for Differential Item Functioning (DIF) analysis.

    This report includes:
    - Analysis summary (method, groups, items analyzed/flagged)
    - DIF statistics table with ETS A/B/C classification
    - DIF effect size plot
    - ETS classification interpretation guide

    Unlike other report types, DIFAnalysisReport does not require a pre-fitted
    model - it takes DIF results directly.

    Parameters
    ----------
    dif_results : dict
        DIF analysis results from compute_dif, containing:
        - 'statistic': Test statistic for each item
        - 'p_value': P-value for each item
        - 'effect_size': Effect size measure
        - 'classification': ETS classification (A/B/C)
    data : ndarray
        Response matrix (n_persons x n_items).
    groups : ndarray
        Group membership array.
    method : str
        DIF detection method used ('likelihood_ratio', 'wald', 'lord', 'raju').
    item_names : list of str, optional
        Names for items. If None, uses Item_1, Item_2, etc.
    title : str, optional
        Report title.

    Examples
    --------
    >>> from mirt.diagnostics import compute_dif
    >>> from mirt.reports import DIFAnalysisReport
    >>> import numpy as np
    >>> groups = np.array([0] * 250 + [1] * 250)
    >>> dif_results = compute_dif(data, groups, model="2PL")
    >>> report = DIFAnalysisReport(dif_results, data, groups)
    >>> report.save("dif_analysis.html")
    """

    default_title = "Differential Item Functioning (DIF) Analysis Report"

    def __init__(
        self,
        dif_results: dict[str, NDArray[np.float64]],
        data: NDArray[np.int_],
        groups: NDArray,
        method: Literal[
            "likelihood_ratio", "wald", "lord", "raju"
        ] = "likelihood_ratio",
        item_names: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        self.dif_results = dif_results
        self.data = np.asarray(data)
        self.groups = np.asarray(groups)
        self.method = method
        self.item_names = item_names or [f"Item_{i + 1}" for i in range(data.shape[1])]
        self.title = title or self.default_title

    def _build_content(self) -> str:
        from mirt.reports._plots import create_dif_plot_base64
        from mirt.reports._templates import embedded_plot, section

        sections = []

        sections.append(self._build_analysis_summary())

        sections.append(section("DIF Statistics", self._build_dif_table()))

        dif_plot = create_dif_plot_base64(self.dif_results, item_names=self.item_names)
        sections.append(
            section("DIF Effect Sizes", embedded_plot(dif_plot, "DIF Plot"))
        )

        sections.append(
            section("ETS Classification Guide", self._classification_guide())
        )

        return "\n".join(sections)

    def _build_analysis_summary(self) -> str:
        from mirt.reports._templates import section, summary_box

        unique_groups = np.unique(self.groups)
        n_items = self.data.shape[1]
        classifications = self.dif_results.get("classification", ["A"] * n_items)
        n_flagged = sum(1 for c in classifications if c != "A")
        n_moderate = sum(1 for c in classifications if c == "B")
        n_large = sum(1 for c in classifications if c == "C")

        summary_html = f"""
        <p><strong>Method:</strong> {self.method.replace("_", " ").title()}</p>
        <p><strong>Reference Group:</strong> {unique_groups[0]} (n = {np.sum(self.groups == unique_groups[0])})</p>
        <p><strong>Focal Group:</strong> {unique_groups[1]} (n = {np.sum(self.groups == unique_groups[1])})</p>
        <p><strong>Items Analyzed:</strong> {n_items}</p>
        <p><strong>Items Flagged:</strong> {n_flagged} ({100 * n_flagged / n_items:.1f}%)
            - B (Moderate): {n_moderate}, C (Large): {n_large}</p>
        """
        return section("Analysis Summary", summary_box(summary_html))

    def _build_dif_table(self) -> str:
        from mirt.reports._templates import format_value, table_from_data

        headers = ["Item", "Statistic", "p-value", "Effect Size", "Class"]
        rows: list[list[str]] = []

        statistic = self.dif_results.get("statistic", np.zeros(len(self.item_names)))
        p_value = self.dif_results.get("p_value", np.ones(len(self.item_names)))
        effect_size = self.dif_results.get(
            "effect_size", np.zeros(len(self.item_names))
        )
        classification = self.dif_results.get(
            "classification", ["A"] * len(self.item_names)
        )

        for i in range(len(self.item_names)):
            stat = statistic[i]
            p = p_value[i]
            es = effect_size[i]
            cls = classification[i]

            quality = {"A": "good", "B": "warning", "C": "poor"}.get(str(cls), None)

            p_quality = "poor" if p < 0.05 else None

            rows.append(
                [
                    self.item_names[i],
                    format_value(stat, ".3f"),
                    format_value(p, ".4f", p_quality),
                    format_value(es, ".4f"),
                    f'<span class="stat-value {quality}">{cls}</span>',
                ]
            )

        return table_from_data(headers, rows)

    def _classification_guide(self) -> str:
        from mirt.reports._templates import summary_box

        return summary_box(
            """
        <h4>ETS DIF Classification System</h4>
        <ul>
            <li><strong class="good">A (Negligible):</strong> |effect size| &lt; 0.426 OR not statistically significant</li>
            <li><strong class="warning">B (Moderate):</strong> 0.426 &le; |effect size| &lt; 0.638 AND statistically significant</li>
            <li><strong class="poor">C (Large):</strong> |effect size| &ge; 0.638 AND statistically significant</li>
        </ul>
        <h4>Recommended Actions</h4>
        <ul>
            <li><strong>A items:</strong> No action needed</li>
            <li><strong>B items:</strong> Review for content bias; may retain if substantively justified</li>
            <li><strong>C items:</strong> Strong recommendation for removal or revision; requires expert review</li>
        </ul>
        <p><em>Note:</em> DIF detection is statistical evidence of differential functioning,
        not definitive proof of bias. Items should be reviewed by content experts to
        determine if the DIF reflects true bias or real group differences in the construct.</p>
        """
        )

    def generate(self) -> str:
        """Generate the complete HTML report.

        Returns
        -------
        str
            Complete HTML document.
        """
        from mirt.reports._templates import html_document

        content = self._build_content()
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return html_document(self.title, content, generated_at)

    def save(self, path: str | Path) -> Path:
        """Save report to file.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            Absolute path to saved file.
        """
        path = Path(path)
        html = self.generate()
        path.write_text(html, encoding="utf-8")
        return path.resolve()
