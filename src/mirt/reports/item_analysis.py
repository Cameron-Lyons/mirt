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
    include_plots : bool
        Embed visualizations in the report. Set to False for a lightweight
        report that does not require matplotlib. Default True.

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
        include_plots: bool = True,
    ) -> None:
        super().__init__(fit_result, title)
        self.responses = np.asarray(responses)
        self.theta = theta
        self.include_plots = include_plots

    def _build_content(self) -> str:
        from mirt.diagnostics.itemfit import compute_itemfit
        from mirt.reports._templates import section

        sections = []

        sections.append(self._model_summary_section())

        sections.append(
            section("Item Parameters", self._parameter_table(include_tests=True))
        )

        fit_stats = compute_itemfit(self.fit_result, self.responses)
        sections.append(
            section(
                "Item Fit Statistics",
                self._itemfit_table(fit_stats, include_guide=True),
            )
        )

        if self.include_plots:
            from mirt.reports._plots import (
                create_icc_plot_base64,
                create_information_plot_base64,
                create_itemfit_plot_base64,
                create_wright_map_base64,
            )
            from mirt.reports._templates import embedded_plot

            plot_base64 = create_itemfit_plot_base64(
                fit_stats, item_names=self.fit_result.model.item_names
            )
            sections.append(
                section(
                    "Item Fit Plot",
                    embedded_plot(plot_base64, "Item Fit Statistics"),
                )
            )

            icc_base64 = create_icc_plot_base64(self.fit_result.model)
            sections.append(
                section(
                    "Item Characteristic Curves",
                    embedded_plot(icc_base64, "ICC Plot"),
                )
            )

            info_base64 = create_information_plot_base64(self.fit_result.model)
            sections.append(
                section(
                    "Test Information Function",
                    embedded_plot(info_base64, "Information Function"),
                )
            )

            if self.theta is not None:
                wright_base64 = create_wright_map_base64(
                    self.fit_result.model, self.theta
                )
                sections.append(
                    section(
                        "Person-Item Map (Wright Map)",
                        embedded_plot(wright_base64, "Wright Map"),
                    )
                )

        return "\n".join(sections)
