"""Base class for report builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.results.fit_result import FitResult

# Mean-square bands: values outside MISFIT are flagged "Misfit", values outside
# CHECK but inside MISFIT are flagged "Check".
_MEAN_SQUARE_CHECK = (0.8, 1.2)
_MEAN_SQUARE_MISFIT = (0.7, 1.3)


class HTMLReport(ABC):
    """Abstract base class for standalone HTML reports."""

    default_title: str = "IRT Analysis Report"

    def __init__(self, title: str | None = None) -> None:
        self.title = title or self.default_title

    @abstractmethod
    def _build_content(self) -> str:
        """Build the HTML content sections."""
        raise NotImplementedError

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

    @staticmethod
    def _write_html(path: str | Path, html: str) -> Path:
        """Write an already-rendered report and return its absolute path."""
        output_path = Path(path)
        output_path.write_text(html, encoding="utf-8")
        return output_path.resolve()

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
        return self._write_html(path, self.generate())


class ReportBuilder(HTMLReport):
    """Abstract base class for fitted-model report builders.

    Subclasses implement specific report types by overriding
    the _build_content method.

    Parameters
    ----------
    fit_result : FitResult
        Fitted model result.
    title : str, optional
        Report title. Defaults to class-specific title.

    Attributes
    ----------
    fit_result : FitResult
        The fitted model result.
    title : str
        Report title.
    """

    def __init__(
        self,
        fit_result: FitResult,
        title: str | None = None,
    ) -> None:
        super().__init__(title)
        self.fit_result = fit_result

    def _response_matrix(self, responses: Any) -> NDArray[Any]:
        """Read responses as the model's items, with missing values coded -1.

        A DataFrame with named columns is matched to the model's item names
        first, as :func:`mirt.fscores` does, so a reordered frame gives the
        same report. Nullable columns (``pd.NA``) and ``NaN`` become the
        missing code ``-1`` that :func:`mirt.fit_mirt` and the diagnostics
        accept; other values are left for the diagnostics to validate.
        """
        import numpy as np

        from mirt.utils.data import _missing_coded_responses, align_response_columns

        model = getattr(self.fit_result, "model", None)
        aligned = align_response_columns(responses, getattr(model, "item_names", None))
        return np.asarray(_missing_coded_responses(aligned))

    def _limited_information_fit(
        self, responses: NDArray[Any]
    ) -> tuple[dict[str, float], str]:
        """Compute fit indices and name the statistic behind them.

        Binary items use M2 and ordinal items M2*. When M2* has no degrees
        of freedom and C2 has some, C2 is reported, as
        :func:`mirt.diagnostics.modelfit.compute_fit_indices` does by default,
        but without its fallback warning: the report names the statistic.
        """
        import numpy as np

        from mirt.diagnostics.modelfit import compute_fit_indices

        model = self.fit_result.model
        counts = (
            model.n_categories
            if getattr(model, "is_polytomous", False)
            else [2] * model.n_items
        )
        statistic = "M2*" if np.any(np.asarray(counts) > 2) else "M2"
        indices = compute_fit_indices(self.fit_result, responses, type=statistic)
        if statistic == "M2*" and indices["M2_df"] == 0:
            c2_indices = compute_fit_indices(self.fit_result, responses, type="C2")
            if c2_indices["M2_df"] > 0:
                return c2_indices, "C2"
        return indices, statistic

    @staticmethod
    def _statistic_note(statistic: str) -> str:
        """Explain a C2 statistic that replaced an untestable M2*."""
        if statistic != "C2":
            return ""
        return (
            "<p><em>Note:</em> M2* has no degrees of freedom for these items and "
            "parameters, so the hybrid C2 statistic (Cai &amp; Monroe, 2014) is "
            "reported, and RMSEA, CFI and TLI are based on it.</p>"
        )

    def _wright_map_section(
        self, theta: NDArray[np.float64], title: str, *, level: int = 2
    ) -> str:
        """Embed one Wright map per factor, or a note where none applies.

        Factors without located items, and families without item locations
        such as nominal response models, are reported in a note instead of
        failing the report.
        """
        import numpy as np

        from mirt.reports._plots import create_wright_map_base64
        from mirt.reports._templates import embedded_plot, escape_text, section

        model = self.fit_result.model
        n_factors = int(getattr(model, "n_factors", 1))
        theta_values = np.asarray(theta)
        factors = range(n_factors if theta_values.ndim == 2 else 1)
        plots: list[str] = []
        notes: list[str] = []
        for factor in factors:
            label = f" (factor {factor + 1})" if n_factors > 1 else ""
            try:
                image = create_wright_map_base64(model, theta, factor=factor)
            except ValueError as exc:
                notes.append(f"Wright map{label} not shown: {exc}.")
            else:
                plots.append(embedded_plot(image, f"Wright Map{label}"))
        content = "".join(plots) + "".join(
            f"<p><em>{escape_text(note)}</em></p>" for note in notes
        )
        return section(title, content, level=level)

    def _model_summary_section(self, label: str = "Model") -> str:
        """Render model dimensions and fit statistics as a summary section."""
        from mirt.reports._templates import (
            escape_text,
            format_value,
            section,
            summary_box,
        )

        model = self.fit_result.model
        stats = self.fit_result.fit_statistics()
        html = f"""
        <p><strong>{escape_text(label)}:</strong> {escape_text(model.model_name)}</p>
        <p><strong>Items:</strong> {model.n_items} | <strong>Factors:</strong> {model.n_factors}</p>
        <p><strong>Persons:</strong> {stats["n_observations"]} | <strong>Parameters:</strong> {stats["n_parameters"]}</p>
        <p><strong>Log-Likelihood:</strong> {format_value(stats["log_likelihood"], ".2f")}</p>
        <p><strong>AIC:</strong> {format_value(stats["aic"], ".2f")} | <strong>BIC:</strong> {format_value(stats["bic"], ".2f")}</p>
        <p><strong>Converged:</strong> {stats["converged"]} ({stats["n_iterations"]} iterations)</p>
        """
        return section("Model Summary", summary_box(html))

    def _parameter_table(self, *, include_tests: bool, alpha: float = 0.05) -> str:
        """Tabulate every parameter cell with normal-approximation inference.

        Matrix parameters get one row per cell, labelled as in
        :meth:`FitResult.summary`, and unknown standard errors render as NA.
        """
        import numpy as np

        from mirt.reports._templates import format_value, table_from_data

        statistics = self.fit_result.parameter_statistics(alpha)
        headers = ["Item", "Parameter", "Estimate", "SE"]
        if include_tests:
            headers += ["z", "p"]
        headers.append(f"{(1.0 - alpha) * 100:.0f}% CI")

        rows: list[list[str]] = []
        for name, values in statistics.items():
            estimates = values["estimate"]
            for index in np.ndindex(estimates.shape):
                lower = values["ci_lower"][index]
                upper = values["ci_upper"][index]
                row = [
                    self.fit_result._parameter_label(name, estimates.shape, index),
                    name,
                    format_value(estimates[index], ".4f"),
                    format_value(values["standard_error"][index], ".4f"),
                ]
                if include_tests:
                    row += [
                        format_value(values["z"][index], ".3f"),
                        format_value(values["p_value"][index], ".4f"),
                    ]
                row.append(
                    f"[{lower:.3f}, {upper:.3f}]"
                    if np.isfinite(lower) and np.isfinite(upper)
                    else "NA"
                )
                rows.append(row)
        return table_from_data(headers, rows)

    def _itemfit_table(
        self,
        fit_stats: dict[str, NDArray[np.float64]],
        *,
        include_guide: bool,
    ) -> str:
        """Tabulate infit/outfit mean squares with misfit flags."""
        import numpy as np

        from mirt.reports._templates import format_value, summary_box, table_from_data

        model = self.fit_result.model
        infit = np.asarray(fit_stats.get("infit", np.ones(model.n_items)), dtype=float)
        outfit = np.asarray(
            fit_stats.get("outfit", np.ones(model.n_items)), dtype=float
        )

        def outside(bounds: tuple[float, float]) -> NDArray[np.bool_]:
            low, high = bounds
            return (infit < low) | (infit > high) | (outfit < low) | (outfit > high)

        misfit = outside(_MEAN_SQUARE_MISFIT)
        check = outside(_MEAN_SQUARE_CHECK) & ~misfit

        rows: list[list[str]] = []
        for i in range(model.n_items):
            flag = ""
            quality: str | None = None
            if misfit[i]:
                flag, quality = "Misfit", "poor"
            elif check[i]:
                flag, quality = "Check", "warning"
            rows.append(
                [
                    model.item_names[i],
                    format_value(infit[i], ".3f", quality),
                    format_value(outfit[i], ".3f", quality),
                    flag,
                ]
            )

        table = table_from_data(["Item", "Infit", "Outfit", "Flag"], rows)
        if not include_guide:
            return table
        return table + summary_box(
            f"""
            <p><strong>Interpretation Guide:</strong></p>
            <ul>
                <li>Acceptable fit: 0.7 - 1.3 (lenient), 0.8 - 1.2 (strict)</li>
                <li>Items flagged for misfit: {int(misfit.sum())} / {model.n_items}</li>
            </ul>
            """
        )
