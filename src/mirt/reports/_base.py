"""Base class for report builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


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
