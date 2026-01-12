"""Base class for report builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


class ReportBuilder(ABC):
    """Abstract base class for IRT report builders.

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

    default_title: str = "IRT Analysis Report"

    def __init__(
        self,
        fit_result: FitResult,
        title: str | None = None,
    ) -> None:
        self.fit_result = fit_result
        self.title = title or self.default_title

    @abstractmethod
    def _build_content(self) -> str:
        """Build the HTML content sections.

        Returns
        -------
        str
            HTML content for the report body.
        """
        pass

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
