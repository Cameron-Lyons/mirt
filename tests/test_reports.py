"""Tests for HTML report generation."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="matplotlib not installed"
)


@pytest.fixture
def fitted_model():
    """Create a fitted 2PL model for testing."""
    from mirt import fit_mirt, simdata

    np.random.seed(42)
    data = simdata(n_persons=200, n_items=10)
    result = fit_mirt(data, model="2PL", max_iter=50)
    return result, data


@pytest.fixture
def theta_estimates(fitted_model):
    """Get theta estimates for Wright map."""
    from mirt import fscores

    result, data = fitted_model
    scores = fscores(result, data, method="EAP")
    return scores.theta


class TestTemplates:
    """Tests for HTML template functions."""

    def test_html_document(self):
        from mirt.reports._templates import html_document

        html = html_document("Test Title", "<p>Content</p>", "2025-01-15")
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Title</title>" in html
        assert "<p>Content</p>" in html
        assert "2025-01-15" in html

    def test_section(self):
        from mirt.reports._templates import section

        html = section("Section Title", "<p>Body</p>", level=2)
        assert "<h2>Section Title</h2>" in html
        assert "<p>Body</p>" in html

        html_h3 = section("H3 Title", "content", level=3)
        assert "<h3>H3 Title</h3>" in html_h3

    def test_table_from_data(self):
        from mirt.reports._templates import table_from_data

        html = table_from_data(
            headers=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
        )
        assert "<table>" in html
        assert "<th>A</th>" in html
        assert "<td>1</td>" in html

    def test_embedded_plot(self):
        from mirt.reports._templates import embedded_plot

        html = embedded_plot("base64data", "Alt text")
        assert "data:image/png;base64,base64data" in html
        assert 'alt="Alt text"' in html

    def test_summary_box(self):
        from mirt.reports._templates import summary_box

        html = summary_box("<p>Summary</p>")
        assert 'class="summary-box"' in html
        assert "<p>Summary</p>" in html

    def test_format_value(self):
        from mirt.reports._templates import format_value

        assert "1.2346" in format_value(1.23456, ".4f")
        assert "NA" in format_value(np.nan, ".4f")
        assert 'class="stat-value good"' in format_value(1.0, ".2f", "good")


@requires_matplotlib
class TestPlotConversion:
    """Tests for matplotlib to base64 conversion."""

    def test_figure_to_base64(self):
        import matplotlib.pyplot as plt

        from mirt.reports._plots import figure_to_base64

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        b64 = figure_to_base64(fig)
        plt.close(fig)

        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_create_icc_plot_base64(self, fitted_model):
        from mirt.reports._plots import create_icc_plot_base64

        result, _ = fitted_model
        b64 = create_icc_plot_base64(result.model)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_create_information_plot_base64(self, fitted_model):
        from mirt.reports._plots import create_information_plot_base64

        result, _ = fitted_model
        b64 = create_information_plot_base64(result.model)
        assert isinstance(b64, str)
        assert len(b64) > 100


@requires_matplotlib
class TestItemAnalysisReport:
    """Tests for ItemAnalysisReport."""

    def test_generate(self, fitted_model, theta_estimates):
        from mirt.reports import ItemAnalysisReport

        result, data = fitted_model
        report = ItemAnalysisReport(result, data, theta=theta_estimates)
        html = report.generate()

        assert "<!DOCTYPE html>" in html
        assert "Item Analysis Report" in html
        assert "Model Summary" in html
        assert "Item Parameters" in html
        assert "Item Fit Statistics" in html

    def test_save(self, fitted_model):
        from mirt.reports import ItemAnalysisReport

        result, data = fitted_model
        report = ItemAnalysisReport(result, data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            saved_path = report.save(path)
            assert saved_path.exists()
            content = saved_path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_custom_title(self, fitted_model):
        from mirt.reports import ItemAnalysisReport

        result, data = fitted_model
        report = ItemAnalysisReport(result, data, title="Custom Title")
        html = report.generate()
        assert "Custom Title" in html


@requires_matplotlib
class TestModelFitReport:
    """Tests for ModelFitReport."""

    def test_generate(self, fitted_model):
        from mirt.reports import ModelFitReport

        result, data = fitted_model
        report = ModelFitReport(result, data)
        html = report.generate()

        assert "<!DOCTYPE html>" in html
        assert "Model Fit Report" in html
        assert "Fit Indices" in html
        assert "RMSEA" in html
        assert "CFI" in html


@requires_matplotlib
class TestFullDiagnosticReport:
    """Tests for FullDiagnosticReport."""

    def test_generate(self, fitted_model, theta_estimates):
        from mirt.reports import FullDiagnosticReport

        result, data = fitted_model
        report = FullDiagnosticReport(result, data, theta=theta_estimates)
        html = report.generate()

        assert "<!DOCTYPE html>" in html
        assert "Full IRT Diagnostic Report" in html
        assert "Model Summary" in html
        assert "Item Parameters" in html
        assert "Item Fit Statistics" in html
        assert "Model Fit Indices" in html

    def test_without_ld(self, fitted_model):
        from mirt.reports import FullDiagnosticReport

        result, data = fitted_model
        report = FullDiagnosticReport(result, data, include_ld=False)
        html = report.generate()
        assert "<!DOCTYPE html>" in html


@requires_matplotlib
class TestDIFAnalysisReport:
    """Tests for DIFAnalysisReport."""

    def test_generate(self):
        from mirt.diagnostics import compute_dif
        from mirt.reports import DIFAnalysisReport
        from mirt.utils.simulation import simdata

        np.random.seed(42)
        data = simdata(n_persons=400, n_items=10)
        groups = np.array([0] * 200 + [1] * 200)

        dif_results = compute_dif(data, groups, model="2PL", max_iter=50)
        report = DIFAnalysisReport(dif_results, data, groups)
        html = report.generate()

        assert "<!DOCTYPE html>" in html
        assert "DIF" in html
        assert "Analysis Summary" in html
        assert "DIF Statistics" in html
        assert "ETS Classification" in html


@requires_matplotlib
class TestGenerateReport:
    """Tests for the generate_report convenience function."""

    def test_item_analysis(self, fitted_model, theta_estimates):
        from mirt.reports import generate_report

        result, data = fitted_model
        html = generate_report(
            result, data, report_type="item_analysis", theta=theta_estimates
        )
        assert "Item Analysis Report" in html

    def test_model_fit(self, fitted_model):
        from mirt.reports import generate_report

        result, data = fitted_model
        html = generate_report(result, data, report_type="model_fit")
        assert "Model Fit Report" in html

    def test_full_diagnostic(self, fitted_model, theta_estimates):
        from mirt.reports import generate_report

        result, data = fitted_model
        html = generate_report(
            result, data, report_type="full_diagnostic", theta=theta_estimates
        )
        assert "Full IRT Diagnostic Report" in html

    def test_save_to_file(self, fitted_model):
        from mirt.reports import generate_report

        result, data = fitted_model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            generate_report(result, data, report_type="model_fit", output_path=path)
            assert path.exists()

    def test_invalid_report_type(self, fitted_model):
        from mirt.reports import generate_report

        result, data = fitted_model
        with pytest.raises(ValueError, match="Unknown report type"):
            generate_report(result, data, report_type="invalid")
