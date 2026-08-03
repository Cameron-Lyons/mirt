"""Contract tests for safe and lightweight HTML reports."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtValidationError


@pytest.fixture(scope="module")
def fitted_report_data() -> tuple[Any, np.ndarray]:
    from mirt import fit_mirt, simdata

    responses = simdata(n_persons=120, n_items=8, seed=418)
    result = fit_mirt(responses, model="2PL", max_iter=40)
    return result, responses


def test_document_and_section_escape_external_text() -> None:
    from mirt.reports._templates import html_document, section

    document = html_document(
        '<script>alert("title")</script>',
        "<p>trusted report markup</p>",
        "2026-01-01<script>",
    )

    assert "<script>" not in document
    assert "&lt;script&gt;alert(&quot;" in document
    assert "<p>trusted report markup</p>" in document
    assert "2026-01-01&lt;script&gt;" in document

    heading = section("Fit <unsafe>", "<p>content</p>")
    assert "<h2>Fit &lt;unsafe&gt;</h2>" in heading
    assert "<p>content</p>" in heading


@pytest.mark.parametrize("level", [True, 1, 5, 2.5, "2"])
def test_section_rejects_invalid_heading_levels(level: Any) -> None:
    from mirt.reports._templates import section

    with pytest.raises(ValueError, match="level must be"):
        section("Heading", "content", level=level)


def test_tables_escape_labels_but_preserve_generated_markup() -> None:
    from mirt.reports._templates import format_label, format_value, table_from_data

    table = table_from_data(
        ["Item <name>", "Value"],
        [["<img src=x onerror=alert(1)>", format_value(1.25, ".2f")]],
        caption='Summary "table"',
    )

    assert "<img src=x" not in table
    assert "&lt;img src=x onerror=alert(1)&gt;" in table
    assert "Item &lt;name&gt;" in table
    assert "Summary &quot;" in table
    assert '<span class="stat-value">1.25</span>' in table

    label = format_label("<C>", "poor")
    assert "&lt;C&gt;" in label
    assert "<C>" not in label


def test_table_rejects_ragged_rows() -> None:
    from mirt.reports._templates import table_from_data

    with pytest.raises(ValueError, match="same length"):
        table_from_data(["A", "B"], [["only one"]])


def test_plot_attributes_are_escaped() -> None:
    from mirt.reports._templates import embedded_plot

    markup = embedded_plot('abc" onerror="alert(1)', '<plot "label">')

    assert 'onerror="alert(1)"' not in markup
    assert "abc&quot; onerror=&quot;alert(1)" in markup
    assert "&lt;plot &quot;" in markup


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_render_as_missing(value: float) -> None:
    from mirt.reports._templates import format_value

    assert ">NA<" in format_value(value)


def test_quality_classes_are_allowlisted() -> None:
    from mirt.reports._templates import format_label, format_value

    with pytest.raises(ValueError, match="quality must be"):
        format_value(1.0, quality='poor" onclick="alert(1)')
    with pytest.raises(ValueError, match="quality must be"):
        format_label("A", quality="unknown")

    assert "&amp;" in format_value(1.0, "&<6")


def test_plotting_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mirt.reports import _plots

    original_import = builtins.__import__

    def missing_matplotlib(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    _plots._get_pyplot.cache_clear()
    monkeypatch.setattr(builtins, "__import__", missing_matplotlib)
    with pytest.raises(ImportError, match="pip install matplotlib"):
        _plots._get_pyplot()
    _plots._get_pyplot.cache_clear()


@pytest.mark.parametrize("dpi", [0, -1, True, 100.5])
def test_figure_conversion_validates_dpi(dpi: Any) -> None:
    from mirt.reports._plots import figure_to_base64

    with pytest.raises(ValueError, match="positive integer"):
        figure_to_base64(object(), dpi=dpi)


def test_lightweight_reports_do_not_import_matplotlib(
    fitted_report_data: tuple[Any, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mirt.reports import FullDiagnosticReport, ItemAnalysisReport, ModelFitReport

    result, responses = fitted_report_data
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError("lightweight rendering imported matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    reports = [
        ItemAnalysisReport(result, responses, include_plots=False),
        ModelFitReport(result, responses, include_plots=False),
        FullDiagnosticReport(
            result,
            responses,
            include_ld=False,
            include_plots=False,
        ),
    ]

    for report in reports:
        html = report.generate()
        assert "<!DOCTYPE html>" in html
        assert "data:image/" not in html


def test_report_title_and_item_names_are_escaped(
    fitted_report_data: tuple[Any, np.ndarray],
) -> None:
    from mirt.reports import ItemAnalysisReport

    result, responses = fitted_report_data
    original_names = result.model.item_names
    result.model.item_names = ["<script>bad()</script>", *original_names[1:]]
    try:
        html = ItemAnalysisReport(
            result,
            responses,
            title="Analysis <unsafe>",
            include_plots=False,
        ).generate()
    finally:
        result.model.item_names = original_names

    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "Analysis &lt;unsafe&gt;" in html


def test_generate_report_renders_once_when_saving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mirt.reports import ModelFitReport, generate_report

    calls = 0

    def render_once(self: ModelFitReport) -> str:
        nonlocal calls
        calls += 1
        return "<html>single render</html>"

    monkeypatch.setattr(ModelFitReport, "generate", render_once)
    output_path = tmp_path / "report.html"
    html = generate_report(
        object(),
        np.zeros((1, 1), dtype=int),
        report_type="model_fit",
        output_path=output_path,
    )

    assert calls == 1
    assert html == "<html>single render</html>"
    assert output_path.read_text(encoding="utf-8") == html


def _dif_inputs() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    results = {
        "statistic": np.array([1.0, 2.0]),
        "p_value": np.array([0.4, 0.01]),
        "effect_size": np.array([0.1, 0.7]),
        "classification": np.array(["A", "C"]),
    }
    data = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    groups = np.array(["<reference>", "<reference>", "focal&", "focal&"])
    return results, data, groups


def test_lightweight_dif_report_escapes_group_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mirt.reports import DIFAnalysisReport

    results, data, groups = _dif_inputs()
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError("lightweight rendering imported matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    html = DIFAnalysisReport(
        results,
        data,
        groups,
        include_plots=False,
    ).generate()

    assert "data:image/" not in html
    assert "&lt;reference&gt;" in html
    assert "focal&amp;" in html


def test_dif_report_validates_shapes_and_classifications() -> None:
    from mirt.reports import DIFAnalysisReport

    results, data, groups = _dif_inputs()
    with pytest.raises(MirtDataError, match="exactly two groups"):
        DIFAnalysisReport(results, data, np.zeros(data.shape[0]))
    with pytest.raises(MirtDataError, match="one value per response row"):
        DIFAnalysisReport(results, data, groups[:-1])
    with pytest.raises(MirtValidationError, match="one name per response column"):
        DIFAnalysisReport(results, data, groups, item_names=["one"])

    invalid_results = {**results, "classification": np.array(["A", "D"])}
    with pytest.raises(MirtValidationError, match="must be A, B, or C"):
        DIFAnalysisReport(invalid_results, data, groups)


def test_plot_figure_is_closed_after_rendering_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirt.plotting
    from mirt.reports import _plots

    figure = object()
    axis = object()

    class FakePyplot:
        def __init__(self) -> None:
            self.closed: list[object] = []

        def subplots(self, *, figsize: tuple[float, float]) -> tuple[object, object]:
            assert figsize == (10, 6)
            return figure, axis

        def close(self, value: object) -> None:
            self.closed.append(value)

    pyplot = FakePyplot()

    def fail_plot(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(_plots, "_get_pyplot", lambda: pyplot)
    monkeypatch.setattr(mirt.plotting, "plot_icc", fail_plot)

    with pytest.raises(RuntimeError, match="plot failed"):
        _plots.create_icc_plot_base64(object())
    assert pyplot.closed == [figure]


def test_report_exports_do_not_depend_on_plotting_extra() -> None:
    import mirt

    assert "generate_report" in mirt.__all__
    assert "ItemAnalysisReport" in mirt.__all__
