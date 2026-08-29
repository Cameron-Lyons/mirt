"""Tests for structured benchmark reporting and regression checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("mirt_benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def _backend_info(name: str = "numpy") -> dict[str, Any]:
    return {
        "current_backend": name,
        "effective_backend": name,
        "rust_available": name == "rust",
    }


def _report(
    *results: benchmark.BenchResult,
    persons: int = 100,
    items: int = 10,
    backend: str = "numpy",
) -> dict[str, Any]:
    return benchmark.build_report(
        results,
        suites=("fit", "scoring", "cat"),
        n_persons=persons,
        n_items=items,
        repeats=len(results[0].times),
        warmups=1,
        backend_info=_backend_info(backend),
    )


class TestBenchResult:
    def test_calculates_complete_summary(self) -> None:
        result = benchmark.BenchResult("fit", (1.0, 2.0, 3.0, 4.0))

        assert result.median == 2.5
        assert result.mean == 2.5
        assert result.minimum == 1.0
        assert result.maximum == 4.0
        assert result.standard_deviation == pytest.approx(1.11803398875)
        assert result.to_dict() == {
            "name": "fit",
            "times_seconds": [1.0, 2.0, 3.0, 4.0],
            "median_seconds": 2.5,
            "mean_seconds": 2.5,
            "min_seconds": 1.0,
            "max_seconds": 4.0,
            "standard_deviation_seconds": pytest.approx(1.11803398875),
            "repeats": 4,
        }

    @pytest.mark.parametrize(
        ("name", "times", "message"),
        [
            ("", (1.0,), "name"),
            ("fit", (), "at least one"),
            ("fit", (-1.0,), "non-negative"),
            ("fit", (float("nan"),), "finite"),
            ("fit", (float("inf"),), "finite"),
        ],
    )
    def test_rejects_invalid_measurements(
        self,
        name: str,
        times: tuple[float, ...],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            benchmark.BenchResult(name, times)

    def test_warmups_are_run_but_not_measured(self) -> None:
        calls: list[int] = []

        times = benchmark._time(
            lambda: calls.append(len(calls)),
            repeats=3,
            warmups=2,
        )

        assert calls == [0, 1, 2, 3, 4]
        assert len(times) == 3
        assert all(value >= 0.0 for value in times)

    @pytest.mark.parametrize(
        ("repeats", "warmups", "message"),
        [
            (0, 0, "repeats"),
            (True, 0, "repeats"),
            (1, -1, "warmups"),
            (1, True, "warmups"),
        ],
    )
    def test_timer_validates_direct_calls(
        self,
        repeats: object,
        warmups: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            benchmark._time(
                lambda: None,
                repeats=repeats,
                warmups=warmups,
            )


class TestBenchmarkReports:
    def test_report_contains_versioned_workload_and_environment_metadata(self) -> None:
        report = _report(benchmark.BenchResult("fit", (1.0, 1.2)))

        assert report["schema_version"] == benchmark.SCHEMA_VERSION
        assert report["generated_at"].endswith("+00:00")
        assert report["configuration"] == {
            "suites": ["fit", "scoring", "cat"],
            "persons": 100,
            "items": 10,
            "repeats": 2,
            "warmups": 1,
        }
        assert report["environment"]["effective_backend"] == "numpy"
        assert report["environment"]["python_version"]
        assert report["environment"]["numpy_version"]
        assert report["benchmarks"][0]["median_seconds"] == 1.1

    def test_report_round_trip_supports_nested_output_directories(
        self,
        tmp_path: Path,
    ) -> None:
        report = _report(benchmark.BenchResult("fit", (1.0,)))
        destination = tmp_path / "nested" / "report.json"

        benchmark.write_report(report, str(destination))
        loaded = benchmark.load_report(destination)

        assert loaded == json.loads(destination.read_text(encoding="utf-8"))
        assert loaded["benchmarks"][0]["name"] == "fit"

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ([], "JSON object"),
            ({"schema_version": 99}, "schema version"),
            (
                {
                    "schema_version": 1,
                    "environment": {},
                    "configuration": {},
                    "benchmarks": [],
                },
                "measurements",
            ),
            (
                {
                    "schema_version": 1,
                    "environment": {},
                    "configuration": {},
                    "benchmarks": [{"name": "fit", "median_seconds": 0.0}],
                },
                "positive finite median",
            ),
        ],
    )
    def test_rejects_malformed_baselines(
        self,
        tmp_path: Path,
        payload: object,
        message: str,
    ) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            benchmark.load_report(path)


class TestBenchmarkComparisons:
    def test_classifies_improvements_stability_and_regressions(self) -> None:
        baseline = _report(
            benchmark.BenchResult("fast", (1.0,)),
            benchmark.BenchResult("steady", (1.0,)),
            benchmark.BenchResult("slow", (1.0,)),
        )
        current = _report(
            benchmark.BenchResult("fast", (0.7,)),
            benchmark.BenchResult("steady", (1.1,)),
            benchmark.BenchResult("slow", (1.3,)),
        )

        comparisons = benchmark.compare_results(
            current,
            baseline,
            max_regression_percent=20.0,
        )

        assert [comparison.status for comparison in comparisons] == [
            "improved",
            "stable",
            "regressed",
        ]
        assert comparisons[0].change_percent == pytest.approx(-30.0)
        assert comparisons[2].change_percent == pytest.approx(30.0)
        assert [comparison.regressed for comparison in comparisons] == [
            False,
            False,
            True,
        ]

    @pytest.mark.parametrize(
        ("current_kwargs", "message"),
        [
            ({"persons": 101}, "person count"),
            ({"items": 11}, "item count"),
            ({"backend": "rust"}, "backend"),
        ],
    )
    def test_rejects_incompatible_workloads(
        self,
        current_kwargs: dict[str, Any],
        message: str,
    ) -> None:
        baseline = _report(benchmark.BenchResult("em_fit_2pl", (1.0,)))
        current = _report(
            benchmark.BenchResult("em_fit_2pl", (1.0,)),
            **current_kwargs,
        )

        with pytest.raises(ValueError, match=message):
            benchmark.compare_results(
                current,
                baseline,
                max_regression_percent=20.0,
            )

    def test_requires_every_current_measurement_in_baseline(self) -> None:
        baseline = _report(benchmark.BenchResult("fit", (1.0,)))
        current = _report(benchmark.BenchResult("scoring", (1.0,)))

        with pytest.raises(ValueError, match="missing benchmark"):
            benchmark.compare_results(
                current,
                baseline,
                max_regression_percent=20.0,
            )

    @pytest.mark.parametrize("value", [-1.0, float("nan"), True])
    def test_rejects_invalid_direct_regression_limits(self, value: object) -> None:
        report = _report(benchmark.BenchResult("fit", (1.0,)))

        with pytest.raises(ValueError, match="finite and non-negative"):
            benchmark.compare_results(
                report,
                report,
                max_regression_percent=value,
            )


class TestBenchmarkCommand:
    def test_suite_selection_is_deduplicated_and_canonical(self) -> None:
        assert benchmark.resolve_suites(None) == ("fit", "scoring", "cat")
        assert benchmark.resolve_suites(["cat", "fit", "cat"]) == ("fit", "cat")
        assert benchmark.resolve_suites(["scoring", "all"]) == (
            "fit",
            "scoring",
            "cat",
        )

    @pytest.mark.parametrize("suites", [(), ("unknown",)])
    def test_runner_rejects_invalid_direct_suite_selection(
        self,
        suites: tuple[str, ...],
    ) -> None:
        with pytest.raises(ValueError, match="suite"):
            benchmark.run_suites(
                suites,
                n_persons=10,
                n_items=5,
                repeats=1,
                warmups=0,
            )

    def test_json_stdout_remains_machine_readable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        received: dict[str, Any] = {}

        def fake_run(suites, **kwargs):
            received["suites"] = tuple(suites)
            received.update(kwargs)
            return [benchmark.BenchResult("eap_scoring", (0.25, 0.3))]

        monkeypatch.setattr(benchmark, "run_suites", fake_run)
        monkeypatch.setattr(benchmark.mirt, "set_backend", lambda value: None)
        monkeypatch.setattr(
            benchmark.mirt,
            "get_backend_info",
            lambda: _backend_info(),
        )

        exit_code = benchmark.main(
            [
                "--suite",
                "scoring",
                "--persons",
                "200",
                "--items",
                "12",
                "--repeats",
                "2",
                "--warmups",
                "3",
                "--json",
                "-",
            ]
        )

        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert exit_code == 0
        assert received == {
            "suites": ("scoring",),
            "n_persons": 200,
            "n_items": 12,
            "repeats": 2,
            "warmups": 3,
        }
        assert report["configuration"]["suites"] == ["scoring"]
        assert report["benchmarks"][0]["median_seconds"] == 0.275
        assert "eap_scoring" in captured.err

    def test_regression_writes_report_and_returns_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline = _report(
            benchmark.BenchResult("eap_scoring", (1.0,)),
            persons=200,
            items=12,
        )
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        benchmark.write_report(baseline, str(baseline_path))
        monkeypatch.setattr(
            benchmark,
            "run_suites",
            lambda *args, **kwargs: [benchmark.BenchResult("eap_scoring", (1.3,))],
        )
        monkeypatch.setattr(benchmark.mirt, "set_backend", lambda value: None)
        monkeypatch.setattr(
            benchmark.mirt,
            "get_backend_info",
            lambda: _backend_info(),
        )

        exit_code = benchmark.main(
            [
                "--suite",
                "scoring",
                "--persons",
                "200",
                "--items",
                "12",
                "--repeats",
                "1",
                "--baseline",
                str(baseline_path),
                "--max-regression",
                "20",
                "--json",
                str(current_path),
            ]
        )

        current = json.loads(current_path.read_text(encoding="utf-8"))
        assert exit_code == 1
        assert current["comparisons"] == [
            {
                "name": "eap_scoring",
                "baseline_median_seconds": 1.0,
                "current_median_seconds": 1.3,
                "change_percent": pytest.approx(30.0),
                "max_regression_percent": 20.0,
                "status": "regressed",
            }
        ]

    def test_invalid_baseline_fails_before_workloads_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline_path = tmp_path / "invalid.json"
        baseline_path.write_text('{"schema_version": 99}', encoding="utf-8")

        def unexpected_run(*args, **kwargs):
            raise AssertionError("workloads should not run for an invalid baseline")

        monkeypatch.setattr(benchmark, "run_suites", unexpected_run)

        with pytest.raises(SystemExit) as error:
            benchmark.main(["--baseline", str(baseline_path)])

        assert error.value.code == 2

    @pytest.mark.parametrize(
        "arguments",
        [
            ["--repeats", "0"],
            ["--warmups", "-1"],
            ["--persons", "0"],
            ["--items", "0"],
            ["--max-regression", "nan"],
        ],
    )
    def test_rejects_invalid_cli_values(self, arguments: list[str]) -> None:
        with pytest.raises(SystemExit) as error:
            benchmark.main(arguments)

        assert error.value.code == 2
