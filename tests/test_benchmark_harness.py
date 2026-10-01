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
    def test_optional_traced_memory_is_reported(self) -> None:
        result = benchmark.BenchResult("information", (0.1,), peak_traced_bytes=123)
        assert result.to_dict()["peak_traced_bytes"] == 123
        with pytest.raises(ValueError, match="peak traced"):
            benchmark.BenchResult("information", (0.1,), peak_traced_bytes=-1)

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

    @pytest.mark.parametrize(
        "name",
        [
            "patterns_repeated",
            "patterns_distinct",
            "posterior_highest_density",
            "q3_complete",
            "q3_missing",
            "ld_complete",
            "ld_missing",
        ],
    )
    def test_new_workloads_reject_mismatched_person_counts(self, name: str) -> None:
        result = benchmark.BenchResult(name, (1.0,))
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(result, persons=100),
                _report(result, persons=200),
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
    @pytest.mark.parametrize("suite", ["kernels", "optimization", "information"])
    def test_new_suites_run_and_enforce_person_count(self, suite: str) -> None:
        results = benchmark.run_suites(
            (suite,), n_persons=12, n_items=2, repeats=1, warmups=0
        )
        assert results
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_suite_selection_is_deduplicated_and_canonical(self) -> None:
        assert benchmark.resolve_suites(None) == (
            "fit",
            "scoring",
            "posterior",
            "bayesian",
            "patterns",
            "data",
            "diagnostics",
            "misfit",
            "fit-statistics",
            "model-fit",
            "cat",
            "kernels",
            "gpu-likelihood",
            "optimization",
            "information",
            "latent-density",
            "kernel-smoothing",
            "empirical",
            "classical",
            "reliability",
            "curves",
            "asymmetric",
            "logistic-information",
            "unipolar",
            "logistic-probability",
            "multidimensional-information",
            "multidimensional-probability",
            "multidimensional-fit",
            "logistic-fit",
            "weighted-em",
            "weighted-mstep",
            "polytomous-fit",
            "item-curvature",
            "bl-fit",
            "irtree-fit",
            "mcem-fit",
            "variational",
            "gvem-uncertainty",
            "variational-objective",
            "variational-mstep",
            "regularized",
            "qmcem-mstep",
            "qmcem-fit",
            "mcem-sampling",
        )
        assert benchmark.resolve_suites(["cat", "fit", "cat"]) == ("fit", "cat")
        assert benchmark.resolve_suites(["scoring", "all"]) == (
            "fit",
            "scoring",
            "posterior",
            "bayesian",
            "patterns",
            "data",
            "diagnostics",
            "misfit",
            "fit-statistics",
            "model-fit",
            "cat",
            "kernels",
            "gpu-likelihood",
            "optimization",
            "information",
            "latent-density",
            "kernel-smoothing",
            "empirical",
            "classical",
            "reliability",
            "curves",
            "asymmetric",
            "logistic-information",
            "unipolar",
            "logistic-probability",
            "multidimensional-information",
            "multidimensional-probability",
            "multidimensional-fit",
            "logistic-fit",
            "weighted-em",
            "weighted-mstep",
            "polytomous-fit",
            "item-curvature",
            "bl-fit",
            "irtree-fit",
            "mcem-fit",
            "variational",
            "gvem-uncertainty",
            "variational-objective",
            "variational-mstep",
            "regularized",
            "qmcem-mstep",
            "qmcem-fit",
            "mcem-sampling",
        )

    def test_tensor_suite_requires_optional_runtime(self, monkeypatch):
        from mirt import _gpu_backend

        monkeypatch.setattr(_gpu_backend, "is_torch_available", lambda: False)
        with pytest.raises(ValueError, match="optional PyTorch"):
            benchmark.run_suites(
                ("gpu-likelihood",), n_persons=8, n_items=2, repeats=1, warmups=0
            )

    def test_tensor_suite_records_device_and_checks_workload_compatibility(self):
        from mirt import _gpu_backend

        if not _gpu_backend.is_torch_available():
            pytest.skip("PyTorch not installed")
        torch, device = _gpu_backend._load_torch_runtime()
        results = benchmark.run_suites(
            ("gpu-likelihood",), n_persons=12, n_items=4, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"gpu_likelihood_{device.type}_{label}"
            for label in (
                "1pl",
                "2pl",
                "complete_e_step",
                "3pl",
                "mirt_3d",
                "grm",
                "grm_mixed",
                "gpcm",
                "gpcm_mixed",
                "pcm_mixed",
            )
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes is None for result in results)
        report = benchmark.build_report(
            results,
            suites=("gpu-likelihood",),
            n_persons=12,
            n_items=4,
            repeats=2,
            warmups=1,
            backend_info=_backend_info(),
        )
        assert report["environment"]["torch_version"] == str(torch.__version__)
        assert report["environment"]["tensor_device"] == str(device)
        other = json.loads(json.dumps(report))
        other["configuration"]["persons"] = 13
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(report, other, max_regression_percent=5)
        other["configuration"]["persons"] = 12
        other["environment"]["torch_version"] = "different version"
        with pytest.raises(ValueError, match="tensor runtime"):
            benchmark.compare_results(report, other, max_regression_percent=5)

    def test_weighted_em_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["weighted-em"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert {result.name for result in results} == {
            "weighted_e_step_2pl",
            "weighted_e_step_mirt",
            "weighted_e_step_grm",
            "weighted_fit_2pl",
            "weighted_fit_grm",
        }
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_weighted_mstep_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["weighted-mstep"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"weighted_{kind}_{model}"
            for model in ("2pl", "2pl_3d", "grm", "nrm")
            for kind in ("mstep", "standard_errors")
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_polytomous_fit_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["polytomous-fit"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"polytomous_{kind}_{model}"
            for model in (
                "grm_1d",
                "grm_2d",
                "gpcm_1d",
                "gpcm_2d",
                "pcm_1d",
                "nrm_1d",
                "nrm_2d",
            )
            for kind in ("mstep", "fit", "weighted_fit")
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_item_curvature_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["item-curvature"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"item_curvature_{model}_{method}_{jobs}workers"
            for model in ("2pl_1d", "2pl_3d", "gpcm_1d", "grm_2d", "nrm_2d")
            for method, jobs in (
                ("central", 1),
                ("forward", 1),
                ("richardson", 1),
                ("central", 2),
            )
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_bl_fit_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["bl-fit"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"bl_{model}_{stage}"
            for model in (
                "2pl_1d",
                "3pl_1d",
                "2pl_2d",
                "grm_1d",
                "gpcm_1d",
                "nrm_2d",
                "mirt_2d",
                "bifactor_3d",
            )
            for stage in ("optimize", "fit")
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_irtree_fit_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["irtree-fit"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"irtree_{spec}_{stage}"
            for spec in ("bockenholt", "extreme_midpoint", "direction_intensity")
            for stage in ("e_step", "counts", "uncertainty", "fit")
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_mcem_fit_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["mcem-fit"], n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"mcem_{model}_{stage}"
            for model in ("2pl_3d", "3pl", "grm_2d", "gpcm_2d", "nrm_2d", "mirt_3d")
            for stage in ("refresh", "e_step", "mstep", "uncertainty", "fit")
        ]
        for result in results:
            assert len(result.times) == 1
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=8, items=2),
                    _report(result, persons=9, items=2),
                    max_regression_percent=5.0,
                )

    def test_mcem_sampling_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ("mcem-sampling",), n_persons=8, n_items=2, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"mcem_{model}_{method}_{stage}"
            for model in ("2pl_3d", "grm_2d", "2pl_3d_correlated", "mirt_6d_correlated")
            for method in ("posterior", "stochastic")
            for stage in ("prior", "e_step", "uncertainty", "fit")
        ]
        assert all(len(result.times) == 1 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=8, items=2),
                _report(*results, persons=9, items=2),
                max_regression_percent=20,
            )

    def test_variational_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["variational"], n_persons=20, n_items=4, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"variational_{kind}_{dimensions}d"
            for dimensions in (1, 3, 6)
            for kind in ("gvem", "sparse")
        ]
        for result in results:
            assert len(result.times) == 2
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=5.0,
                )

    def test_gvem_uncertainty_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ["gvem-uncertainty"], n_persons=12, n_items=4, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"gvem_{kind}_{dimensions}d"
            for dimensions in (1, 3)
            for kind in ("standard_errors", "fit")
        ]
        for result in results:
            assert len(result.times) == 2
            assert result.peak_traced_bytes > 0
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=12),
                    _report(result, persons=13),
                    max_regression_percent=5.0,
                )

    def test_variational_objective_suite_records_time_memory_and_checks_person_count(
        self,
    ) -> None:
        results = benchmark.run_suites(
            ("variational-objective",), n_persons=12, n_items=4, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"variational_objective_{name}_{n_factors}d"
            for n_factors in (1, 3, 6)
            for name in ("gvem", "sparse")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_variational_mstep_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ("variational-mstep",), n_persons=12, n_items=4, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"variational_mstep_{name}_{label}"
            for label in ("2pl_1d", "2pl_3d", "2pl_6d", "1pl")
            for name in ("gvem", "sparse")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_qmcem_mstep_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ("qmcem-mstep",), n_persons=12, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"qmcem_mstep_{label}"
            for label in ("2pl", "2pl_3d", "mirt_3d", "grm", "gpcm_2d", "nrm_2d")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_qmcem_fit_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ("qmcem-fit",), n_persons=12, n_items=3, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"qmcem_{stage}_{label}"
            for label in ("2pl", "2pl_3d", "mirt_3d", "grm", "gpcm_2d", "nrm_2d")
            for stage in ("refresh", "e_step", "uncertainty", "fit")
        ]
        assert all(len(result.times) == 1 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_regularized_suite_records_time_memory_and_checks_person_count(self):
        results = benchmark.run_suites(
            ("regularized",), n_persons=12, n_items=3, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            f"regularized_{kind}_{factors}d_{missing}"
            for factors in (2, 3)
            for missing in ("complete", "missing")
            for kind in ("e_step", "fit")
        ]
        assert all(len(result.times) == 1 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=12),
                _report(*results, persons=13),
                max_regression_percent=20,
            )

    def test_latent_density_suite_records_time_memory_and_checks_point_count(
        self,
    ) -> None:
        results = benchmark.run_suites(
            ("latent-density",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"gaussian_{method}_{n_dimensions}d"
            for n_dimensions in (1, 3, 8)
            for method in ("update", "log_density")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_kernel_smoothing_suite_records_time_memory_and_checks_person_count(
        self,
    ) -> None:
        results = benchmark.run_suites(
            ("kernel-smoothing",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"kernel_smoothing_{n_points}_{label}"
            for n_points in (81, 401)
            for label in ("complete", "missing")
        ] + [
            f"kernel_smoothing_gam_{kind}_{label}"
            for kind in ("2pl", "grm")
            for label in ("complete", "missing")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_empirical_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("empirical",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"empirical_rmsea_{kind}_{n_bins}_{label}"
            for kind in ("2pl", "grm")
            for n_bins in (10, 100)
            for label in ("complete", "missing")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_classical_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("classical",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"traditional_{correlation}_{label}"
            for correlation in ("corrected", "uncorrected")
            for label in ("complete", "missing")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_reliability_suite_records_time_memory_and_checks_person_count(
        self,
    ) -> None:
        results = benchmark.run_suites(
            ("reliability",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"reliability_{method}_{kind}"
            for kind in ("2pl", "grm")
            for method in ("sem", "empirical")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_curves_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("curves",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"curves_{method}_{kind}"
            for kind in ("2pl", "grm")
            for method in (
                "test_information",
                "item_information",
                "selected_information",
                "expected_score",
            )
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_asymmetric_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("asymmetric",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"asymmetric_{label}{method}"
            for label in ("", "cll_", "nll_")
            for method in ("probability", "information", "pairs")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=21),
                    max_regression_percent=20.0,
                )

    def test_logistic_information_suite_measures_full_and_single_queries(self) -> None:
        results = benchmark.run_suites(
            ("logistic-information",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"logistic_information_{model}_{selection}"
            for model in ("2pl", "3pl", "4pl", "2pl_multi")
            for selection in ("full", "single")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_unipolar_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("unipolar",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"unipolar_{method}" for method in ("probability", "information", "pairs")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_logistic_probability_suite_measures_full_single_and_pairs(self) -> None:
        results = benchmark.run_suites(
            ("logistic-probability",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"logistic_probability_{model}_{selection}"
            for model in ("1pl", "2pl", "3pl", "4pl", "2pl_multi")
            for selection in ("full", "single", "pairs")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_multidimensional_information_suite_measures_scalars_and_matrices(
        self,
    ) -> None:
        results = benchmark.run_suites(
            ("multidimensional-information",),
            n_persons=20,
            n_items=3,
            repeats=2,
            warmups=1,
        )
        assert [result.name for result in results] == [
            f"multidimensional_information_{model}_{selection}"
            for model in ("mirt", "bifactor")
            for selection in (
                "full",
                "single",
                "item_matrix",
                "test_matrix",
                "selection",
            )
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_multidimensional_probability_suite_measures_all_query_forms(self) -> None:
        results = benchmark.run_suites(
            ("multidimensional-probability",),
            n_persons=20,
            n_items=3,
            repeats=2,
            warmups=1,
        )
        assert [result.name for result in results] == [
            f"multidimensional_probability_{model}_{selection}"
            for model in ("mirt", "bifactor")
            for selection in ("full", "single", "pairs", "one_person")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_multidimensional_fit_suite_measures_affine_em(self) -> None:
        results = benchmark.run_suites(
            ("multidimensional-fit",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"multidimensional_fit_{model}"
            for model in ("mirt", "bifactor", "confirmatory")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_logistic_fit_suite_measures_python_item_objectives(self) -> None:
        results = benchmark.run_suites(
            ("logistic-fit",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"logistic_fit_{model}"
            for model in ("1pl", "2pl", "2pl_multi", "3pl", "4pl")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=20),
                _report(*results, persons=21),
                max_regression_percent=20.0,
            )

    def test_data_suite_runs_and_checks_workload_size(self) -> None:
        results = benchmark.run_suites(
            ("data",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "pairwise_available",
            "mode_imputation",
            "item_statistics",
        ]
        assert all(len(result.times) == 2 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=40),
                    max_regression_percent=20.0,
                )

    def test_bayesian_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("bayesian",), n_persons=6, n_items=2, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "waic",
            "psis_normal",
            "psis_heavy_tail",
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=6),
                    _report(result, persons=12),
                    max_regression_percent=20.0,
                )

    def test_diagnostics_suite_records_time_and_memory(self) -> None:
        results = benchmark.run_suites(
            ("diagnostics",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "q3_complete",
            "q3_missing",
            "ld_complete",
            "ld_missing",
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)

    def test_patterns_suite_runs(self) -> None:
        results = benchmark.run_suites(
            ("patterns",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "patterns_repeated",
            "patterns_distinct",
        ]
        assert all(len(result.times) == 2 for result in results)

    def test_misfit_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("misfit",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "misfit_2pl_complete",
            "misfit_2pl_missing",
            "misfit_grm_complete",
            "misfit_grm_missing",
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=40),
                    max_regression_percent=20.0,
                )

    def test_fit_statistics_suite_records_time_and_memory(self) -> None:
        results = benchmark.run_suites(
            ("fit-statistics",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            "mean_squares_item_complete",
            "mean_squares_person_complete",
            "mean_squares_item_missing",
            "mean_squares_person_missing",
            "personfit_2pl",
            "itemfit_2pl",
            "personfit_grm",
            "itemfit_grm",
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)

    def test_posterior_suite_runs_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("posterior",), n_persons=10, n_items=3, repeats=1, warmups=0
        )
        assert [result.name for result in results] == [
            "posterior_summaries",
            "posterior_highest_density",
        ]
        assert results[0].median > 0.0
        with pytest.raises(ValueError, match="person count"):
            benchmark.compare_results(
                _report(*results, persons=10),
                _report(*results, persons=20),
                max_regression_percent=20.0,
            )

    def test_model_fit_suite_records_time_memory_and_checks_person_count(self) -> None:
        results = benchmark.run_suites(
            ("model-fit",), n_persons=20, n_items=3, repeats=2, warmups=1
        )
        assert [result.name for result in results] == [
            f"model_fit_{model}_{missing}_{integration}"
            for model in ("2pl", "grm")
            for missing in ("complete", "missing")
            for integration in ("empirical", "quadrature")
        ]
        assert all(len(result.times) == 2 for result in results)
        assert all(result.peak_traced_bytes > 0 for result in results)
        for result in results:
            with pytest.raises(ValueError, match="person count"):
                benchmark.compare_results(
                    _report(result, persons=20),
                    _report(result, persons=40),
                    max_regression_percent=20.0,
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
