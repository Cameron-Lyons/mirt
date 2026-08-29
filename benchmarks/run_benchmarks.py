"""Reproducible timing, reporting, and regression checks for core workloads."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np

import mirt
from mirt.cat import CATEngine

SCHEMA_VERSION = 1
SUITE_ORDER = ("fit", "scoring", "cat")


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Repeated timing measurements for one named workload."""

    name: str
    times: tuple[float, ...]

    def __post_init__(self) -> None:
        resolved_name = self.name.strip()
        resolved_times = tuple(float(value) for value in self.times)
        if not resolved_name:
            raise ValueError("benchmark name must be non-empty")
        if not resolved_times:
            raise ValueError("benchmark times must contain at least one measurement")
        if any(not math.isfinite(value) or value < 0.0 for value in resolved_times):
            raise ValueError("benchmark times must be finite non-negative values")
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "times", resolved_times)

    @property
    def median(self) -> float:
        """Median elapsed seconds."""
        return statistics.median(self.times)

    @property
    def mean(self) -> float:
        """Mean elapsed seconds."""
        return statistics.fmean(self.times)

    @property
    def minimum(self) -> float:
        """Fastest elapsed seconds."""
        return min(self.times)

    @property
    def maximum(self) -> float:
        """Slowest elapsed seconds."""
        return max(self.times)

    @property
    def standard_deviation(self) -> float:
        """Population standard deviation in seconds."""
        return statistics.pstdev(self.times)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible measurement record."""
        return {
            "name": self.name,
            "times_seconds": list(self.times),
            "median_seconds": self.median,
            "mean_seconds": self.mean,
            "min_seconds": self.minimum,
            "max_seconds": self.maximum,
            "standard_deviation_seconds": self.standard_deviation,
            "repeats": len(self.times),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Comparison between current and baseline median timings."""

    name: str
    baseline_seconds: float
    current_seconds: float
    change_percent: float
    max_regression_percent: float
    status: str

    @property
    def regressed(self) -> bool:
        """Whether the configured regression limit was exceeded."""
        return self.status == "regressed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible comparison record."""
        return {
            "name": self.name,
            "baseline_median_seconds": self.baseline_seconds,
            "current_median_seconds": self.current_seconds,
            "change_percent": self.change_percent,
            "max_regression_percent": self.max_regression_percent,
            "status": self.status,
        }


def _time(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmups: int,
) -> tuple[float, ...]:
    """Run untimed warmups followed by repeated wall-clock measurements."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    for _ in range(warmups):
        fn()

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return tuple(times)


def bench_em_fit(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> BenchResult:
    """Benchmark a unidimensional 2PL EM fit."""
    responses = mirt.simdata(
        model="2PL",
        n_persons=n_persons,
        n_items=n_items,
        seed=42,
    )

    def run() -> None:
        mirt.fit_mirt(
            responses,
            model="2PL",
            n_quadpts=21,
            max_iter=80,
            tol=1e-3,
        )

    return BenchResult(
        "em_fit_2pl",
        _time(run, repeats=repeats, warmups=warmups),
    )


def bench_scoring(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> BenchResult:
    """Benchmark EAP scoring for one fitted 2PL model."""
    responses = mirt.simdata(
        model="2PL",
        n_persons=n_persons,
        n_items=n_items,
        seed=43,
    )
    fit = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=21,
        max_iter=80,
        tol=1e-3,
    )

    def run() -> None:
        mirt.fscores(fit, responses, method="EAP")

    return BenchResult(
        "eap_scoring",
        _time(run, repeats=repeats, warmups=warmups),
    )


def bench_cat(
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> BenchResult:
    """Benchmark a batch of 20 adaptive-test simulations."""
    responses = mirt.simdata(model="2PL", n_persons=400, n_items=n_items, seed=44)
    fit = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=21,
        max_iter=80,
        tol=1e-3,
    )
    engine = CATEngine(
        fit.model,
        item_selection="MFI",
        stopping_rule="SE",
        se_threshold=0.35,
        max_items=min(15, n_items),
    )
    thetas = np.linspace(-2, 2, 20)

    def run() -> None:
        for theta in thetas:
            engine.run_simulation(true_theta=float(theta))

    return BenchResult(
        "cat_batch_20",
        _time(run, repeats=repeats, warmups=warmups),
    )


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative command-line integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a finite non-negative command-line float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--warmups", type=_non_negative_int, default=1)
    parser.add_argument("--persons", type=_positive_int, default=500)
    parser.add_argument("--items", type=_positive_int, default=25)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("all", *SUITE_ORDER),
        help="Run one suite; repeat the option to select multiple suites",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "rust"),
        default="auto",
        help="Force the computational backend for the run",
    )
    parser.add_argument(
        "--json",
        "--output-json",
        dest="json_output",
        metavar="PATH",
        help="Write a structured report to PATH, or use '-' for standard output",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare medians with a structured report from an earlier run",
    )
    parser.add_argument(
        "--max-regression",
        type=_non_negative_float,
        default=20.0,
        metavar="PERCENT",
        help="Fail when a median exceeds its baseline by more than this percentage",
    )
    return parser


def resolve_suites(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve repeated suite options into stable execution order."""
    if not requested or "all" in requested:
        return SUITE_ORDER
    selected = set(requested)
    return tuple(name for name in SUITE_ORDER if name in selected)


def run_suites(
    suites: Sequence[str],
    *,
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int,
) -> list[BenchResult]:
    """Execute selected benchmark suites in their canonical order."""
    unknown = set(suites) - set(SUITE_ORDER)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown benchmark suites: {names}")
    if not suites:
        raise ValueError("at least one benchmark suite is required")
    results: list[BenchResult] = []
    if "fit" in suites:
        results.append(bench_em_fit(n_persons, n_items, repeats, warmups))
    if "scoring" in suites:
        results.append(bench_scoring(n_persons, n_items, repeats, warmups))
    if "cat" in suites:
        results.append(bench_cat(n_items, repeats, warmups))
    return results


def environment_metadata(backend_info: Mapping[str, Any]) -> dict[str, Any]:
    """Capture runtime metadata needed to interpret benchmark results."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "mirt_version": mirt.__version__,
        "requested_backend": backend_info["current_backend"],
        "effective_backend": backend_info["effective_backend"],
        "rust_available": bool(backend_info["rust_available"]),
    }


def build_report(
    results: Sequence[BenchResult],
    *,
    suites: Sequence[str],
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int,
    backend_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a versioned structured report from benchmark results."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(backend_info),
        "configuration": {
            "suites": list(suites),
            "persons": n_persons,
            "items": n_items,
            "repeats": repeats,
            "warmups": warmups,
        },
        "benchmarks": [result.to_dict() for result in results],
    }


def load_report(path: Path) -> dict[str, Any]:
    """Load and validate the stable fields of a structured benchmark report."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read baseline report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline report is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("baseline report must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"baseline report must use schema version {SCHEMA_VERSION}")
    if not isinstance(payload.get("environment"), dict):
        raise ValueError("baseline report is missing environment metadata")
    if not isinstance(payload.get("configuration"), dict):
        raise ValueError("baseline report is missing configuration metadata")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("baseline report must contain benchmark measurements")

    names: set[str] = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise ValueError("each baseline benchmark must be a JSON object")
        name = benchmark.get("name")
        median = benchmark.get("median_seconds")
        if not isinstance(name, str) or not name:
            raise ValueError("each baseline benchmark must have a non-empty name")
        if name in names:
            raise ValueError(f"baseline report contains duplicate benchmark {name!r}")
        names.add(name)
        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not math.isfinite(float(median))
            or float(median) <= 0.0
        ):
            raise ValueError(
                f"baseline benchmark {name!r} must have a positive finite median"
            )
    return payload


def _validate_baseline_compatibility(
    current_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> None:
    """Reject workload or backend mismatches that invalidate comparison."""
    current_config = current_report["configuration"]
    baseline_config = baseline_report["configuration"]
    current_environment = current_report["environment"]
    baseline_environment = baseline_report["environment"]
    current_names = {benchmark["name"] for benchmark in current_report["benchmarks"]}

    if baseline_config.get("items") != current_config.get("items"):
        raise ValueError("baseline item count does not match the current run")
    person_workloads = {"em_fit_2pl", "eap_scoring"}
    if current_names & person_workloads and baseline_config.get(
        "persons"
    ) != current_config.get("persons"):
        raise ValueError("baseline person count does not match the current run")
    if baseline_environment.get("effective_backend") != current_environment.get(
        "effective_backend"
    ):
        raise ValueError("baseline effective backend does not match the current run")


def compare_results(
    current_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    max_regression_percent: float,
) -> list[BenchmarkComparison]:
    """Compare current medians against a compatible structured baseline."""
    if (
        isinstance(max_regression_percent, bool)
        or not isinstance(max_regression_percent, (int, float))
        or not math.isfinite(float(max_regression_percent))
        or max_regression_percent < 0.0
    ):
        raise ValueError("max regression percentage must be finite and non-negative")
    _validate_baseline_compatibility(current_report, baseline_report)
    baseline_by_name = {
        benchmark["name"]: float(benchmark["median_seconds"])
        for benchmark in baseline_report["benchmarks"]
    }
    comparisons: list[BenchmarkComparison] = []
    for benchmark in current_report["benchmarks"]:
        name = benchmark["name"]
        if name not in baseline_by_name:
            raise ValueError(f"baseline report is missing benchmark {name!r}")
        baseline_seconds = baseline_by_name[name]
        current_seconds = float(benchmark["median_seconds"])
        change_percent = 100.0 * (current_seconds / baseline_seconds - 1.0)
        if change_percent > max_regression_percent:
            status = "regressed"
        elif change_percent < -max_regression_percent:
            status = "improved"
        else:
            status = "stable"
        comparisons.append(
            BenchmarkComparison(
                name=name,
                baseline_seconds=baseline_seconds,
                current_seconds=current_seconds,
                change_percent=change_percent,
                max_regression_percent=max_regression_percent,
                status=status,
            )
        )
    return comparisons


def write_report(report: Mapping[str, Any], destination: str) -> None:
    """Write a structured report to a file or standard output."""
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination == "-":
        sys.stdout.write(serialized)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def print_human_report(
    report: Mapping[str, Any],
    comparisons: Sequence[BenchmarkComparison],
    *,
    stream: TextIO,
) -> None:
    """Render concise human-readable benchmark and comparison tables."""
    environment = report["environment"]
    print(
        f"backend={environment['requested_backend']} "
        f"effective={environment['effective_backend']} "
        f"rust={environment['rust_available']}",
        file=stream,
    )
    for benchmark in report["benchmarks"]:
        print(
            f"{benchmark['name']:16s}  "
            f"median={benchmark['median_seconds']:.4f}s  "
            f"mean={benchmark['mean_seconds']:.4f}s  "
            f"min={benchmark['min_seconds']:.4f}s  "
            f"max={benchmark['max_seconds']:.4f}s  "
            f"n={benchmark['repeats']}",
            file=stream,
        )
    for comparison in comparisons:
        print(
            f"{comparison.name:16s}  baseline={comparison.baseline_seconds:.4f}s  "
            f"change={comparison.change_percent:+.1f}%  "
            f"status={comparison.status}",
            file=stream,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected suites and return a regression-sensitive process status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    suites = resolve_suites(args.suite)

    baseline: dict[str, Any] | None = None
    if args.baseline is not None:
        try:
            baseline = load_report(args.baseline)
        except ValueError as exc:
            parser.error(str(exc))

    if args.backend == "rust" and not mirt.is_rust_available():
        parser.error("Rust backend requested but extension is not available")
    mirt.set_backend(args.backend)
    backend_info = mirt.get_backend_info()
    results = run_suites(
        suites,
        n_persons=args.persons,
        n_items=args.items,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    report = build_report(
        results,
        suites=suites,
        n_persons=args.persons,
        n_items=args.items,
        repeats=args.repeats,
        warmups=args.warmups,
        backend_info=backend_info,
    )

    comparisons: list[BenchmarkComparison] = []
    if baseline is not None:
        try:
            comparisons = compare_results(
                report,
                baseline,
                max_regression_percent=args.max_regression,
            )
        except ValueError as exc:
            parser.error(str(exc))
        report["comparisons"] = [comparison.to_dict() for comparison in comparisons]

    human_stream = sys.stderr if args.json_output == "-" else sys.stdout
    print_human_report(report, comparisons, stream=human_stream)
    if args.json_output is not None:
        write_report(report, args.json_output)
    return 1 if any(comparison.regressed for comparison in comparisons) else 0


if __name__ == "__main__":
    raise SystemExit(main())
