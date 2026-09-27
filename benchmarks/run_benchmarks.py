"""Reproducible timing, reporting, and regression checks for core workloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np

import mirt
from mirt.cat import CATEngine

SCHEMA_VERSION = 1
SUITE_ORDER = (
    "fit",
    "scoring",
    "posterior",
    "patterns",
    "data",
    "diagnostics",
    "fit-statistics",
    "cat",
    "kernels",
    "optimization",
    "information",
    "latent-density",
)


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Repeated timing measurements for one named workload."""

    name: str
    times: tuple[float, ...]
    peak_traced_bytes: int | None = None

    def __post_init__(self) -> None:
        resolved_name = self.name.strip()
        resolved_times = tuple(float(value) for value in self.times)
        if not resolved_name:
            raise ValueError("benchmark name must be non-empty")
        if not resolved_times:
            raise ValueError("benchmark times must contain at least one measurement")
        if any(not math.isfinite(value) or value < 0.0 for value in resolved_times):
            raise ValueError("benchmark times must be finite non-negative values")
        if self.peak_traced_bytes is not None and (
            isinstance(self.peak_traced_bytes, bool)
            or not isinstance(self.peak_traced_bytes, int)
            or self.peak_traced_bytes < 0
        ):
            raise ValueError("peak traced bytes must be a non-negative integer")
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
        payload = {
            "name": self.name,
            "times_seconds": list(self.times),
            "median_seconds": self.median,
            "mean_seconds": self.mean,
            "min_seconds": self.minimum,
            "max_seconds": self.maximum,
            "standard_deviation_seconds": self.standard_deviation,
            "repeats": len(self.times),
        }
        if self.peak_traced_bytes is not None:
            payload["peak_traced_bytes"] = self.peak_traced_bytes
        return payload


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


def _peak_traced_bytes(fn: Callable[[], object]) -> int:
    """Measure Python/NumPy allocations in a separate, untimed execution."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


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


def bench_kernels(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Isolate cached likelihood kernels on responses with missing values."""
    from mirt.backends.rust import compute_log_likelihoods_2pl

    rng = np.random.default_rng(61)
    points = np.linspace(-3, 3, 41)
    a, b = rng.uniform(0.5, 1.5, n_items), rng.normal(size=n_items)
    responses = rng.integers(-1, 2, (n_persons, n_items))
    results = [
        BenchResult(
            "likelihood_2pl",
            _time(
                lambda: compute_log_likelihoods_2pl(responses, points, a, b),
                repeats=repeats,
                warmups=warmups,
            ),
        )
    ]
    for factory in (mirt.GradedResponseModel, mirt.GeneralizedPartialCredit):
        model = factory(n_items, n_categories=5)
        data = rng.integers(-1, 5, (n_persons, n_items))
        results.append(
            BenchResult(
                f"likelihood_{model.model_name.lower()}",
                _time(
                    lambda: model.log_likelihood_batch(data, points[:, None]),
                    repeats=repeats,
                    warmups=warmups,
                ),
            )
        )
    return results


def bench_optimization(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Time complete polytomous fits, MAP/ML scoring, and repetitive EM data."""
    results = []
    for kind in ("GRM", "GPCM", "PCM"):
        data = mirt.simdata(
            model=kind, n_categories=5, n_persons=n_persons, n_items=n_items, seed=62
        )
        results.append(
            BenchResult(
                f"em_fit_{kind.lower()}",
                _time(
                    lambda: mirt.fit_mirt(
                        data,
                        model=kind,
                        n_categories=5,
                        n_quadpts=21,
                        max_iter=20,
                        tol=1e-12,
                    ),
                    repeats=repeats,
                    warmups=warmups,
                ),
            )
        )
    rng = np.random.default_rng(63)
    model = mirt.TwoParameterLogistic(n_items)
    model.set_parameters(
        discrimination=rng.uniform(0.5, 1.5, n_items),
        difficulty=rng.normal(size=n_items),
    )
    model._is_fitted = True
    data = rng.integers(-1, 2, (n_persons, n_items))
    for method in ("MAP", "ML"):
        results.append(
            BenchResult(
                f"{method.lower()}_scoring",
                _time(
                    lambda: mirt.fscores(model, data, method=method),
                    repeats=repeats,
                    warmups=warmups,
                ),
            )
        )
    pool = data[: min(32, n_persons)]
    repeated = pool[rng.integers(0, len(pool), n_persons)]
    results.append(
        BenchResult(
            "em_fit_repeated",
            _time(
                lambda: mirt.fit_mirt(
                    repeated, model="2PL", n_quadpts=21, max_iter=20, tol=1e-12
                ),
                repeats=repeats,
                warmups=warmups,
            ),
        )
    )
    return results


def bench_latent_density(
    n_persons: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure Gaussian updates/evaluation, using persons as the point count."""
    from mirt.estimation.latent_density import GaussianDensity

    rng = np.random.default_rng(68)
    results = []
    for n_dimensions in (1, 3, 8):
        points = rng.normal(size=(n_persons, n_dimensions))
        weights = rng.uniform(size=n_persons)
        density = GaussianDensity(
            n_dimensions=n_dimensions, estimate_mean=True, estimate_cov=True
        )

        def update() -> None:
            density.update(points, weights)

        times = _time(update, repeats=repeats, warmups=warmups)
        results.append(
            BenchResult(
                f"gaussian_update_{n_dimensions}d", times, _peak_traced_bytes(update)
            )
        )

        def evaluate() -> object:
            return density.log_density(points)

        times = _time(evaluate, repeats=repeats, warmups=warmups)
        results.append(
            BenchResult(
                f"gaussian_log_density_{n_dimensions}d",
                times,
                _peak_traced_bytes(evaluate),
            )
        )
    return results


def bench_information(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> BenchResult:
    """Measure marginal-information time and separately traced Python/NumPy peak storage."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.standard_errors import (
        _posterior_from_model,
        compute_observed_information,
    )

    model = mirt.TwoParameterLogistic(n_items)
    data = np.random.default_rng(64).integers(0, 2, (n_persons, n_items))
    quad = GaussHermiteQuadrature(n_points=15)
    posterior = _posterior_from_model(model, data, quad)

    def run():
        return compute_observed_information(
            model, data, posterior, quad, prior_mass=quad.weights
        )

    times = _time(run, repeats=repeats, warmups=warmups)
    peak = _peak_traced_bytes(run)
    return BenchResult("marginal_information", times, peak_traced_bytes=peak)


def bench_posterior(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Benchmark summaries of a two-factor posterior without fitting overhead."""
    rng = np.random.default_rng(45)
    model = mirt.TwoParameterLogistic(n_items=n_items, n_factors=2)
    model.set_parameters(
        discrimination=rng.uniform(0.5, 1.5, size=(n_items, 2)),
        difficulty=rng.normal(size=n_items),
    )
    model._is_fitted = True
    responses = rng.integers(0, 2, size=(n_persons, n_items))
    posterior = mirt.ability_posterior(model, responses, n_quadpts=21)

    def run() -> None:
        posterior.quantile([0.025, 0.5, 0.975])
        posterior.classification_probabilities()
        _ = posterior.entropy
        posterior.sample(5, seed=46)

    return [
        BenchResult(
            "posterior_summaries", _time(run, repeats=repeats, warmups=warmups)
        ),
        BenchResult(
            "posterior_highest_density",
            _time(
                posterior.highest_density_intervals, repeats=repeats, warmups=warmups
            ),
        ),
    ]


def bench_patterns(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Time public pattern collapsing with repeated and mostly distinct rows."""
    rng = np.random.default_rng(47)
    pool = rng.integers(-1, 5, size=(min(256, n_persons), n_items))
    repeated = pool[rng.integers(0, len(pool), size=n_persons)]
    distinct = rng.integers(-1, 5, size=(n_persons, n_items))
    return [
        BenchResult(
            name,
            _time(
                lambda: mirt.collapse_patterns(responses),
                repeats=repeats,
                warmups=warmups,
            ),
        )
        for name, responses in (
            ("patterns_repeated", repeated),
            ("patterns_distinct", distinct),
        )
    ]


def bench_data(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Time missing-data counts, mode imputation, and item summaries."""
    from mirt.utils.classical import itemstats
    from mirt.utils.imputation import impute_responses, pairwise_available

    rng = np.random.default_rng(48)
    responses = rng.integers(0, 5, size=(n_persons, n_items))
    responses[rng.random(responses.shape) < 0.1] = -1
    responses[0] = 0  # Keep every item imputable even in small benchmark runs.
    workloads = (
        ("pairwise_available", lambda: pairwise_available(responses)),
        ("mode_imputation", lambda: impute_responses(responses, method="mode")),
        ("item_statistics", lambda: itemstats(responses)),
    )
    return [
        BenchResult(name, _time(run, repeats=repeats, warmups=warmups))
        for name, run in workloads
    ]


def bench_diagnostics(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Measure NumPy pairwise diagnostics with complete and incomplete data."""
    from mirt.diagnostics.ld import _compute_ld_chi2_g2
    from mirt.utils.residuals import _compute_ld_matrix

    rng = np.random.default_rng(4781)
    results = []
    for name, missing_fraction in (("q3_complete", 0.0), ("q3_missing", 0.15)):
        residuals = rng.normal(size=(n_persons, n_items))
        residuals[rng.random(residuals.shape) < missing_fraction] = np.nan

        def run():
            return _compute_ld_matrix(residuals)

        times = _time(run, repeats=repeats, warmups=warmups)
        results.append(BenchResult(name, times, _peak_traced_bytes(run)))

    for name, missing_fraction in (("ld_complete", 0.0), ("ld_missing", 0.15)):
        probabilities = rng.uniform(0.05, 0.95, size=(n_persons, n_items))
        responses = (rng.random(probabilities.shape) < probabilities).astype(np.int32)
        responses[rng.random(responses.shape) < missing_fraction] = -1

        def run_ld():
            return _compute_ld_chi2_g2(
                None,
                responses,
                np.empty((n_persons, 1)),
                n_quadpts=21,
                positive_probabilities=probabilities,
            )

        times = _time(run_ld, repeats=repeats, warmups=warmups)
        results.append(BenchResult(name, times, _peak_traced_bytes(run_ld)))
    return results


def bench_fit_statistics(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Measure mean squares and item/person fit, including allocations."""
    from mirt.diagnostics.itemfit import compute_itemfit
    from mirt.diagnostics.personfit import compute_personfit
    from mirt.models.dichotomous import TwoParameterLogistic
    from mirt.models.polytomous import GradedResponseModel
    from mirt.utils.numeric import compute_fit_stats

    rng = np.random.default_rng(5187)
    results = []
    for label, missing_fraction in (("complete", 0.0), ("missing", 0.15)):
        expected = rng.uniform(0.05, 0.95, size=(n_persons, n_items))
        variance = expected * (1.0 - expected)
        responses = (rng.random(expected.shape) < expected).astype(np.int32)
        responses[rng.random(responses.shape) < missing_fraction] = -1
        for axis, dimension in ((0, "item"), (1, "person")):

            def run_mean_squares():
                return compute_fit_stats(responses, expected, variance, axis)

            times = _time(run_mean_squares, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"mean_squares_{dimension}_{label}",
                    times,
                    _peak_traced_bytes(run_mean_squares),
                )
            )

    theta = rng.normal(size=(n_persons, 1))
    for name, model, categories in (
        ("2pl", TwoParameterLogistic(n_items), 2),
        ("grm", GradedResponseModel(n_items, n_categories=5), 5),
    ):
        responses = rng.integers(0, categories, size=(n_persons, n_items))
        responses[rng.random(responses.shape) < 0.15] = -1

        def run_personfit():
            return compute_personfit(model, responses, theta, p_adjust="fdr_bh")

        times = _time(run_personfit, repeats=repeats, warmups=warmups)
        results.append(
            BenchResult(f"personfit_{name}", times, _peak_traced_bytes(run_personfit))
        )

        def run_itemfit():
            return compute_itemfit(
                model,
                responses,
                theta=theta,
                statistics=["infit", "outfit", "S_X2"],
                p_adjust="fdr_bh",
            )

        times = _time(run_itemfit, repeats=repeats, warmups=warmups)
        results.append(
            BenchResult(f"itemfit_{name}", times, _peak_traced_bytes(run_itemfit))
        )
    return results


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
    if "posterior" in suites:
        results.extend(bench_posterior(n_persons, n_items, repeats, warmups))
    if "patterns" in suites:
        results.extend(bench_patterns(n_persons, n_items, repeats, warmups))
    if "data" in suites:
        results.extend(bench_data(n_persons, n_items, repeats, warmups))
    if "diagnostics" in suites:
        results.extend(bench_diagnostics(n_persons, n_items, repeats, warmups))
    if "fit-statistics" in suites:
        results.extend(bench_fit_statistics(n_persons, n_items, repeats, warmups))
    if "cat" in suites:
        results.append(bench_cat(n_items, repeats, warmups))
    if "kernels" in suites:
        results.extend(bench_kernels(n_persons, n_items, repeats, warmups))
    if "optimization" in suites:
        results.extend(bench_optimization(n_persons, n_items, repeats, warmups))
    if "information" in suites:
        results.append(bench_information(n_persons, n_items, repeats, warmups))
    if "latent-density" in suites:
        results.extend(bench_latent_density(n_persons, repeats, warmups))
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
        "thread_settings": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "RAYON_NUM_THREADS",
            )
        },
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
    person_workloads = {
        "em_fit_2pl",
        "eap_scoring",
        "posterior_summaries",
        "posterior_highest_density",
        "patterns_repeated",
        "patterns_distinct",
        "pairwise_available",
        "mode_imputation",
        "item_statistics",
        "q3_complete",
        "q3_missing",
        "ld_complete",
        "ld_missing",
        "likelihood_2pl",
        "likelihood_grm",
        "likelihood_gpcm",
        "em_fit_grm",
        "em_fit_gpcm",
        "em_fit_pcm",
        "em_fit_repeated",
        "map_scoring",
        "ml_scoring",
        "marginal_information",
        "gaussian_update_1d",
        "gaussian_update_3d",
        "gaussian_update_8d",
        "gaussian_log_density_1d",
        "gaussian_log_density_3d",
        "gaussian_log_density_8d",
    }
    if current_names & person_workloads and baseline_config.get(
        "persons"
    ) != current_config.get("persons"):
        raise ValueError("baseline person count does not match the current run")
    if baseline_environment.get("effective_backend") != current_environment.get(
        "effective_backend"
    ):
        raise ValueError("baseline effective backend does not match the current run")
    if "thread_settings" in baseline_environment and baseline_environment[
        "thread_settings"
    ] != current_environment.get("thread_settings"):
        raise ValueError("baseline thread settings do not match the current run")


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
        peak = benchmark.get("peak_traced_bytes")
        memory = "" if peak is None else f"  traced_peak={peak / 1024**2:.2f}MiB"
        print(
            f"{benchmark['name']:16s}  "
            f"median={benchmark['median_seconds']:.4f}s  "
            f"mean={benchmark['mean_seconds']:.4f}s  "
            f"min={benchmark['min_seconds']:.4f}s  "
            f"max={benchmark['max_seconds']:.4f}s  "
            f"n={benchmark['repeats']}{memory}",
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
