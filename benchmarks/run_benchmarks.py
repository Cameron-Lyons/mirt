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
    "bayesian",
    "patterns",
    "data",
    "diagnostics",
    "misfit",
    "fit-statistics",
    "model-fit",
    "cat",
    "kernels",
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


def bench_weighted_em(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure weighted E-steps, uncertainty, and five-iteration complete fits."""
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.weighted import WeightedEMEstimator

    rng = np.random.default_rng(42)
    weights = rng.uniform(0.25, 2.0, n_persons)
    results = []
    for name, model in (
        ("2pl", mirt.TwoParameterLogistic(n_items)),
        ("mirt", mirt.TwoParameterLogistic(n_items, n_factors=2)),
        ("grm", mirt.GradedResponseModel(n_items, n_categories=5)),
    ):
        responses = mirt.simdata(
            model=model.model_name,
            n_items=n_items,
            n_categories=5,
            theta=rng.normal(size=(n_persons, model.n_factors)),
            seed=42,
        )
        responses[rng.random(responses.shape) < 0.1] = -1
        estimator = WeightedEMEstimator(n_quadpts=7 if name == "mirt" else 21)
        estimator._quadrature = GaussHermiteQuadrature(
            n_points=estimator.n_quadpts, n_dimensions=model.n_factors
        )
        mean, cov = np.zeros(model.n_factors), np.eye(model.n_factors)

        def e_step():
            return estimator._e_step_weighted(model, responses, mean, cov, weights)

        results.append(
            BenchResult(
                f"weighted_e_step_{name}",
                _time(e_step, repeats=repeats, warmups=warmups),
                peak_traced_bytes=_peak_traced_bytes(e_step),
            )
        )
        if name == "mirt":
            continue

        def fit():
            return WeightedEMEstimator(n_quadpts=21, max_iter=5, tol=1e-12).fit(
                model.copy(), responses, weights=weights
            )

        results.append(
            BenchResult(
                f"weighted_fit_{name}",
                _time(fit, repeats=repeats, warmups=warmups),
                peak_traced_bytes=_peak_traced_bytes(fit),
            )
        )
    return results


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


def bench_kernel_smoothing(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure weighted calibration and shared empirical smoothing kernels."""
    from mirt.models.nonparametric import KernelSmoothingModel
    from mirt.models.polytomous import GradedResponseModel
    from mirt.utils.empirical import itemGAM

    rng = np.random.default_rng(69)
    theta = rng.normal(size=n_persons)
    responses = rng.integers(0, 2, size=(n_persons, n_items))
    weights = rng.uniform(0.1, 2.0, size=n_persons)
    missing_responses = responses.copy()
    missing_responses[rng.random(responses.shape) < 0.1] = -1
    missing_responses[0] = responses[0]
    results = []
    for n_points in (81, 401):
        model = KernelSmoothingModel(
            n_items=n_items, theta_grid=np.linspace(-4.0, 4.0, n_points)
        )
        for label, data in (("complete", responses), ("missing", missing_responses)):

            def run() -> None:
                model.calibrate(data, theta, sample_weight=weights)

            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"kernel_smoothing_{n_points}_{label}",
                    times,
                    _peak_traced_bytes(run),
                )
            )
    for kind, model, categories in (
        ("2pl", mirt.TwoParameterLogistic(n_items), 2),
        ("grm", GradedResponseModel(n_items, 4), 4),
    ):
        data = rng.integers(0, categories, size=(n_persons, n_items))
        missing_data = data.copy()
        missing_data[rng.random(data.shape) < 0.1] = -1
        missing_data[0] = data[0]
        for label, item_responses in (("complete", data), ("missing", missing_data)):

            def smooth() -> None:
                itemGAM(model, item_responses, theta, n_grid=101, bandwidth=0.5)

            times = _time(smooth, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"kernel_smoothing_gam_{kind}_{label}",
                    times,
                    _peak_traced_bytes(smooth),
                )
            )
    return results


def bench_empirical(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure binned empirical fit with complete and missing item scores."""
    from mirt.models.polytomous import GradedResponseModel
    from mirt.utils.empirical import empirical_rmsea

    rng = np.random.default_rng(70)
    theta = rng.normal(size=n_persons)
    results = []
    for kind, model, categories in (
        ("2pl", mirt.TwoParameterLogistic(n_items), 2),
        ("grm", GradedResponseModel(n_items, 4), 4),
    ):
        responses = rng.integers(0, categories, size=(n_persons, n_items))
        missing = responses.copy()
        missing[rng.random(responses.shape) < 0.1] = -1
        for n_bins in (10, 100):
            for label, data in (("complete", responses), ("missing", missing)):

                def run() -> None:
                    empirical_rmsea(model, data, theta, n_bins=n_bins)

                times = _time(run, repeats=repeats, warmups=warmups)
                results.append(
                    BenchResult(
                        f"empirical_rmsea_{kind}_{n_bins}_{label}",
                        times,
                        _peak_traced_bytes(run),
                    )
                )
    return results


def bench_classical(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure shared classical moments and alpha-if-deleted calculations."""
    from mirt.utils.classical import traditional

    if n_persons < 2 or n_items < 2:
        raise ValueError("classical benchmarks require at least two persons and items")
    rng = np.random.default_rng(71)
    responses = rng.integers(0, 2, size=(n_persons, n_items))
    missing = responses.copy()
    missing[rng.random(responses.shape) < 0.1] = -1
    missing[:2] = responses[:2]
    results = []
    for corrected, correlation in ((True, "corrected"), (False, "uncorrected")):
        for label, data in (("complete", responses), ("missing", missing)):

            def run() -> None:
                traditional(data, use_corrected_correlation=corrected)

            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"traditional_{correlation}_{label}",
                    times,
                    _peak_traced_bytes(run),
                )
            )
    return results


def bench_reliability(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure information-based reliability and measurement-error summaries."""
    from mirt.utils.reliability import empirical_rxx, sem

    if n_persons < 2:
        raise ValueError("reliability benchmarks require at least two persons")
    rng = np.random.default_rng(72)
    theta = rng.normal(size=n_persons)
    models = (
        ("2pl", mirt.TwoParameterLogistic(n_items)),
        ("grm", mirt.GradedResponseModel(n_items, n_categories=4)),
    )
    results = []
    for label, model in models:
        for name, statistic in (("sem", sem), ("empirical", empirical_rxx)):

            def run() -> None:
                statistic(model, theta)

            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"reliability_{name}_{label}", times, _peak_traced_bytes(run)
                )
            )
    return results


def bench_curves(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure dense and selected information and expected-score curves."""
    from mirt.utils.information import expected_score, iteminfo, testinfo

    theta = np.random.default_rng(73).normal(size=n_persons)
    selected = [n_items - 1, 0, n_items - 1]
    results = []
    for label, model in (
        ("2pl", mirt.TwoParameterLogistic(n_items)),
        ("grm", mirt.GradedResponseModel(n_items, n_categories=4)),
    ):
        for name, run in (
            ("test_information", lambda: testinfo(model, theta)),
            ("item_information", lambda: iteminfo(model, theta)),
            ("selected_information", lambda: iteminfo(model, theta, selected)),
            ("expected_score", lambda: expected_score(model, theta)),
        ):
            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(f"curves_{name}_{label}", times, _peak_traced_bytes(run))
            )
    return results


def _bench_response_curves(
    model: Any,
    theta: np.ndarray,
    indices: np.ndarray,
    prefix: str,
    repeats: int,
    warmups: int,
) -> list[BenchResult]:
    results = []
    for name, run in (
        ("probability", lambda: model.probability(theta)),
        ("information", lambda: model.information(theta)),
        ("pairs", lambda: model.probability_pairs(theta, indices)),
    ):
        times = _time(run, repeats=repeats, warmups=warmups)
        results.append(BenchResult(f"{prefix}{name}", times, _peak_traced_bytes(run)))
    return results


def bench_unipolar(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure the symmetric unipolar curve and its information function."""
    from mirt.models.dichotomous import UnipolarLogLogistic

    rng = np.random.default_rng(77)
    model = UnipolarLogLogistic(n_items).set_parameters(
        discrimination=rng.uniform(0.5, 2.0, n_items),
        difficulty=rng.normal(size=n_items),
    )
    theta = rng.normal(size=(n_persons, 1))
    indices = rng.integers(n_items, size=n_persons)
    return _bench_response_curves(model, theta, indices, "unipolar_", repeats, warmups)


def bench_asymmetric(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure asymmetric probabilities and information with realistic parameters."""
    from mirt.models.dichotomous import (
        ComplementaryLogLog,
        FiveParameterLogistic,
        NegativeLogLog,
    )

    rng = np.random.default_rng(75)
    model = FiveParameterLogistic(n_items).set_parameters(
        discrimination=rng.uniform(0.5, 2.0, n_items),
        difficulty=rng.normal(size=n_items),
        guessing=np.full(n_items, 0.2),
        upper=np.full(n_items, 0.9),
        asymmetry=rng.uniform(0.5, 2.0, n_items),
    )
    theta = rng.normal(size=(n_persons, 1))
    indices = rng.integers(n_items, size=n_persons)
    parameters = {
        "discrimination": model.discrimination,
        "difficulty": model.difficulty,
    }
    models = (
        ("", model),
        ("cll_", ComplementaryLogLog(n_items).set_parameters(**parameters)),
        ("nll_", NegativeLogLog(n_items).set_parameters(**parameters)),
    )
    results = []
    for label, model in models:
        results.extend(
            _bench_response_curves(
                model, theta, indices, f"asymmetric_{label}", repeats, warmups
            )
        )
    return results


def bench_logistic_information(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure full and selected logistic information curves, including MIRT."""
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    rng = np.random.default_rng(76)
    results = []
    for label, model in (
        ("2pl", TwoParameterLogistic(n_items)),
        ("3pl", ThreeParameterLogistic(n_items)),
        ("4pl", FourParameterLogistic(n_items)),
        ("2pl_multi", TwoParameterLogistic(n_items, n_factors=3)),
    ):
        parameters = model.parameters
        parameters["discrimination"] = rng.uniform(
            0.5, 2.0, parameters["discrimination"].shape
        )
        parameters["difficulty"] = rng.normal(size=n_items)
        if "upper" in parameters:
            parameters["upper"] = np.full(n_items, 0.9)
        model.set_parameters(**parameters)
        theta = rng.normal(size=(n_persons, model.n_factors))
        for selection, item_idx in (("full", None), ("single", n_items - 1)):

            def run() -> None:
                model.information(theta, item_idx)

            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"logistic_information_{label}_{selection}",
                    times,
                    _peak_traced_bytes(run),
                )
            )
    return results


def bench_logistic_probability(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure full, single-item, and paired logistic probabilities."""
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    rng = np.random.default_rng(78)
    results = []
    for label, model in (
        ("1pl", OneParameterLogistic(n_items)),
        ("2pl", TwoParameterLogistic(n_items)),
        ("3pl", ThreeParameterLogistic(n_items)),
        ("4pl", FourParameterLogistic(n_items)),
        ("2pl_multi", TwoParameterLogistic(n_items, n_factors=3)),
    ):
        parameters = model.parameters
        if label == "1pl":
            parameters.pop("discrimination")
        else:
            parameters["discrimination"] = rng.uniform(
                0.5, 2.0, parameters["discrimination"].shape
            )
        parameters["difficulty"] = rng.normal(size=n_items)
        if "upper" in parameters:
            parameters["upper"] = np.full(n_items, 0.9)
        model.set_parameters(**parameters)
        theta = rng.normal(size=(n_persons, model.n_factors))
        indices = rng.integers(n_items, size=n_persons)
        for selection, run in (
            ("full", lambda: model.probability(theta)),
            ("single", lambda: model.probability(theta, n_items - 1)),
            ("pairs", lambda: model.probability_pairs(theta, indices)),
        ):
            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"logistic_probability_{label}_{selection}",
                    times,
                    _peak_traced_bytes(run),
                )
            )
    return results


def bench_logistic_fit(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure Python 1PL–4PL EM objectives, including multidimensional 2PL."""
    from mirt.estimation.em import EMEstimator
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )

    rng = np.random.default_rng(82)
    models = (
        ("1pl", OneParameterLogistic(n_items)),
        ("2pl", TwoParameterLogistic(n_items)),
        ("2pl_multi", TwoParameterLogistic(n_items, n_factors=2)),
        ("3pl", ThreeParameterLogistic(n_items)),
        ("4pl", FourParameterLogistic(n_items)),
    )
    results = []
    for label, template in models:
        theta = rng.normal(size=(n_persons, template.n_factors))
        responses = (
            rng.random((n_persons, n_items)) < template.probability(theta)
        ).astype(int)
        responses[rng.random(responses.shape) < 0.1] = -1

        def run() -> None:
            EMEstimator(
                n_quadpts=15 if template.n_factors == 1 else 7,
                max_iter=8,
                tol=1e-9,
                use_rust=False,
                use_gpu=False,
                compute_standard_errors=False,
            ).fit(template.copy(), responses)

        results.append(
            BenchResult(
                f"logistic_fit_{label}",
                _time(run, repeats=repeats, warmups=warmups),
                _peak_traced_bytes(run),
            )
        )
    return results


def bench_multidimensional_fit(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure NumPy EM fitting with affine item gradients and missing responses."""
    from mirt.estimation.em import EMEstimator
    from mirt.models.bifactor import BifactorModel
    from mirt.models.multidimensional import MultidimensionalModel

    rng = np.random.default_rng(81)
    pattern = np.ones((n_items, 3))
    pattern[: n_items // 2, 2] = 0.0
    pattern[n_items // 2 :, 1] = 0.0
    models = (
        ("mirt", MultidimensionalModel(n_items, 2)),
        ("bifactor", BifactorModel(n_items, np.arange(n_items) % 2)),
        (
            "confirmatory",
            MultidimensionalModel(
                n_items,
                3,
                model_type="confirmatory",
                loading_pattern=pattern,
            ),
        ),
    )
    results = []
    for label, template in models:
        theta = rng.normal(size=(n_persons, template.n_factors))
        responses = (
            rng.random((n_persons, n_items)) < template.probability(theta)
        ).astype(int)
        responses[rng.random(responses.shape) < 0.1] = -1

        def run() -> None:
            EMEstimator(
                n_quadpts=7,
                max_iter=8,
                tol=1e-9,
                use_rust=False,
                use_gpu=False,
                compute_standard_errors=False,
            ).fit(template.copy(), responses)

        results.append(
            BenchResult(
                f"multidimensional_fit_{label}",
                _time(run, repeats=repeats, warmups=warmups),
                _peak_traced_bytes(run),
            )
        )
    return results


def bench_multidimensional_probability(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure dense and bifactor full, single-item, and aligned probabilities."""
    from mirt.models.bifactor import BifactorModel
    from mirt.models.multidimensional import MultidimensionalModel

    rng = np.random.default_rng(80)
    models = (
        (
            "mirt",
            MultidimensionalModel(n_items, 3).set_parameters(
                slopes=rng.normal(size=(n_items, 3)),
                intercepts=rng.normal(size=n_items),
            ),
        ),
        (
            "bifactor",
            BifactorModel(n_items, np.arange(n_items) % 4).set_parameters(
                general_loadings=rng.uniform(0.5, 2.0, n_items),
                specific_loadings=rng.normal(size=n_items),
                intercepts=rng.normal(size=n_items),
            ),
        ),
    )
    results = []
    for label, model in models:
        theta = rng.normal(size=(n_persons, model.n_factors))
        indices = rng.integers(n_items, size=n_persons)
        for selection, run in (
            ("full", lambda: model.probability(theta)),
            ("single", lambda: model.probability(theta, n_items - 1)),
            ("pairs", lambda: model.probability_pairs(theta, indices)),
            ("one_person", lambda: model.probability(theta[:1])),
        ):
            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"multidimensional_probability_{label}_{selection}",
                    times,
                    _peak_traced_bytes(run),
                )
            )
    return results


def bench_multidimensional_information(
    n_persons: int, n_items: int, repeats: int, warmups: int = 0
) -> list[BenchResult]:
    """Measure Fisher information and adaptive selection using item matrices."""
    from mirt.cat.mcat_selection import DOptimality
    from mirt.models.bifactor import BifactorModel
    from mirt.models.multidimensional import MultidimensionalModel

    rng = np.random.default_rng(79)
    models = (
        (
            "mirt",
            MultidimensionalModel(n_items, n_factors=3).set_parameters(
                slopes=rng.normal(size=(n_items, 3)),
                intercepts=rng.normal(size=n_items),
            ),
        ),
        (
            "bifactor",
            BifactorModel(n_items, np.arange(n_items) % 4).set_parameters(
                general_loadings=rng.uniform(0.5, 2.0, n_items),
                specific_loadings=rng.normal(size=n_items),
                intercepts=rng.normal(size=n_items),
            ),
        ),
    )
    results = []
    for label, model in models:
        theta = rng.normal(size=(n_persons, model.n_factors))
        covariance = np.eye(model.n_factors)
        available_items = set(range(n_items))
        strategy = DOptimality()
        for selection, run in (
            ("full", lambda: model.information(theta)),
            ("single", lambda: model.information(theta, n_items - 1)),
            ("item_matrix", lambda: model.item_information_matrix(theta, n_items - 1)),
            ("test_matrix", lambda: model.test_information_matrix(theta)),
            (
                "selection",
                lambda: strategy.get_item_criteria(
                    model, theta[0], covariance, available_items
                ),
            ),
        ):
            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(
                    f"multidimensional_information_{label}_{selection}",
                    times,
                    _peak_traced_bytes(run),
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


def bench_bayesian(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Measure information criteria on precomputed posterior likelihoods."""
    from mirt.diagnostics.bayesian import psis_loo, waic

    rng = np.random.default_rng(7832)
    log_lik = rng.normal(-2.0, 0.7, size=(1000, n_persons * n_items))

    def run_waic():
        return waic(log_lik)

    times = _time(run_waic, repeats=repeats, warmups=warmups)
    results = [BenchResult("waic", times, _peak_traced_bytes(run_waic))]
    for name, heavy_tail in (("psis_normal", False), ("psis_heavy_tail", True)):
        log_lik = rng.normal(-2.0, 0.7, size=(4000, n_persons))
        if heavy_tail:
            log_lik -= rng.pareto(1.8, size=log_lik.shape)
        relative_eff = np.linspace(0.2, 1.0, n_persons)

        def run_psis():
            return psis_loo(log_lik, relative_eff=relative_eff)

        times = _time(run_psis, repeats=repeats, warmups=warmups)
        results.append(BenchResult(name, times, _peak_traced_bytes(run_psis)))
    return results


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


def bench_misfit(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Measure misfit identification, including probability and result storage."""
    from mirt.diagnostics.residuals import identify_misfitting_patterns
    from mirt.models import GradedResponseModel, TwoParameterLogistic

    rng = np.random.default_rng(8392)
    theta = rng.normal(size=(n_persons, 1))
    results = []
    for name, model in (
        ("2pl", TwoParameterLogistic(n_items)),
        ("grm", GradedResponseModel(n_items, n_categories=5)),
    ):
        probabilities = model.probability(theta)
        draws = rng.random((n_persons, n_items))
        if probabilities.ndim == 2:
            responses = (draws < probabilities).astype(np.int_)
        else:
            responses = np.sum(
                draws[:, :, None] > np.cumsum(probabilities, axis=2)[:, :, :-1],
                axis=2,
            )
        del probabilities, draws
        for label, missing_fraction in (("complete", 0.0), ("missing", 0.15)):
            data = responses.copy()
            data[rng.random(data.shape) < missing_fraction] = -1

            def run():
                return identify_misfitting_patterns(model, data, theta)

            times = _time(run, repeats=repeats, warmups=warmups)
            results.append(
                BenchResult(f"misfit_{name}_{label}", times, _peak_traced_bytes(run))
            )
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


def bench_model_fit(
    n_persons: int,
    n_items: int,
    repeats: int,
    warmups: int = 0,
) -> list[BenchResult]:
    """Measure model-fit moments with empirical and quadrature integration."""
    from mirt.diagnostics.modelfit import compute_fit_indices
    from mirt.models import GradedResponseModel, TwoParameterLogistic

    rng = np.random.default_rng(7241)
    theta = rng.normal(size=(n_persons, 1))
    results = []
    for name, model, categories in (
        ("2pl", TwoParameterLogistic(n_items), 2),
        ("grm", GradedResponseModel(n_items, n_categories=5), 5),
    ):
        for label, missing_fraction in (("complete", 0.0), ("missing", 0.15)):
            responses = rng.integers(0, categories, size=(n_persons, n_items))
            responses[rng.random(responses.shape) < missing_fraction] = -1
            for integration, abilities in (("empirical", theta), ("quadrature", None)):

                def run():
                    return compute_fit_indices(model, responses, theta=abilities)

                times = _time(run, repeats=repeats, warmups=warmups)
                results.append(
                    BenchResult(
                        f"model_fit_{name}_{label}_{integration}",
                        times,
                        _peak_traced_bytes(run),
                    )
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
    if "bayesian" in suites:
        results.extend(bench_bayesian(n_persons, n_items, repeats, warmups))
    if "patterns" in suites:
        results.extend(bench_patterns(n_persons, n_items, repeats, warmups))
    if "data" in suites:
        results.extend(bench_data(n_persons, n_items, repeats, warmups))
    if "diagnostics" in suites:
        results.extend(bench_diagnostics(n_persons, n_items, repeats, warmups))
    if "misfit" in suites:
        results.extend(bench_misfit(n_persons, n_items, repeats, warmups))
    if "fit-statistics" in suites:
        results.extend(bench_fit_statistics(n_persons, n_items, repeats, warmups))
    if "model-fit" in suites:
        results.extend(bench_model_fit(n_persons, n_items, repeats, warmups))
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
    if "kernel-smoothing" in suites:
        results.extend(bench_kernel_smoothing(n_persons, n_items, repeats, warmups))
    if "empirical" in suites:
        results.extend(bench_empirical(n_persons, n_items, repeats, warmups))
    if "classical" in suites:
        results.extend(bench_classical(n_persons, n_items, repeats, warmups))
    if "reliability" in suites:
        results.extend(bench_reliability(n_persons, n_items, repeats, warmups))
    if "curves" in suites:
        results.extend(bench_curves(n_persons, n_items, repeats, warmups))
    if "asymmetric" in suites:
        results.extend(bench_asymmetric(n_persons, n_items, repeats, warmups))
    if "logistic-information" in suites:
        results.extend(bench_logistic_information(n_persons, n_items, repeats, warmups))
    if "unipolar" in suites:
        results.extend(bench_unipolar(n_persons, n_items, repeats, warmups))
    if "logistic-probability" in suites:
        results.extend(bench_logistic_probability(n_persons, n_items, repeats, warmups))
    if "multidimensional-information" in suites:
        results.extend(
            bench_multidimensional_information(n_persons, n_items, repeats, warmups)
        )
    if "multidimensional-probability" in suites:
        results.extend(
            bench_multidimensional_probability(n_persons, n_items, repeats, warmups)
        )
    if "multidimensional-fit" in suites:
        results.extend(bench_multidimensional_fit(n_persons, n_items, repeats, warmups))
    if "logistic-fit" in suites:
        results.extend(bench_logistic_fit(n_persons, n_items, repeats, warmups))
    if "weighted-em" in suites:
        results.extend(bench_weighted_em(n_persons, n_items, repeats, warmups))
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
        "waic",
        "psis_normal",
        "psis_heavy_tail",
        "patterns_repeated",
        "patterns_distinct",
        "pairwise_available",
        "mode_imputation",
        "item_statistics",
        "q3_complete",
        "q3_missing",
        "ld_complete",
        "ld_missing",
        "misfit_2pl_complete",
        "misfit_2pl_missing",
        "misfit_grm_complete",
        "misfit_grm_missing",
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
        "weighted_e_step_2pl",
        "weighted_e_step_mirt",
        "weighted_e_step_grm",
        "weighted_fit_2pl",
        "weighted_fit_grm",
    }
    person_workloads.update(
        name
        for name in current_names
        if name.startswith(
            (
                "model_fit_",
                "mean_squares_",
                "itemfit_",
                "personfit_",
                "kernel_smoothing_",
                "empirical_rmsea_",
                "traditional_",
                "reliability_",
                "curves_",
                "asymmetric_",
                "logistic_information_",
                "unipolar_",
                "logistic_probability_",
                "multidimensional_information_",
                "multidimensional_probability_",
                "multidimensional_fit_",
                "logistic_fit_",
            )
        )
    )
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
