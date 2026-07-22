#!/usr/bin/env python3
"""Timing harness for EM fit, scoring, and CAT (not run in default pytest)."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np

import mirt
from mirt.cat import CATEngine


@dataclass
class BenchResult:
    name: str
    times: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times)


def _time(fn, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def bench_em_fit(n_persons: int, n_items: int, repeats: int) -> BenchResult:
    responses = mirt.simdata(model="2PL", n_persons=n_persons, n_items=n_items, seed=42)

    def run() -> None:
        mirt.fit_mirt(responses, model="2PL", n_quadpts=21, max_iter=80, tol=1e-3)

    return BenchResult("em_fit_2pl", _time(run, repeats))


def bench_scoring(n_persons: int, n_items: int, repeats: int) -> BenchResult:
    responses = mirt.simdata(model="2PL", n_persons=n_persons, n_items=n_items, seed=43)
    fit = mirt.fit_mirt(responses, model="2PL", n_quadpts=21, max_iter=80, tol=1e-3)

    def run() -> None:
        mirt.fscores(fit, responses, method="EAP")

    return BenchResult("eap_scoring", _time(run, repeats))


def bench_cat(n_items: int, repeats: int) -> BenchResult:
    responses = mirt.simdata(model="2PL", n_persons=400, n_items=n_items, seed=44)
    fit = mirt.fit_mirt(responses, model="2PL", n_quadpts=21, max_iter=80, tol=1e-3)
    engine = CATEngine(
        fit.model,
        item_selection="MFI",
        stopping_rule="SE",
        se_threshold=0.35,
        max_items=min(15, n_items),
    )
    thetas = np.linspace(-2, 2, 20)

    def run() -> None:
        for th in thetas:
            engine.run_simulation(true_theta=float(th))

    return BenchResult("cat_batch_20", _time(run, repeats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--persons", type=int, default=500)
    parser.add_argument("--items", type=int, default=25)
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "rust"),
        default="auto",
        help="Force computational backend for the run",
    )
    args = parser.parse_args()

    if args.backend == "rust" and not mirt.is_rust_available():
        raise SystemExit("Rust backend requested but extension is not available")
    mirt.set_backend(args.backend)

    info = mirt.get_backend_info()
    print(
        f"backend={info['current_backend']} effective={info['effective_backend']} "
        f"rust={info['rust_available']}"
    )

    results = [
        bench_em_fit(args.persons, args.items, args.repeats),
        bench_scoring(args.persons, args.items, args.repeats),
        bench_cat(args.items, args.repeats),
    ]
    for r in results:
        print(
            f"{r.name:16s}  median={r.median:.4f}s  mean={r.mean:.4f}s  "
            f"n={len(r.times)}"
        )


if __name__ == "__main__":
    main()
