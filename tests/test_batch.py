"""Regression coverage for validated and parallel batch fitting."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import mirt
import mirt.utils as mirt_utils
import mirt.utils.batch as batch_utils
import mirt.utils.dataframe as dataframe_utils
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.batch import (
    BatchFitResult,
    GridFitResult,
    fit_model_grid,
    fit_models,
)

RESPONSES = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])


def _result(model: str, *, bic: float = 10.0, converged: bool = True):
    return SimpleNamespace(
        model=SimpleNamespace(
            model_name=model,
            parameters={"parameter": np.zeros(1)},
        ),
        log_likelihood=-4.0,
        aic=bic + 1.0,
        bic=bic,
        converged=converged,
        n_observations=len(RESPONSES),
    )


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([], "at least one"),
        (["1PL", "1PL"], "duplicate"),
        (["unsupported"], "unsupported"),
        ("1PL", "sequence"),
    ],
)
def test_fit_models_rejects_invalid_model_sequences(models, message, monkeypatch):
    monkeypatch.setattr(mirt, "fit_mirt", lambda *args, **kwargs: _result("1PL"))
    with pytest.raises(MirtValidationError, match=message):
        fit_models(models, RESPONSES)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5])
def test_fit_models_rejects_invalid_worker_counts(n_jobs):
    with pytest.raises(MirtValidationError, match="n_jobs"):
        fit_models(["1PL"], RESPONSES, n_jobs=n_jobs)


def test_fit_models_rejects_unknown_error_policy():
    with pytest.raises(MirtValidationError, match="on_error"):
        fit_models(["1PL"], RESPONSES, on_error="ignore")  # type: ignore[arg-type]


def test_fit_models_rejects_unknown_parallel_backend():
    with pytest.raises(MirtValidationError, match="parallel_backend"):
        fit_models(
            ["1PL"],
            RESPONSES,
            parallel_backend="subprocess",
        )  # type: ignore[arg-type]


def test_process_backend_reports_platform_limit(monkeypatch):
    def unavailable_executor(*args: Any, **kwargs: Any):
        del args, kwargs
        raise PermissionError("semaphores unavailable")

    monkeypatch.setattr(batch_utils, "ProcessPoolExecutor", unavailable_executor)
    with pytest.raises(RuntimeError, match="process parallel backend is unavailable"):
        fit_models(
            ["1PL", "2PL"],
            RESPONSES,
            n_jobs=2,
            parallel_backend="process",
        )


def test_process_backend_configures_responses_through_initializer(monkeypatch):
    initialization_args: list[tuple[Any, ...]] = []
    submitted_args: list[tuple[Any, ...]] = []
    monkeypatch.setattr(batch_utils, "_PROCESS_RESPONSES", None)

    class ImmediateProcessPool:
        def __init__(
            self,
            *,
            initializer,
            initargs,
            **kwargs,
        ):
            del kwargs
            initialization_args.append(initargs)
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def submit(self, function, *args):
            submitted_args.append(args)
            future = Future()
            try:
                future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(batch_utils, "ProcessPoolExecutor", ImmediateProcessPool)
    monkeypatch.setattr(
        mirt,
        "fit_mirt",
        lambda data, model, **kwargs: _result(model),
    )

    batch = fit_models(
        ["1PL", "2PL"],
        RESPONSES,
        n_jobs=2,
        parallel_backend="process",
    )

    assert list(batch.results) == ["1PL", "2PL"]
    assert len(initialization_args) == 1
    assert len(initialization_args[0]) == 1
    assert initialization_args[0][0] is RESPONSES
    assert len(submitted_args) == 2
    assert all(len(args) == 1 for args in submitted_args)
    assert all(
        not any(isinstance(value, np.ndarray) for value in vars(args[0]).values())
        for args in submitted_args
    )


def test_fit_models_validates_responses_before_fitting(monkeypatch):
    calls = 0

    def fake_fit(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return _result("1PL")

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    with pytest.raises(MirtDataError, match="2D"):
        fit_models(["1PL"], np.array([0, 1]))

    assert calls == 0


def test_fit_models_runs_independent_models_concurrently(monkeypatch):
    barrier = threading.Barrier(2)
    thread_ids: set[int] = set()

    def fake_fit(data, model, **kwargs):
        del data, kwargs
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=2)
        return _result(model, bic=9.0 if model == "2PL" else 12.0)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    batch = fit_models(["1PL", "2PL"], RESPONSES, n_jobs=2)

    assert list(batch.results) == ["1PL", "2PL"]
    assert batch.best_model == "2PL"
    assert len(thread_ids) == 2


def test_fit_models_skip_policy_retains_failure_details(monkeypatch):
    def fake_fit(data, model, **kwargs):
        del data, kwargs
        if model == "1PL":
            raise ValueError("incompatible start")
        return _result(model)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    batch = fit_models(
        ["1PL", "2PL"],
        RESPONSES,
        n_jobs=2,
        on_error="skip",
    )

    assert list(batch.results) == ["2PL"]
    assert batch.failures == {"1PL": "ValueError: incompatible start"}
    assert batch.best_model == "2PL"
    assert "Models failed: 1" in batch.summary()


def test_fit_models_raise_policy_adds_model_context(monkeypatch):
    def fake_fit(*args: Any, **kwargs: Any):
        del args, kwargs
        raise ValueError("bad fit")

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    with pytest.raises(RuntimeError, match="fit failed for 1PL") as exc_info:
        fit_models(["1PL"], RESPONSES)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_fit_models_rejects_all_failed_skip_batch(monkeypatch):
    def fake_fit(*args: Any, **kwargs: Any):
        del args, kwargs
        raise ArithmeticError("no finite solution")

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    with pytest.raises(RuntimeError, match="all requested model fits failed"):
        fit_models(["1PL", "2PL"], RESPONSES, on_error="skip")


def test_fit_models_treats_nonfinite_statistics_as_failures(monkeypatch):
    def fake_fit(data, model, **kwargs):
        del data, kwargs
        if model == "1PL":
            return _result(model, bic=np.inf)
        return _result(model)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    batch = fit_models(["1PL", "2PL"], RESPONSES, on_error="skip")

    assert list(batch.results) == ["2PL"]
    assert "non-finite" in batch.failures["1PL"]


def test_fit_models_falls_back_without_dataframe_dependencies(monkeypatch):
    monkeypatch.setattr(
        mirt,
        "fit_mirt",
        lambda data, model, **kwargs: _result(model),
    )

    def missing_backend(*args: Any, **kwargs: Any):
        del args, kwargs
        raise ImportError("no dataframe backend")

    monkeypatch.setattr(dataframe_utils, "create_dataframe", missing_backend)
    batch = fit_models(["1PL"], RESPONSES)

    assert batch.comparison == [
        {
            "Model": "1PL",
            "LogLik": -4.0,
            "AIC": 11.0,
            "BIC": 10.0,
            "Converged": True,
        }
    ]


@pytest.mark.parametrize(
    ("keyword", "values", "message"),
    [
        ("n_factors_range", [], "at least one"),
        ("n_factors_range", [1, 1], "duplicate"),
        ("n_factors_range", [0], "greater than or equal to 1"),
        ("n_factors_range", [True], "greater than or equal to 1"),
        ("n_quadpts_range", [4], "greater than or equal to 5"),
        ("n_quadpts_range", [5, 5], "duplicate"),
    ],
)
def test_fit_model_grid_rejects_invalid_ranges(keyword, values, message):
    with pytest.raises(MirtValidationError, match=message):
        fit_model_grid(["1PL"], RESPONSES, **{keyword: values})


def test_fit_model_grid_retains_skipped_combination_details(monkeypatch):
    def fake_fit(data, model, n_factors, **kwargs):
        del data, kwargs
        return _result(model, bic=10.0 + n_factors)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    grid = fit_model_grid(
        ["1PL", "2PL"],
        RESPONSES,
        n_factors_range=[1, 2],
        n_quadpts_range=[5],
        n_jobs=2,
    )

    assert isinstance(grid, dict)
    assert isinstance(grid, GridFitResult)
    assert list(grid) == ["1PL_f1_q5", "2PL_f1_q5", "2PL_f2_q5"]
    assert "only supports n_factors=1" in grid.failures["1PL_f2_q5"]


@pytest.mark.parametrize("model", ["1PL", "3PL", "4PL", "PCM"])
def test_fit_model_grid_does_not_run_unidimensional_models_as_multidimensional(
    model, monkeypatch
):
    calls = 0

    def fake_fit(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return _result(model)

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    grid = fit_model_grid(
        [model],  # type: ignore[list-item]
        RESPONSES,
        n_factors_range=[2],
        n_quadpts_range=[5],
    )

    assert not grid
    assert calls == 0
    assert "only supports n_factors=1" in next(iter(grid.failures.values()))


def test_fit_model_grid_can_raise_on_failed_combination(monkeypatch):
    def fake_fit(*args: Any, **kwargs: Any):
        del args, kwargs
        raise ValueError("bad combination")

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    with pytest.raises(RuntimeError, match="1PL_f1_q5"):
        fit_model_grid(
            ["1PL"],
            RESPONSES,
            n_quadpts_range=[5],
            on_error="raise",
        )


def test_parallel_and_sequential_real_fits_match():
    responses = np.tile(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        ),
        (5, 1),
    )
    kwargs = {
        "responses": responses,
        "n_quadpts": 5,
        "max_iter": 2,
        "tol": 1e-3,
    }

    sequential = fit_models(["1PL", "3PL"], n_jobs=1, **kwargs)
    parallel = fit_models(["1PL", "3PL"], n_jobs=2, **kwargs)

    assert list(parallel.results) == list(sequential.results)
    assert parallel.best_model == sequential.best_model
    for model in sequential.results:
        assert parallel.results[model].log_likelihood == pytest.approx(
            sequential.results[model].log_likelihood
        )


def test_process_and_sequential_real_fits_match():
    responses = np.tile(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        ),
        (5, 1),
    )
    kwargs = {
        "responses": responses,
        "n_quadpts": 5,
        "max_iter": 2,
        "tol": 1e-3,
    }

    sequential = fit_models(["1PL", "3PL"], n_jobs=1, **kwargs)
    parallel = fit_models(
        ["1PL", "3PL"],
        n_jobs=2,
        parallel_backend="process",
        **kwargs,
    )

    assert parallel.best_model == sequential.best_model
    for model in sequential.results:
        assert parallel.results[model].log_likelihood == pytest.approx(
            sequential.results[model].log_likelihood
        )


def test_process_grid_returns_successes_and_serialized_failures():
    responses = np.tile(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        ),
        (5, 1),
    )

    grid = fit_model_grid(
        ["1PL", "2PL"],
        responses,
        n_factors_range=[2],
        n_quadpts_range=[5],
        max_iter=1,
        tol=1e-3,
        n_jobs=2,
        parallel_backend="process",
    )

    assert list(grid) == ["2PL_f2_q5"]
    assert "only supports n_factors=1" in grid.failures["1PL_f2_q5"]


def test_batch_symbols_are_available_from_public_namespaces():
    symbols = {
        "fit_models": fit_models,
        "fit_model_grid": fit_model_grid,
        "BatchFitResult": BatchFitResult,
        "GridFitResult": GridFitResult,
    }
    for name, symbol in symbols.items():
        assert getattr(mirt_utils, name) is symbol
        assert getattr(mirt, name) is symbol
