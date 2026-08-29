from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import mirt
from mirt.diagnostics._utils import fit_group_models
from mirt.estimation.em import EMEstimator


def _responses() -> np.ndarray:
    return np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )


def test_parameter_only_fit_skips_em_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_inference(*args: Any, **kwargs: Any) -> None:
        pytest.fail("standard-error computation should be skipped")

    monkeypatch.setattr(EMEstimator, "_compute_standard_errors", unexpected_inference)

    result = mirt.fit_mirt(
        _responses(),
        model="1PL",
        n_quadpts=5,
        max_iter=2,
        use_rust=False,
        compute_standard_errors=False,
    )

    assert result.standard_errors == {}
    statistics = result.parameter_statistics()
    assert statistics
    assert all(
        np.isnan(values["standard_error"]).all() for values in statistics.values()
    )


def test_parameter_only_native_fit_preserves_estimates() -> None:
    options = {
        "model": "2PL",
        "n_quadpts": 7,
        "max_iter": 3,
        "use_rust": True,
    }

    full = mirt.fit_mirt(_responses(), compute_standard_errors=True, **options)
    parameter_only = mirt.fit_mirt(
        _responses(), compute_standard_errors=False, **options
    )

    assert full.standard_errors
    assert parameter_only.standard_errors == {}
    assert parameter_only.log_likelihood == pytest.approx(full.log_likelihood)
    for name, values in full.model.parameters.items():
        np.testing.assert_allclose(parameter_only.model.parameters[name], values)


@pytest.mark.parametrize("value", [0, 1, None, "no"])
def test_parameter_only_control_requires_a_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="compute_standard_errors"):
        mirt.fit_mirt(_responses(), compute_standard_errors=value)  # type: ignore[arg-type]


def test_group_fits_skip_inference_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: list[bool] = []

    def fake_fit(data: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        options.append(kwargs["compute_standard_errors"])
        return SimpleNamespace(model=SimpleNamespace())

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
    responses = _responses()

    fit_group_models(responses[:4], responses[4:])
    fit_group_models(responses[:4], responses[4:], compute_standard_errors=True)

    assert options == [False, False, True, True]


def test_spawned_dtf_bootstrap_matches_serial_results() -> None:
    responses = _responses()
    data = np.vstack((responses, responses[:, ::-1]))
    groups = np.repeat((0, 1), responses.shape[0])
    options = {
        "model": "1PL",
        "n_quadpts": 7,
        "n_bootstrap": 4,
        "random_state": 2026,
        "max_iter": 2,
        "use_rust": False,
    }

    serial = mirt.compute_dtf(data, groups, n_jobs=1, **options)
    parallel = mirt.compute_dtf(data, groups, n_jobs=2, **options)

    assert parallel["DTF"] == pytest.approx(serial["DTF"])
    assert parallel["DTF_SE"] == pytest.approx(serial["DTF_SE"])
    assert parallel["p_value"] == pytest.approx(serial["p_value"])
    np.testing.assert_allclose(
        parallel["confidence_interval"], serial["confidence_interval"]
    )
    assert parallel["n_bootstrap_successful"] == 4
    assert parallel["n_bootstrap_failed"] == 0


def test_spawned_reliability_bootstrap_matches_serial_results() -> None:
    responses = _responses()
    data = np.vstack((responses, responses[:, ::-1]))
    groups = np.repeat((0, 1), responses.shape[0])
    options = {
        "model": "1PL",
        "n_bootstrap": 4,
        "seed": 2026,
        "n_points": 7,
        "max_iter": 2,
        "use_rust": False,
    }

    serial = mirt.reliability_invariance(data, groups, n_jobs=1, **options)
    parallel = mirt.reliability_invariance(data, groups, n_jobs=2, **options)

    for key in (
        "reliability_ref",
        "reliability_focal",
        "reliability_diff",
        "reliability_diff_se",
        "z",
        "p_value",
    ):
        assert parallel[key] == pytest.approx(serial[key])
    np.testing.assert_allclose(
        parallel["reliability_diff_ci"], serial["reliability_diff_ci"]
    )
    assert parallel["n_bootstrap_successful"] == 4
    assert parallel["n_bootstrap_failed"] == 0
