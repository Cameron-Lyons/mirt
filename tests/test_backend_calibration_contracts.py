from types import SimpleNamespace

import numpy as np
import pytest

from mirt.backends.rust import calibration as calibration_module


def _fixed_inputs():
    return {
        "responses": np.array(
            [
                [1.0, 0.0, 1.0, -9.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        ),
        "anchor_items": [0, 1],
        "new_items": [2, 3],
        "anchor_disc": np.array([1.2, 0.8]),
        "anchor_diff": np.array([-0.4, 0.6]),
        "theta_grid": np.linspace(-2.0, 2.0, 5),
        "quad_weights": np.array([0.05, 0.2, 0.5, 0.2, 0.05]),
        "max_iter": 10,
        "tol": 1e-5,
    }


def _stocking_inputs():
    return {
        "disc_old": np.array([0.8, 1.1, 1.6]),
        "diff_old": np.array([-0.7, 0.2, 1.0]),
        "disc_new": np.array([0.9, 1.0, 1.4]),
        "diff_new": np.array([-0.5, 0.1, 1.2]),
        "a": 1.05,
        "b": -0.15,
        "theta_grid": np.linspace(-3.0, 3.0, 13),
    }


def _stocking_reference(**kwargs):
    total = 0.0
    for item in range(kwargs["disc_old"].size):
        for theta in kwargs["theta_grid"]:
            old_logit = kwargs["disc_old"][item] * (theta - kwargs["diff_old"][item])
            transformed_theta = kwargs["a"] * theta + kwargs["b"]
            new_logit = kwargs["disc_new"][item] * (
                transformed_theta - kwargs["diff_new"][item]
            )
            old_probability = 1.0 / (1.0 + np.exp(-old_logit))
            new_probability = 1.0 / (1.0 + np.exp(-new_logit))
            total += (old_probability - new_probability) ** 2
    return total


def test_stocking_lord_numpy_matches_scalar_reference(monkeypatch):
    inputs = _stocking_inputs()
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)

    result = calibration_module.stocking_lord_criterion(**inputs)

    assert isinstance(result, float)
    assert result == pytest.approx(_stocking_reference(**inputs), rel=1e-14)


def test_stocking_lord_numpy_preserves_result_across_chunks(monkeypatch):
    inputs = _stocking_inputs()
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)
    monkeypatch.setattr(calibration_module, "_quad_chunk_size", lambda *_: 2)

    result = calibration_module.stocking_lord_criterion(**inputs)

    assert result == pytest.approx(_stocking_reference(**inputs), rel=1e-14)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"disc_old": np.ones((2, 1))}, "one-dimensional"),
        ({"disc_old": np.array([])}, "non-empty"),
        ({"diff_old": np.ones(2)}, "same length"),
        ({"disc_new": np.array([1.0, np.nan, 1.0])}, "finite"),
        ({"a": np.inf}, "a must be a finite"),
        ({"b": "shift"}, "b must be a finite"),
        ({"theta_grid": np.array([])}, "non-empty"),
    ],
)
def test_stocking_lord_rejects_invalid_inputs(monkeypatch, override, message):
    inputs = _stocking_inputs()
    inputs.update(override)
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        calibration_module.stocking_lord_criterion(**inputs)


def test_stocking_lord_native_and_numpy_match(monkeypatch):
    if not calibration_module.mirt_rs:
        pytest.skip("native backend is not installed")
    inputs = _stocking_inputs()
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: True)
    native = calibration_module.stocking_lord_criterion(**inputs)
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)
    numpy_result = calibration_module.stocking_lord_criterion(**inputs)

    assert native == pytest.approx(numpy_result, rel=1e-14, abs=1e-14)


def test_stocking_lord_dispatches_normalized_vectors(monkeypatch):
    captured = {}

    def fake_criterion(*args):
        captured["args"] = args
        return 3.25

    inputs = _stocking_inputs()
    inputs["disc_old"] = [0.8, 1.1, 1.6]
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        calibration_module,
        "mirt_rs",
        SimpleNamespace(stocking_lord_criterion=fake_criterion),
    )

    result = calibration_module.stocking_lord_criterion(**inputs)

    assert result == 3.25
    for vector in (*captured["args"][:4], captured["args"][6]):
        assert vector.dtype == np.float64
        assert vector.flags.c_contiguous


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"responses": np.ones(4)}, "two-dimensional"),
        ({"responses": np.array([[0.0, 2.0, 1.0, 0.0]])}, "coded as 0 or 1"),
        ({"anchor_items": [0, 4]}, "out-of-bounds"),
        ({"anchor_items": [0, 0]}, "duplicate"),
        ({"new_items": [1, 3]}, "must be disjoint"),
        ({"anchor_disc": np.ones(1)}, "shape"),
        ({"anchor_disc": np.array([1.0, 0.0])}, "positive"),
        ({"theta_grid": np.array([])}, "non-empty"),
        ({"quad_weights": np.ones(4)}, "one value per"),
        ({"quad_weights": np.array([0.1, 0.2, -0.1, 0.3, 0.5])}, "non-negative"),
        ({"max_iter": 0}, "greater than or equal to 1"),
        ({"tol": np.inf}, "finite"),
        ({"disc_bounds": (0.0, 5.0)}, "positive lower"),
        ({"prob_clamp": (0.0, 0.99)}, "strictly between"),
        ({"init_disc": 6.0}, "within disc_bounds"),
        ({"min_count": -1.0}, "non-negative"),
        ({"min_valid_points": 6}, "must not exceed"),
    ],
)
def test_fixed_calib_rejects_invalid_inputs_before_dispatch(
    monkeypatch, override, message
):
    inputs = _fixed_inputs()
    inputs.update(override)
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        calibration_module.fixed_calib_em(**inputs)


def test_fixed_calib_dispatches_normalized_inputs(monkeypatch):
    sentinel = (
        np.ones(2),
        np.zeros(2),
        np.zeros(3),
        -4.0,
        2,
        True,
    )
    captured = {}

    def fake_fixed_calib(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        calibration_module,
        "mirt_rs",
        SimpleNamespace(fixed_calib_em=fake_fixed_calib),
    )

    result = calibration_module.fixed_calib_em(**_fixed_inputs())

    assert result is sentinel
    responses, _, _, anchor_disc, anchor_diff, theta_grid, weights, *_ = captured[
        "args"
    ]
    assert responses.dtype == np.int32
    assert responses[0, 3] == -1
    assert responses.flags.c_contiguous
    for vector in (anchor_disc, anchor_diff, theta_grid, weights):
        assert vector.dtype == np.float64
        assert vector.flags.c_contiguous


def test_fixed_calib_validates_before_reporting_missing_backend(monkeypatch):
    monkeypatch.setattr(calibration_module, "rust_enabled", lambda: False)

    with pytest.raises(RuntimeError, match="Rust backend required for fixed_calib_em"):
        calibration_module.fixed_calib_em(**_fixed_inputs())
