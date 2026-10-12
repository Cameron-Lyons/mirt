from types import SimpleNamespace

import numpy as np
import pytest

from mirt.backends.rust import cat as cat_module


def _item_bank():
    return {
        "discrimination": np.array([0.7, 1.1, 1.6, 0.9]),
        "difficulty": np.array([-1.0, -0.2, 0.7, 1.3]),
    }


def _simulation_inputs(theta_name="true_thetas"):
    points = np.linspace(-3.0, 3.0, 11)
    weights = np.exp(-0.5 * points**2)
    weights /= weights.sum()
    return {
        theta_name: np.array([-1.0, 0.0, 1.0]),
        **_item_bank(),
        "quad_points": points,
        "quad_weights": weights,
        "se_threshold": 0.3,
        "max_items": 4,
        "min_items": 2,
        "n_replications": 3,
        "seed": 91,
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"true_thetas": np.array([])}, "non-empty"),
        ({"true_thetas": np.array([0.0, np.nan])}, "only finite values"),
        ({"difficulty": np.array([-1.0, 0.0, 1.0])}, "same length"),
        ({"quad_weights": np.ones(4)}, "one value per quad_points entry"),
        ({"quad_weights": np.zeros(11)}, "non-negative with a positive sum"),
        ({"se_threshold": np.inf}, "finite number"),
        ({"se_threshold": -0.1}, "non-negative"),
        ({"max_items": 0}, "greater than or equal to 1"),
        ({"min_items": 5}, "must not exceed"),
        ({"n_replications": 0}, "greater than or equal to 1"),
        ({"seed": -1}, "between 0"),
        ({"seed": 2**64}, "between 0"),
        ({"seed": True}, "between 0"),
    ],
)
def test_batch_simulation_rejects_invalid_inputs(monkeypatch, override, message):
    inputs = _simulation_inputs()
    inputs.update(override)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        cat_module.cat_simulate_batch_full(**inputs)


@pytest.mark.parametrize(
    ("function_name", "native_name", "theta_name"),
    [
        ("cat_simulate_batch_full", "cat_simulate_batch", "true_thetas"),
        ("cat_conditional_mse", "cat_conditional_mse", "eval_thetas"),
    ],
)
def test_simulation_dispatches_normalized_inputs(
    monkeypatch, function_name, native_name, theta_name
):
    captured = {}
    sentinel = (np.array([1.0]),) * (6 if native_name == "cat_simulate_batch" else 4)

    def fake_simulation(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        cat_module, "mirt_rs", SimpleNamespace(**{native_name: fake_simulation})
    )

    result = getattr(cat_module, function_name)(**_simulation_inputs(theta_name))

    assert result is sentinel
    for vector in captured["args"][:5]:
        assert vector.dtype == np.float64
        assert vector.flags.c_contiguous
    assert captured["args"][5:] == (0.3, 4, 2, 3, 91)


def test_full_batch_simulation_retains_native_paths(monkeypatch):
    sentinel = (
        np.array([0.1]),
        np.array([0.2]),
        np.array([2], dtype=np.int32),
        np.array([0.0]),
        np.array([[3, 1, -1]], dtype=np.int32),
        np.array([[1, 0, -1]], dtype=np.int32),
    )
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        cat_module,
        "mirt_rs",
        SimpleNamespace(cat_simulate_batch=lambda *args: sentinel),
    )

    result = cat_module.cat_simulate_batch_full(**_simulation_inputs())

    assert result is sentinel


def test_full_batch_simulation_rejects_invalid_native_payload(monkeypatch):
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        cat_module,
        "mirt_rs",
        SimpleNamespace(cat_simulate_batch=lambda *args: (np.array([1.0]),) * 4),
    )

    with pytest.raises(RuntimeError, match="invalid payload"):
        cat_module.cat_simulate_batch_full(**_simulation_inputs())


def test_installed_full_batch_paths_align_with_counts(monkeypatch):
    if not cat_module.mirt_rs:
        pytest.skip("native backend is not installed")
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)

    result = cat_module.cat_simulate_batch_full(**_simulation_inputs())

    assert result is not None
    theta, standard_error, counts, true_theta, items, responses = result
    assert (
        theta.shape == standard_error.shape == counts.shape == true_theta.shape == (9,)
    )
    assert items.shape == responses.shape == (9, 4)
    for row, count_value in enumerate(counts):
        count = int(count_value)
        assert 2 <= count <= 4
        assert len(set(items[row, :count].tolist())) == count
        assert np.isin(responses[row, :count], [0, 1]).all()
        assert np.all(items[row, count:] == -1)
        assert np.all(responses[row, count:] == -1)


def test_optional_simulation_returns_none_without_native_backend(monkeypatch):
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    assert cat_module.cat_simulate_batch_full(**_simulation_inputs()) is None
    assert cat_module.cat_conditional_mse(**_simulation_inputs("eval_thetas")) is None
