from types import SimpleNamespace

import numpy as np
import pytest

from mirt.backends.rust import cat as cat_module


def _item_bank():
    return {
        "discrimination": np.array([0.7, 1.1, 1.6, 0.9]),
        "difficulty": np.array([-1.0, -0.2, 0.7, 1.3]),
    }


def _eap_inputs():
    points = np.linspace(-3.0, 3.0, 15)
    weights = np.exp(-0.5 * points**2)
    weights /= weights.sum()
    return {
        "administered_items": np.array([3, 0, 2, 1], dtype=np.int64),
        "responses": np.array([1.0, 0.0, -7.0, 1.0]),
        **_item_bank(),
        "quad_points": points,
        "quad_weights": weights,
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


def _eap_reference(inputs):
    items = np.asarray(inputs["administered_items"])
    responses = np.asarray(inputs["responses"])
    points = inputs["quad_points"]
    weights = inputs["quad_weights"]
    log_likelihood = np.zeros(points.size)
    for q, point in enumerate(points):
        for position, item in enumerate(items):
            response = responses[position]
            if response < 0:
                continue
            logit = inputs["discrimination"][item] * (
                point - inputs["difficulty"][item]
            )
            sign = 1.0 if response == 1 else -1.0
            log_likelihood[q] -= np.logaddexp(0.0, -sign * logit)
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0.0
    log_weights[positive] = np.log(weights[positive])
    log_posterior = log_likelihood + log_weights
    log_posterior -= np.logaddexp.reduce(log_posterior)
    posterior = np.exp(log_posterior)
    theta = float(posterior @ points)
    standard_error = float(np.sqrt(posterior @ np.square(points - theta)))
    return theta, standard_error


def test_item_information_numpy_matches_formula(monkeypatch):
    bank = _item_bank()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    result = cat_module.cat_compute_item_info(0.35, **bank)

    logits = bank["discrimination"] * (0.35 - bank["difficulty"])
    probability = 1.0 / (1.0 + np.exp(-logits))
    expected = bank["discrimination"] ** 2 * probability * (1.0 - probability)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)


def test_item_information_native_and_numpy_match(monkeypatch):
    if not cat_module.mirt_rs:
        pytest.skip("native backend is not installed")
    bank = _item_bank()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    native = cat_module.cat_compute_item_info(0.35, **bank)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)
    numpy_result = cat_module.cat_compute_item_info(0.35, **bank)

    np.testing.assert_allclose(native, numpy_result, rtol=1e-14, atol=1e-14)


def test_max_information_returns_negative_one_when_none_are_available(monkeypatch):
    bank = _item_bank()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    selected = cat_module.cat_select_max_info(
        0.0, **bank, available_mask=np.zeros(4, dtype=bool)
    )

    assert selected == -1


def test_max_information_respects_availability_mask(monkeypatch):
    bank = _item_bank()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    selected = cat_module.cat_select_max_info(
        0.0, **bank, available_mask=np.array([True, False, True, False])
    )

    information = cat_module.cat_compute_item_info(0.0, **bank)
    expected = int(
        np.flatnonzero([True, False, True, False])[np.argmax(information[[0, 2]])]
    )
    assert selected == expected


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(4, dtype=bool),
        np.array([True, False, True, False]),
    ],
)
def test_max_information_native_and_numpy_match(monkeypatch, mask):
    if not cat_module.mirt_rs:
        pytest.skip("native backend is not installed")
    bank = _item_bank()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    native = cat_module.cat_select_max_info(0.0, **bank, available_mask=mask)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)
    numpy_result = cat_module.cat_select_max_info(0.0, **bank, available_mask=mask)

    assert native == numpy_result


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"theta": np.inf}, "theta must be a finite"),
        ({"discrimination": np.ones((2, 1))}, "one-dimensional"),
        ({"difficulty": np.ones(2)}, "same length"),
        ({"difficulty": np.array([0.0, 0.0, np.nan, 0.0])}, "finite"),
    ],
)
def test_item_information_rejects_invalid_inputs(monkeypatch, override, message):
    inputs = {"theta": 0.0, **_item_bank()}
    inputs.update(override)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        cat_module.cat_compute_item_info(**inputs)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.ones(4, dtype=np.int8), "boolean vector"),
        (np.ones(3, dtype=bool), "one value per item"),
        (np.ones((4, 1), dtype=bool), "boolean vector"),
    ],
)
def test_max_information_rejects_invalid_masks(monkeypatch, mask, message):
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        cat_module.cat_select_max_info(0.0, **_item_bank(), available_mask=mask)


def test_eap_numpy_matches_stable_scalar_reference(monkeypatch):
    inputs = _eap_inputs()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    result = cat_module.cat_eap_update(**inputs)

    np.testing.assert_allclose(result, _eap_reference(inputs), rtol=1e-14, atol=1e-14)


def test_eap_numpy_preserves_result_across_chunks(monkeypatch):
    inputs = _eap_inputs()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)
    monkeypatch.setattr(cat_module, "_quad_chunk_size", lambda *_: 2)

    result = cat_module.cat_eap_update(**inputs)

    np.testing.assert_allclose(result, _eap_reference(inputs), rtol=1e-14, atol=1e-14)


def test_eap_empty_history_returns_prior_moments(monkeypatch):
    inputs = _eap_inputs()
    inputs["administered_items"] = np.array([], dtype=np.int32)
    inputs["responses"] = np.array([], dtype=np.int32)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    theta, standard_error = cat_module.cat_eap_update(**inputs)

    weights = inputs["quad_weights"] / inputs["quad_weights"].sum()
    expected_theta = float(weights @ inputs["quad_points"])
    expected_se = float(
        np.sqrt(weights @ np.square(inputs["quad_points"] - expected_theta))
    )
    assert theta == pytest.approx(expected_theta, abs=1e-14)
    assert standard_error == pytest.approx(expected_se, abs=1e-14)


def test_eap_extreme_logits_remain_finite(monkeypatch):
    inputs = _eap_inputs()
    inputs.update(
        administered_items=np.array([0, 1, 2, 3]),
        responses=np.array([1, 0, 1, 0]),
        discrimination=np.full(4, 1e4),
        difficulty=np.array([-20.0, 20.0, -20.0, 20.0]),
    )
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    result = cat_module.cat_eap_update(**inputs)

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, _eap_reference(inputs), rtol=1e-14, atol=1e-14)


def test_eap_native_and_numpy_match(monkeypatch):
    if not cat_module.mirt_rs:
        pytest.skip("native backend is not installed")
    inputs = _eap_inputs()
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    native = cat_module.cat_eap_update(**inputs)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)
    numpy_result = cat_module.cat_eap_update(**inputs)

    np.testing.assert_allclose(native, numpy_result, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"administered_items": np.array([0.0, 1.0])}, "integer vector"),
        ({"administered_items": np.array([0, 4])}, "out-of-bounds"),
        ({"responses": np.array([1, 0])}, "one value per"),
        ({"responses": np.array([1, 2, 0, 1])}, "coded as 0 or 1"),
        ({"responses": np.array([1, np.inf, 0, 1])}, "finite"),
        ({"quad_weights": np.ones(4)}, "one value per"),
        ({"quad_weights": np.array([1.0, -1.0] + [0.0] * 13)}, "non-negative"),
    ],
)
def test_eap_rejects_invalid_inputs_before_dispatch(monkeypatch, override, message):
    inputs = _eap_inputs()
    inputs.update(override)
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    with pytest.raises(ValueError, match=message):
        cat_module.cat_eap_update(**inputs)


def test_eap_dispatches_normalized_arrays(monkeypatch):
    captured = {}

    def fake_eap(*args):
        captured["args"] = args
        return np.array([0.25]), np.array([0.75])

    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(cat_module, "mirt_rs", SimpleNamespace(cat_eap_update=fake_eap))

    result = cat_module.cat_eap_update(**_eap_inputs())

    assert result == (0.25, 0.75)
    items, responses, disc, diff, points, weights = captured["args"]
    assert items.dtype == np.int32
    assert responses.dtype == np.int32
    assert responses[2] == -1
    for vector in (disc, diff, points, weights):
        assert vector.dtype == np.float64
        assert vector.flags.c_contiguous


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"true_thetas": np.array([])}, "non-empty"),
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
        cat_module.cat_simulate_batch(**inputs)


@pytest.mark.parametrize(
    ("function_name", "theta_name"),
    [
        ("cat_simulate_batch", "true_thetas"),
        ("cat_conditional_mse", "eval_thetas"),
    ],
)
def test_simulation_dispatches_normalized_inputs(
    monkeypatch, function_name, theta_name
):
    captured = {}
    sentinel = (np.array([1.0]),) * 4

    def fake_simulation(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(cat_module, "rust_enabled", lambda: True)
    monkeypatch.setattr(
        cat_module, "mirt_rs", SimpleNamespace(**{function_name: fake_simulation})
    )

    result = getattr(cat_module, function_name)(**_simulation_inputs(theta_name))

    assert result is sentinel
    for vector in captured["args"][:5]:
        assert vector.dtype == np.float64
        assert vector.flags.c_contiguous
    assert captured["args"][5:] == (0.3, 4, 2, 3, 91)


def test_optional_simulation_returns_none_without_native_backend(monkeypatch):
    monkeypatch.setattr(cat_module, "rust_enabled", lambda: False)

    assert cat_module.cat_simulate_batch(**_simulation_inputs()) is None
    assert cat_module.cat_conditional_mse(**_simulation_inputs("eval_thetas")) is None
