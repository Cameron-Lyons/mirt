"""Numerical and memory contracts for streaming misfit identification."""

import tracemalloc

import numpy as np
import pytest

from mirt.constants import PROB_EPSILON
from mirt.diagnostics import residuals
from mirt.models import GradedResponseModel, TwoParameterLogistic


def _reference_flags(model, responses, theta, z_threshold, outfit_threshold):
    probabilities = model.probability(theta)
    if probabilities.ndim == 3:
        categories = np.arange(probabilities.shape[-1])
        expected = probabilities @ categories
        variance = probabilities @ categories**2 - expected**2
    else:
        expected = probabilities
        variance = probabilities * (1 - probabilities)
    z = np.where(
        responses >= 0,
        (responses - expected) / np.sqrt(variance + PROB_EPSILON),
        np.nan,
    )
    flags = {"misfitting_persons": [], "misfitting_items": [], "aberrant_responses": []}
    for axis, name, key in (
        (0, "misfitting_items", "item"),
        (1, "misfitting_persons", "person"),
    ):
        for index in range(responses.shape[1 - axis]):
            scores = z[:, index] if axis == 0 else z[index]
            variances = variance[:, index] if axis == 0 else variance[index]
            valid = np.isfinite(scores)
            if not np.any(valid):
                continue
            # Shared mean-square rules: infit has no variance offset and
            # outfit skips near-deterministic entries (variance <= epsilon).
            observed = responses[:, index] if axis == 0 else responses[index]
            means = expected[:, index] if axis == 0 else expected[index]
            squares = (observed[valid] - means[valid]) ** 2
            weights = np.maximum(variances[valid], 0.0)
            eligible = weights > PROB_EPSILON
            outfit = (
                np.mean(squares[eligible] / weights[eligible])
                if np.any(eligible)
                else np.nan
            )
            infit = (
                squares.sum() / weights.sum()
                if weights.sum() > PROB_EPSILON
                else np.nan
            )
            if outfit > outfit_threshold:
                flags[name].append({key: index, "outfit": outfit, "infit": infit})
    for i in range(len(responses)):
        for j in range(responses.shape[1]):
            if np.isfinite(z[i, j]) and abs(z[i, j]) > z_threshold:
                flags["aberrant_responses"].append(
                    {
                        "person": i,
                        "item": j,
                        "response": responses[i, j],
                        "expected": expected[i, j],
                        "z": z[i, j],
                    }
                )
    return flags


def _assert_flags_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name, entries in actual.items():
        assert len(entries) == len(expected[name])
        for entry, reference in zip(entries, expected[name], strict=True):
            assert entry.keys() == reference.keys()
            for key, value in reference.items():
                if key in ("item", "person", "response"):
                    assert entry[key] == value
                else:
                    np.testing.assert_allclose(
                        entry[key], value, rtol=2e-13, atol=1e-14
                    )


@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("n_factors", [1, 2])
@pytest.mark.parametrize("block_rows", [1, 7, 1000])
def test_flags_match_independent_reference(
    monkeypatch, polytomous, n_factors, block_rows
):
    rng = np.random.default_rng(4912)
    categories = [2, 3, 5, 4, 3] if polytomous else [2] * 5
    model = (
        GradedResponseModel(5, n_categories=categories, n_factors=n_factors)
        if polytomous
        else TwoParameterLogistic(5, n_factors=n_factors)
    )
    theta = rng.normal(size=(74, n_factors))[::2]
    responses = rng.integers(0, categories, size=(74, 5))[::2]
    responses[rng.random(responses.shape) < 0.15] = -7
    responses[0] = -1
    responses[:, -1] = -1
    responses.setflags(write=False)
    theta.setflags(write=False)
    before = responses.copy()
    expected = _reference_flags(model, responses, theta, 1.25, 1.1)
    width = max(categories) if polytomous else 1
    monkeypatch.setattr(
        residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", block_rows * 5 * width
    )

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("misfit identification must not build a full analysis")

    monkeypatch.setattr(residuals, "analyze_residuals", unexpected_analysis)
    original = residuals._compute_residual_arrays
    requested = []

    def capture(model, responses, theta, residual_types, **kwargs):
        requested.append(residual_types)
        assert len(responses) <= block_rows
        return original(model, responses, theta, residual_types, **kwargs)

    monkeypatch.setattr(residuals, "_compute_residual_arrays", capture)
    actual = residuals.identify_misfitting_patterns(
        model, responses, theta, z_threshold=1.25, outfit_threshold=1.1
    )
    _assert_flags_equal(actual, expected)
    assert requested and all(kinds == ("standardized",) for kinds in requested)
    np.testing.assert_array_equal(responses, before)


def test_probability_calls_are_bounded_and_include_every_person_once(monkeypatch):
    model = GradedResponseModel(3, n_categories=[2, 3, 5])
    responses = np.ones((17, 3), dtype=int)
    theta = np.linspace(-1, 1, 17)
    original = model.probability
    seen = []

    def capture(theta, item_idx=None):
        assert item_idx is None
        seen.append(theta.copy())
        return original(theta, item_idx)

    monkeypatch.setattr(model, "probability", capture)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 4 * 3 * 5)
    residuals.identify_misfitting_patterns(model, responses, theta)
    assert [len(block) for block in seen] == [4, 4, 4, 4, 1]
    np.testing.assert_array_equal(np.concatenate(seen).ravel(), theta)


def test_estimates_abilities_once_for_full_population(monkeypatch):
    import mirt.scoring

    model = TwoParameterLogistic(3)
    responses = np.array([[0, 1, 0], [1, 0, 1]] * 9)
    theta = np.linspace(-2, 2, len(responses))
    calls = []

    def score(model, data, method):
        from types import SimpleNamespace

        calls.append((data, method))
        return SimpleNamespace(theta=theta)

    monkeypatch.setattr(mirt.scoring, "fscores", score)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 9)
    actual = residuals.identify_misfitting_patterns(model, responses)
    expected = _reference_flags(model, responses, theta, 2, 1.5)
    _assert_flags_equal(actual, expected)
    assert len(calls) == 1
    assert calls[0][0] is responses and calls[0][1] == "EAP"


def test_nonfinite_residuals_and_empty_pairs_do_not_produce_flags(monkeypatch):
    model = TwoParameterLogistic(3)

    def probability(theta, item_idx=None):
        return np.broadcast_to([np.nan, np.inf, 0.5], (len(theta), 3))

    monkeypatch.setattr(model, "probability", probability)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 6)
    with np.errstate(invalid="ignore"):
        actual = residuals.identify_misfitting_patterns(
            model, np.tile([0, 1, -1], (9, 1)), np.zeros(9)
        )
    assert actual == {
        "misfitting_persons": [],
        "misfitting_items": [],
        "aberrant_responses": [],
    }


def test_memory_retains_counts_and_flags_instead_of_residual_matrices(monkeypatch):
    model = TwoParameterLogistic(20)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 500)
    peaks = []
    for n_persons in (250, 10000):
        responses = np.zeros((n_persons, 20), dtype=np.int_)
        theta = np.zeros(n_persons)
        residuals.identify_misfitting_patterns(model, responses, theta)
        tracemalloc.start()
        try:
            result = residuals.identify_misfitting_patterns(model, responses, theta)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert not any(result.values())
    # Person sums/counts are linear in respondents; residual matrices would
    # exceed this allowance even before unused analyses and pattern dictionaries.
    assert peaks[1] < peaks[0] + 2 * 1024**2


def test_unsupported_batch_output_retains_itemwise_fallback(monkeypatch):
    model = TwoParameterLogistic(2)
    responses = np.array([[0, 1], [1, -1], [1, 0]] * 3)
    theta = np.linspace(-2, 2, 9)
    expected = _reference_flags(model, responses, theta, 1.25, 1.1)
    original = model.probability
    calls = []

    def probability(theta, item_idx=None):
        calls.append(item_idx)
        if item_idx is None:
            return np.zeros((len(theta), 1))
        return original(theta, item_idx)

    monkeypatch.setattr(model, "probability", probability)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 6)
    actual = residuals.identify_misfitting_patterns(
        model, responses, theta, z_threshold=1.25, outfit_threshold=1.1
    )
    _assert_flags_equal(actual, expected)
    assert calls == [None, 0, 1] * 3


def test_thresholds_remain_strict_across_blocks(monkeypatch):
    model = TwoParameterLogistic(2)
    responses = np.array([[0, -1], [1, 1]])
    theta = np.zeros(2)
    threshold = 0.5 / np.sqrt(0.25 + PROB_EPSILON)
    monkeypatch.setattr(residuals, "_MISFIT_TARGET_CHUNK_ELEMENTS", 2)
    # Every observed response has p = 0.5, so each item and person outfit is
    # exactly (0.5 ** 2) / 0.25 = 1.
    result = residuals.identify_misfitting_patterns(
        model, responses, theta, z_threshold=threshold, outfit_threshold=1.0
    )
    assert not any(result.values())
    below = np.nextafter(threshold, 0)
    flagged = residuals.identify_misfitting_patterns(
        model, responses, theta, z_threshold=below, outfit_threshold=1.0
    )
    assert [
        (entry["person"], entry["item"]) for entry in flagged["aberrant_responses"]
    ] == [(0, 0), (1, 0), (1, 1)]
