"""Independent references and memory contracts for model-fit moments."""

import tracemalloc

import numpy as np
import pytest

from mirt.diagnostics import modelfit
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models import GradedResponseModel, TwoParameterLogistic


def _reference_moments(means, seconds, valid, weights=None):
    """Compute each pair separately on its own observed respondents."""
    n_rows, n_items = means.shape
    if weights is None:
        weights = np.ones(n_rows)
    uni = np.full(n_items, np.nan)
    bi = np.full((n_items, n_items), np.nan)
    corr = np.full_like(bi, np.nan)
    pair_means = np.full_like(bi, np.nan)
    counts = np.zeros_like(bi)
    for j in range(n_items):
        selected = valid[:, j]
        if np.any(selected):
            uni[j] = np.average(means[selected, j], weights=weights[selected])
        for k in range(n_items):
            selected = valid[:, j] & valid[:, k]
            counts[j, k] = np.count_nonzero(selected)
            if not np.any(selected):
                continue
            mass = weights[selected]
            left, right = means[selected, j], means[selected, k]
            left_mean = np.average(left, weights=mass)
            right_mean = np.average(right, weights=mass)
            bi[j, k] = np.average(left * right, weights=mass)
            pair_means[j, k] = left_mean
            left_var = np.average(seconds[selected, j], weights=mass) - left_mean**2
            right_var = np.average(seconds[selected, k], weights=mass) - right_mean**2
            denominator = np.sqrt(max(left_var, 0) * max(right_var, 0))
            if denominator > 0:
                corr[j, k] = (bi[j, k] - left_mean * right_mean) / denominator
    return uni, bi, corr, pair_means, counts


@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("missing", ["none", "random", "sparse"])
@pytest.mark.parametrize("integration", ["empirical", "quadrature"])
@pytest.mark.parametrize("block_rows", [1, 7, 1000])
def test_streamed_moments_match_pairwise_reference(
    monkeypatch, polytomous, missing, integration, block_rows
):
    rng = np.random.default_rng(935)
    n_rows, n_items = 31, 6
    categories = [2, 3, 4, 5, 3, 4] if polytomous else [2] * n_items
    model = (
        GradedResponseModel(n_items, n_categories=categories, n_factors=2)
        if polytomous
        else TwoParameterLogistic(n_items, n_factors=2)
    )
    # Slicing the backing arrays exercises non-contiguous, read-only input.
    responses = rng.integers(0, categories, size=(2 * n_rows, n_items)).astype(float)[
        ::2
    ]
    theta = rng.normal(size=(2 * n_rows, 2))[::2]
    if missing != "none":
        responses[rng.random(responses.shape) < 0.2] = -7
        responses[0] = np.nan
    if missing == "sparse":
        responses[:, 0] = np.nan
        responses[:, 1] = 0
        responses[:15, 2] = np.nan
        responses[15:, 3] = -1
        responses[:, 4] = -1
        responses[-1, 4] = 1
    responses.setflags(write=False)
    theta.setflags(write=False)
    original = responses.copy()
    valid = np.isfinite(responses) & (responses >= 0)
    observed = _reference_moments(responses, responses**2, valid)

    abilities = theta if integration == "empirical" else None
    weights = None
    expected_valid = valid
    nodes = theta
    if abilities is None:
        quad = GaussHermiteQuadrature(n_points=5, n_dimensions=2)
        nodes = quad.nodes
        weights = quad.weights / quad.weights.sum()
        expected_valid = np.ones((len(nodes), n_items), dtype=bool)
    probabilities = model.probability(nodes)
    if polytomous:
        scores = np.arange(max(categories))
        means, seconds = probabilities @ scores, probabilities @ scores**2
    else:
        means = seconds = probabilities
    expected = _reference_moments(means, seconds, expected_valid, weights)

    width = max(categories) if polytomous else 1
    monkeypatch.setattr(
        modelfit, "_MOMENT_CHUNK_ELEMENTS", block_rows * n_items * width
    )
    values, max_observed = modelfit._validate_diagnostic_inputs(model, responses)
    moments = modelfit._prepare_fit_moments(model, values, max_observed, abilities, 5)
    for actual, reference in (
        (moments.observed_uni, observed[0]),
        (moments.observed_bi, observed[1]),
        (moments.observed_corr, observed[2]),
        (moments.observed_pair_means, observed[3]),
        (moments.pair_counts, observed[4]),
        (moments.uni_counts, observed[4].diagonal()),
        (moments.expected_uni, expected[0]),
        (moments.expected_bi, expected[1]),
        (moments.expected_corr, expected[2]),
    ):
        np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=2e-13)

    fit = modelfit.compute_fit_indices(model, responses, theta=abilities, n_quadpts=5)
    upper = np.triu_indices(n_items, 1)
    # The inferential statistic is checked against an exhaustive multinomial
    # oracle in test_modelfit_oracle.py, rather than a raw residual sum.
    direct = modelfit.compute_m2(model, responses, theta=abilities, n_quadpts=5)
    np.testing.assert_allclose(
        [fit["M2"], fit["M2_df"], fit["M2_p"]],
        [direct["M2"], direct["df"], direct["p_value"]],
        equal_nan=True,
    )
    correlation_residuals = (observed[2] - expected[2])[upper]
    assert fit["SRMSR"] == pytest.approx(np.sqrt(np.nanmean(correlation_residuals**2)))
    np.testing.assert_array_equal(responses, original)


@pytest.mark.parametrize("integration", ["empirical", "quadrature"])
def test_probability_batches_respect_category_width(monkeypatch, integration):
    model = GradedResponseModel(4, n_categories=[2, 3, 4, 5])
    responses = np.zeros((19, 4), dtype=int)
    theta = np.linspace(-2, 2, 19) if integration == "empirical" else None
    original_probability = model.probability
    rows = []

    def counted_probability(theta, item_idx=None):
        rows.append(len(theta))
        return original_probability(theta, item_idx)

    monkeypatch.setattr(model, "probability", counted_probability)
    monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 3 * 4 * 5)
    modelfit.compute_fit_indices(model, responses, theta=theta, n_quadpts=11)
    assert max(rows) <= 3
    # Covariance integration and parameter derivatives also evaluate curves;
    # all of them must honor the probability storage budget.
    assert sum(rows) >= (19 if integration == "empirical" else 11)


@pytest.mark.parametrize(
    "bad_value,message",
    [(np.inf, "infinite"), (0.5, "integer category"), (2, "between 0 and 1")],
)
def test_invalid_responses_in_final_block(monkeypatch, bad_value, message):
    model = TwoParameterLogistic(3)
    responses = np.zeros((11, 3))
    responses[-1, -1] = bad_value
    monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 6)
    with pytest.raises(ValueError, match=message):
        modelfit.compute_m2(model, responses, theta=np.zeros(11))


def test_invalid_probability_in_final_block(monkeypatch):
    model = TwoParameterLogistic(2)
    responses = np.zeros((11, 2))

    def probability(theta):
        values = np.full((len(theta), 2), 0.5)
        values[theta[:, 0] > 9] = np.nan
        return values

    monkeypatch.setattr(model, "probability", probability)
    monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 6)
    with pytest.raises(ValueError, match="model probabilities must be finite"):
        modelfit.compute_m2(model, responses, theta=np.arange(11))


def test_temporary_memory_does_not_grow_with_response_matrix(monkeypatch):
    model = GradedResponseModel(12, n_categories=5)
    monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 4096)
    peaks = []
    for n_persons in (256, 8192):
        responses = np.ones((n_persons, 12), dtype=np.int64)
        responses[::3, 0] = -1
        theta = np.linspace(-2, 2, n_persons)
        modelfit.compute_m2(model, responses, theta=theta)
        tracemalloc.start()
        try:
            modelfit.compute_m2(model, responses, theta=theta)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] < peaks[0] + 256 * 1024


def test_single_item_vector_probability_output(monkeypatch):
    model = TwoParameterLogistic(1)
    monkeypatch.setattr(model, "probability", lambda theta: np.full(len(theta), 0.5))
    monkeypatch.setattr(modelfit, "_MOMENT_CHUNK_ELEMENTS", 3)
    result = modelfit.compute_fit_indices(model, np.array([[0], [1]] * 5))
    assert result["M2"] == 0.0
    assert np.isnan(result["SRMSR"])
    assert np.isnan(result["CFI"])
