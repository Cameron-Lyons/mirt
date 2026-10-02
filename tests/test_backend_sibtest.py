"""Independent inference oracles and native/NumPy parity for SIBTEST backends."""

import numpy as np
import pytest
from scipy import stats

from mirt import sibtest_items
from mirt.backends.rust import diagnostics
from mirt.backends.rust._helpers import RUST_AVAILABLE


@pytest.fixture(params=["numpy", "rust"])
def backend(request, monkeypatch):
    if request.param == "rust" and not RUST_AVAILABLE:
        pytest.skip("native extension unavailable")
    monkeypatch.setattr(diagnostics, "rust_enabled", lambda: request.param == "rust")
    return diagnostics


def _groups_with_cells():
    rows = []
    for group_cells in ([(10, 8), (40, 20)], [(30, 9), (20, 6)]):
        blocks = []
        for score, (count, successes) in enumerate(group_cells):
            block = np.zeros((count, 2), dtype=int)
            block[:, 0] = score
            block[:successes, 1] = 1
            blocks.append(block)
        rows.append(np.vstack(blocks))
    return rows


def test_backend_beta_matches_hand_calculated_pooled_sampling_error(backend):
    reference, focal = _groups_with_cells()
    beta, se, differences, counts = backend.sibtest_compute_beta(
        reference, focal, reference[:, 0], focal[:, 0], np.array([1])
    )
    variance = 0.4**2 * (0.8 * 0.2 / 9 + 0.3 * 0.7 / 29) + 0.6**2 * (
        0.5 * 0.5 / 39 + 0.3 * 0.7 / 19
    )
    assert beta == pytest.approx(0.32)
    assert se == pytest.approx(np.sqrt(variance))
    np.testing.assert_allclose(differences, [0.5, 0.2])
    np.testing.assert_allclose(counts, [40, 60])


def test_backend_sparse_matching_score_labels_do_not_allocate_by_label(backend):
    reference, focal = _groups_with_cells()
    beta, se, _, _ = backend.sibtest_compute_beta(
        reference,
        focal,
        reference[:, 0] * 2**40,
        focal[:, 0] * 2**40,
        np.array([1]),
    )
    ordinary_beta, ordinary_se, _, _ = backend.sibtest_compute_beta(
        reference, focal, reference[:, 0], focal[:, 0], np.array([1])
    )
    assert beta == ordinary_beta
    assert se == ordinary_se


def test_backend_all_items_matches_public_uncorrected_inference(backend):
    rng = np.random.default_rng(1515)
    data = rng.binomial(1, np.linspace(0.2, 0.8, 10), size=(800, 10))
    # Large group labels used to wrap silently when converted directly to i32.
    groups = np.repeat([2**40, 2**40 + 1], 400)
    anchors = np.array([0, 1, 2, 3, 4, 5, 6])
    expected = sibtest_items(data, groups, anchors, correction=False, p_adjust="none")
    beta, z, p_value = backend.sibtest_all_items(data, groups, anchors)
    for actual, name in ((beta, "beta"), (z, "z"), (p_value, "p_value")):
        np.testing.assert_allclose(actual, expected[name], atol=1e-12, equal_nan=True)


def test_backend_null_with_equal_stratum_effects_retains_variance(backend):
    reference = np.column_stack(
        (np.repeat([0, 1], 100), np.tile(np.repeat([0, 1], 50), 2))
    )
    beta, se, _, _ = backend.sibtest_compute_beta(
        reference, reference, reference[:, 0], reference[:, 0], np.array([1])
    )
    assert beta == 0.0
    assert se == pytest.approx(np.sqrt(0.5 / (99 * 2)))
    assert 2 * stats.norm.sf(abs(beta / se)) == 1.0


def test_backend_no_common_usable_stratum_is_unestimable(backend):
    reference = np.array([[0, 0], [0, 1]])
    focal = np.array([[1, 0], [1, 1]])
    beta, se, differences, counts = backend.sibtest_compute_beta(
        reference, focal, reference[:, 0], focal[:, 0], np.array([1])
    )
    assert np.isnan(beta) and np.isnan(se)
    assert differences.size == counts.size == 0


@pytest.mark.parametrize(
    "replacement",
    [
        {"ref_scores": np.array([0])},
        {"focal_scores": np.array([0.5, 1.0])},
        {"ref_scores": np.array([-1, 0])},
        {"suspect_items": np.array([-1])},
        {"suspect_items": np.array([1, 1])},
        {"focal_data": np.ones((2, 3), dtype=int)},
        {"ref_data": np.array([[0, 0], [0, -1]])},
    ],
)
def test_backend_beta_rejects_invalid_shapes_scores_and_selections(
    backend, replacement
):
    call = dict(
        ref_data=np.array([[0, 0], [0, 1]]),
        focal_data=np.array([[0, 0], [0, 1]]),
        ref_scores=np.array([0, 0]),
        focal_scores=np.array([0, 0]),
        suspect_items=np.array([1]),
    )
    call.update(replacement)
    with pytest.raises(ValueError):
        backend.sibtest_compute_beta(**call)


@pytest.mark.parametrize(
    "replacement",
    [
        {"groups": np.array([0, 0, 0, 0])},
        {"groups": np.array([0, 0, 0, 1])},
        {"groups": np.array([0, 1])},
        {"anchor_items": np.array([0, 0])},
        {"anchor_items": np.array([5])},
        {"data": np.ones((4, 1), dtype=int)},
        {"data": np.full((4, 3), np.nan)},
    ],
)
def test_backend_batch_rejects_invalid_data_and_groups(backend, replacement):
    call = dict(data=np.ones((4, 3), dtype=int), groups=np.array([0, 0, 1, 1]))
    call.update(replacement)
    with pytest.raises(ValueError):
        backend.sibtest_all_items(**call)


def test_backend_single_anchor_item_is_unestimable_for_itself(backend):
    reference, focal = _groups_with_cells()
    data = np.vstack((reference, focal))
    groups = np.repeat([0, 1], 50)
    beta, z, p_value = backend.sibtest_all_items(data, groups, np.array([0]))
    assert np.isnan(beta[0]) and np.isnan(z[0]) and np.isnan(p_value[0])
    assert beta[1] == pytest.approx(0.32)
    assert np.isfinite(p_value[1])
