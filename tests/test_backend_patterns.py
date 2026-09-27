"""Shared response grouping preserves codes, ordering, and backend behavior."""

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mirt
from mirt.backends.rust import patterns as backend
from mirt.backends.rust._helpers import RUST_AVAILABLE, mirt_rs
from mirt.scoring._common import unique_response_patterns
from mirt.utils.collapse import collapse_patterns


@pytest.fixture(params=["numpy", "rust"])
def grouping_backend(request: pytest.FixtureRequest) -> Iterator[str]:
    if request.param == "rust" and not RUST_AVAILABLE:
        pytest.skip("Rust extension is unavailable")
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    yield request.param
    mirt.set_backend(previous)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.dtype(">i8")]
)
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed"])
def test_grouping_preserves_full_width_codes_and_row_order(
    grouping_backend: str, dtype: type | np.dtype, layout: str
) -> None:
    bounds = np.iinfo(dtype)
    responses = np.array(
        [[bounds.max, 0, 1], [bounds.min, 1, 0], [bounds.max, 0, 1], [2, 0, 1]],
        dtype=dtype,
    )
    if layout == "fortran":
        responses = np.asfortranarray(responses)
    elif layout == "strided":
        responses = np.repeat(responses, 2, axis=1)[:, ::2]
    elif layout == "reversed":
        responses = responses[:, ::-1]
    original = responses.copy()
    responses.flags.writeable = False

    first, inverse, counts = backend.response_pattern_indices(responses)

    assert_array_equal(first, [0, 1, 3])
    assert_array_equal(inverse, [0, 1, 0, 2])
    assert_array_equal(counts, [2, 1, 1])
    assert_array_equal(responses[first][inverse], original)
    assert_array_equal(responses, original)
    assert all(values.dtype == np.intp for values in (first, inverse, counts))


@pytest.mark.parametrize("shape", [(0, 3), (0, 0), (3, 0), (1, 1)])
def test_empty_dimensions_and_single_row(grouping_backend: str, shape: tuple) -> None:
    responses = np.zeros(shape, dtype=np.int_)
    first, inverse, counts = backend.response_pattern_indices(responses)
    assert_array_equal(responses[first][inverse], responses)
    assert_array_equal(first, [] if shape[0] == 0 else [0])
    assert_array_equal(counts, [] if shape[0] == 0 else [shape[0]])


@pytest.mark.parametrize("n_patterns", [1, 17, 1000])
def test_random_patterns_match_numpy_uniqueness(
    grouping_backend: str, n_patterns: int
) -> None:
    rng = np.random.default_rng(243)
    source = rng.integers(-1, 5, size=(n_patterns, 7))
    responses = source[rng.integers(0, n_patterns, size=1000)]
    expected, expected_counts = np.unique(responses, axis=0, return_counts=True)

    first, inverse, counts = backend.response_pattern_indices(responses)
    actual = responses[first]
    sort_order = np.lexsort(actual.T[::-1])
    assert_array_equal(actual[sort_order], expected)
    assert_array_equal(counts[sort_order], expected_counts)
    assert_array_equal(actual[inverse], responses)
    assert np.all(np.diff(first) > 0)
    for index, row in enumerate(actual):
        assert first[index] == np.flatnonzero(np.all(responses == row, axis=1))[0]


def test_scoring_and_public_collapsing_share_normalized_patterns(
    grouping_backend: str,
) -> None:
    responses = np.array([[1, np.nan], [0, 2], [1, -999], [0, 2]])
    collapsed = collapse_patterns(responses, missing_code=-7)
    patterns, inverse = unique_response_patterns(collapsed.patterns[collapsed.indices])

    assert_array_equal(patterns, [[1, -7], [0, 2]])
    assert_array_equal(inverse, [0, 1, 0, 1])
    assert_array_equal(collapsed.frequencies, [2, 2])
    assert collapsed.frequencies.dtype == np.dtype(np.int_)
    assert collapsed.indices.dtype == np.dtype(np.int_)
    assert not np.shares_memory(patterns, responses)


@pytest.mark.parametrize("responses", [np.ones(3, dtype=int), np.ones((2, 3))])
def test_invalid_grouping_inputs_raise(grouping_backend: str, responses: np.ndarray):
    with pytest.raises(ValueError, match="two-dimensional signed integer"):
        backend.response_pattern_indices(responses)


def test_numpy_selection_does_not_call_native_code(monkeypatch: pytest.MonkeyPatch):
    class UnavailableNative:
        def __getattr__(self, name: str):
            pytest.fail(f"NumPy backend accessed native function {name}")

    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        monkeypatch.setattr(backend, "mirt_rs", UnavailableNative())
        collapsed = collapse_patterns([[1, 0], [1, 0]])
        assert_array_equal(collapsed.frequencies, [2])
    finally:
        mirt.set_backend(previous)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is unavailable")
@pytest.mark.parametrize("layout", ["strided", "fortran"])
def test_native_boundary_rejects_non_c_contiguous_input(layout: str):
    responses = np.array([[0, 1, 2, 3], [1, 0, 2, 3]], dtype=np.int64)
    responses = (
        responses[:, ::2] if layout == "strided" else np.asfortranarray(responses)
    )
    with pytest.raises(ValueError, match="contiguous"):
        mirt_rs.response_pattern_indices(responses)
