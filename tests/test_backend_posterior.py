"""Posterior row searches agree with independent NumPy searches on both backends."""

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mirt
from mirt.backends.rust import posterior as backend
from mirt.backends.rust._helpers import RUST_AVAILABLE, mirt_rs
from mirt.results import ability_posterior as posterior_module


@pytest.fixture(params=["numpy", "rust"])
def search_backend(request: pytest.FixtureRequest) -> Iterator[str]:
    if request.param == "rust" and not RUST_AVAILABLE:
        pytest.skip("Rust extension is unavailable")
    previous = mirt.get_backend()
    mirt.set_backend(request.param)
    yield request.param
    mirt.set_backend(previous)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_row_search_matches_numpy(
    search_backend: str, side: str, shared: bool, layout: str
):
    rng = np.random.default_rng(284)
    cumulative = np.sort(rng.integers(0, 10, size=(17, 31)) / 10.0, axis=1)
    targets = rng.random((17, 9))
    if layout == "fortran":
        cumulative = np.asfortranarray(cumulative)
        targets = np.asfortranarray(targets)
    elif layout == "strided":
        cumulative = np.repeat(cumulative, 2, axis=1)[:, ::2]
        targets = targets[:, ::-1]
    if shared:
        targets = targets[0]
    cumulative.flags.writeable = False
    targets.flags.writeable = False
    target_rows = np.broadcast_to(targets, (17, targets.shape[-1]))
    expected = np.array(
        [
            np.searchsorted(row, values, side=side)
            for row, values in zip(cumulative, target_rows, strict=True)
        ]
    )

    actual = backend.row_searchsorted(cumulative, targets, side=side)

    assert_array_equal(actual, expected)
    assert actual.dtype == np.intp


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("shared", [False, True])
def test_exact_boundaries_zeros_and_nonfinite_values(
    search_backend: str, side: str, shared: bool
):
    cumulative = np.tile([-np.inf, 0.0, 0.5, 0.5, 1.0, np.inf, np.nan], (1000, 1))
    targets = np.array(
        [
            -np.inf,
            0.0,
            np.nextafter(0.5, 0.0),
            0.5,
            np.nextafter(0.5, 1.0),
            1.0,
            np.inf,
            np.nan,
        ]
    )
    expected = np.tile(np.searchsorted(cumulative[0], targets, side=side), (1000, 1))
    if not shared:
        targets = np.broadcast_to(targets, (1000, targets.size))

    assert_array_equal(
        backend.row_searchsorted(cumulative, targets, side=side), expected
    )


@pytest.mark.parametrize("rows,columns,queries", [(0, 3, 2), (2, 0, 3), (2, 3, 0)])
def test_empty_dimensions(search_backend: str, rows: int, columns: int, queries: int):
    result = backend.row_searchsorted(
        np.zeros((rows, columns)), np.ones((rows, queries))
    )
    assert result.shape == (rows, queries)
    assert_array_equal(result, np.full((rows, queries), columns))


@pytest.mark.parametrize(
    "cumulative,targets,side,message",
    [
        (np.zeros(3), np.zeros(2), "left", "cumulative"),
        (np.zeros((2, 3)), np.zeros((1, 2, 3)), "left", "targets"),
        (np.zeros((2, 3)), np.zeros((3, 2)), "left", "one row"),
        (np.zeros((2, 3)), np.zeros(2), "middle", "side"),
    ],
)
def test_invalid_shapes_and_side(
    search_backend: str, cumulative, targets, side, message
):
    with pytest.raises(ValueError, match=message):
        backend.row_searchsorted(cumulative, targets, side=side)


def test_numpy_backend_never_calls_native(monkeypatch: pytest.MonkeyPatch):
    class UnavailableNative:
        def __getattr__(self, name: str):
            pytest.fail(f"NumPy backend accessed {name}")

    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        monkeypatch.setattr(backend, "mirt_rs", UnavailableNative())
        assert_array_equal(backend.row_searchsorted([[0.0, 1.0]], [0.5]), [[1]])
        assert_array_equal(
            backend.shortest_mass_intervals([0.0, 1.0], [[0.8, 0.2]], 0.5),
            [[0.0], [0.0]],
        )
    finally:
        mirt.set_backend(previous)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is unavailable")
def test_native_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="one row"):
        mirt_rs.row_searchsorted(np.zeros((2, 3)), np.zeros((3, 2)), False)


@pytest.mark.parametrize("level", [0.05, 0.5, 0.95, 1e-20, np.nextafter(1.0, 0.0)])
def test_interval_backends_agree_with_ties_and_zero_weights(
    search_backend: str, level: float
):
    rng = np.random.default_rng(498)
    coordinates = np.arange(47, dtype=np.float64)
    weights = rng.integers(0, 10, size=(29, len(coordinates))).astype(np.float64)
    actual = backend.shortest_mass_intervals(coordinates, weights, level)
    previous = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        expected = backend.shortest_mass_intervals(coordinates, weights, level)
    finally:
        mirt.set_backend(previous)
    assert_array_equal(actual, expected)
    assert np.all(actual[0] <= actual[1])


@pytest.mark.parametrize(
    "coordinates,weights,level,message",
    [
        ([], np.empty((1, 0)), 0.5, "coordinates"),
        ([1.0, 0.0], [[0.5, 0.5]], 0.5, "coordinates"),
        ([0.0, np.inf], [[0.5, 0.5]], 0.5, "coordinates"),
        ([0.0, 1.0], [[1.0]], 0.5, "column"),
        ([0.0, 1.0], [[0.0, 0.0]], 0.5, "positive"),
        ([0.0, 1.0], [[-0.5, 1.5]], 0.5, "nonnegative"),
        ([0.0, 1.0], [[np.nan, 0.5]], 0.5, "finite"),
        ([0.0, 1.0], [[0.5, 0.5]], 1.0, "level"),
    ],
)
def test_interval_input_validation(
    search_backend: str, coordinates, weights, level, message
):
    with pytest.raises(ValueError, match=message):
        backend.shortest_mass_intervals(coordinates, weights, level)


@pytest.mark.parametrize("working_bytes", [1, 8192])
def test_joint_draws_preserve_the_seed_stream_across_batches(
    search_backend: str, working_bytes: int, monkeypatch: pytest.MonkeyPatch
):
    rng = np.random.default_rng(734)
    points = rng.normal(size=(49, 3))
    weights = rng.random((37, 49))
    weights[:, ::7] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    result = mirt.AbilityPosteriorResult(points, weights, np.zeros(37))
    draw_rng = np.random.default_rng(77)
    expected = np.stack(
        [
            points[np.searchsorted(row.cumsum(), draw_rng.random(41), side="right")].T
            for row in weights
        ]
    )
    monkeypatch.setattr(posterior_module, "_SUMMARY_WORKING_BYTES", working_bytes)
    assert_array_equal(result.sample(41, seed=77), expected)


def test_empty_interval_rows_and_single_grid_point(search_backend: str):
    lower, upper = backend.shortest_mass_intervals([2.0], np.empty((0, 1)), 0.5)
    assert lower.shape == upper.shape == (0,)
    assert_array_equal(
        backend.shortest_mass_intervals([2.0], [[3.0]], 0.5), [[2.0], [2.0]]
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension is unavailable")
def test_native_interval_boundary_checks():
    with pytest.raises(ValueError, match="one more column"):
        mirt_rs.shortest_mass_intervals(np.zeros(2), np.zeros((3, 2)), 0.5)
    with pytest.raises(ValueError, match="level"):
        mirt_rs.shortest_mass_intervals(np.zeros(2), np.zeros((3, 3)), np.nan)
