"""Behavioral contracts for cached sample-dataset loading."""

from __future__ import annotations

import numpy as np
import pytest

from mirt import describe_dataset, load_dataset


def test_default_loads_are_independent_and_writable() -> None:
    """The safe default must not expose shared mutable response storage."""
    first = load_dataset("SAT12")
    second = load_dataset("SAT12")

    assert first["data"].flags.writeable
    assert not np.shares_memory(first["data"], second["data"])

    original = int(second["data"][0, 0])
    first["data"][0, 0] = 1 - original
    assert second["data"][0, 0] == original


def test_zero_copy_loads_share_read_only_array_storage() -> None:
    """The allocation-saving mode shares arrays without exposing mutation."""
    first = load_dataset("Science", copy=False)
    second = load_dataset("science", copy=False)

    assert np.shares_memory(first["data"], second["data"])
    assert not first["data"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first["data"][0, 0] = 1
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        first["data"].flags.writeable = True


def test_nested_metadata_containers_remain_independent() -> None:
    """Zero-copy applies only to arrays, not mutable metadata containers."""
    first = load_dataset("SLF", copy=False)
    second = load_dataset("SLF", copy=False)

    first["item_names"][0] = "changed"
    assert second["item_names"][0] == "SLF1"


def test_describe_dataset_omits_array_payloads() -> None:
    metadata = describe_dataset("  verbal_AGGRESSION ")

    assert metadata["name"] == "verbal_aggression"
    assert metadata["n_persons"] == 316
    assert metadata["n_items"] == 24
    assert "data" not in metadata
    assert all(not isinstance(value, np.ndarray) for value in metadata.values())


def test_unknown_dataset_suggests_a_close_match() -> None:
    with pytest.raises(ValueError, match="Did you mean 'LSAT6'"):
        load_dataset("lsat-6")


def test_non_string_dataset_name_has_clear_error() -> None:
    with pytest.raises(TypeError, match="name must be a string"):
        load_dataset(6)  # type: ignore[arg-type]
