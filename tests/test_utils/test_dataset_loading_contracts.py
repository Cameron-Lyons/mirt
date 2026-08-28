"""Behavioral contracts for cached sample-dataset loading."""

from __future__ import annotations

import hashlib

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


@pytest.mark.parametrize(
    ("name", "expected_digest"),
    [
        (
            "verbal_aggression",
            "1eaa34548483ba5321093eb047643219a962f83d9276793bc193bdd4e1484a95",
        ),
        (
            "Attitude",
            "8549c0b7c51fb49c6460bf643c9933ee0a60b3598b7003ab8cebf7579de7da75",
        ),
        (
            "Bock1997",
            "ca83becdba38f232e9386debbb9bf59eb8da48830f19203abdd067ea15f0d01a",
        ),
        ("deAyala", "61114ced9031dbe6ab90bbc214358574aba2bc7c7b0a8db8ab99c6967b9fb5e6"),
    ],
)
def test_vectorized_generators_preserve_dataset_values(
    name: str,
    expected_digest: str,
) -> None:
    responses = load_dataset(name, copy=False)["data"].astype("<i8", copy=False)

    assert hashlib.sha256(responses.tobytes()).hexdigest() == expected_digest
