"""Tests for sample datasets."""

import numpy as np
import pytest

from mirt import list_datasets, load_dataset


class TestDatasets:
    """Tests for dataset loading."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "LSAT6" in datasets or "lsat6" in [d.lower() for d in datasets]

    def test_load_lsat6(self):
        """Test loading LSAT6 dataset."""
        data = load_dataset("LSAT6")

        assert "data" in data
        assert data["data"].ndim == 2

        n_persons, n_items = data["data"].shape
        assert n_items == 5

    def test_load_lsat7(self):
        """Test loading LSAT7 dataset."""
        data = load_dataset("LSAT7")

        assert "data" in data
        n_persons, n_items = data["data"].shape
        assert n_items == 5

    def test_lsat7_matches_the_published_table(self):
        """LSAT7 reproduces Bock & Lieberman's counts and R mirt's 2PL fit."""
        from mirt import fit_mirt

        responses = load_dataset("LSAT7")["data"]
        assert responses.shape == (1000, 5)
        np.testing.assert_allclose(
            responses.mean(axis=0), [0.828, 0.658, 0.772, 0.606, 0.843]
        )

        result = fit_mirt(responses, "2PL", n_quadpts=61, tol=1e-8, max_iter=2000)
        slopes = result.model.parameters["discrimination"].ravel()
        intercepts = -slopes * result.model.parameters["difficulty"]
        # mirt::mirt(LSAT7, 1) in R.
        np.testing.assert_allclose(
            slopes, [0.988, 1.081, 1.706, 0.765, 0.736], atol=3e-3
        )
        np.testing.assert_allclose(
            intercepts, [1.856, 0.808, 1.804, 0.486, 1.855], atol=3e-3
        )
        assert result.log_likelihood == pytest.approx(-2658.805, abs=1e-2)

    def test_load_sat12(self):
        """Test loading SAT12 dataset."""
        data = load_dataset("SAT12")

        assert "data" in data
        n_persons, n_items = data["data"].shape
        assert n_items == 12

    def test_load_science(self):
        """Test loading Science dataset."""
        data = load_dataset("Science")

        assert "data" in data

    def test_load_verbal_aggression(self):
        """Test loading Verbal Aggression dataset."""
        data = load_dataset("verbal_aggression")

        assert "data" in data
        responses = data["data"]
        assert set(responses.flatten()).issubset({0, 1, 2, -1})

    def test_load_fraction_subtraction(self):
        """Test loading Fraction Subtraction dataset."""
        data = load_dataset("fraction_subtraction")

        assert "data" in data
        responses = data["data"]
        valid = responses[responses >= 0]
        assert set(valid.flatten()).issubset({0, 1})

    def test_invalid_dataset(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises((ValueError, KeyError)):
            load_dataset("nonexistent_dataset")

    def test_dataset_metadata(self):
        """Test that datasets include metadata."""
        data = load_dataset("LSAT6")

        assert "data" in data

        if "item_names" in data:
            assert len(data["item_names"]) == data["data"].shape[1]

    def test_responses_dtype(self):
        """Test that responses are integer type."""
        data = load_dataset("LSAT6")
        responses = data["data"]

        assert np.issubdtype(responses.dtype, np.integer)

    def test_case_insensitive(self):
        """Test case-insensitive dataset names."""
        data1 = load_dataset("LSAT6")
        data2 = load_dataset("lsat6")

        np.testing.assert_array_equal(data1["data"], data2["data"])
