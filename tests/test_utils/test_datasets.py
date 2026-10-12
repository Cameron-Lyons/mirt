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

    def test_lsat6_matches_the_published_table(self):
        """LSAT6 reproduces Bock & Lieberman's counts and R mirt's 2PL fit."""
        from mirt import fit_mirt

        responses = load_dataset("LSAT6")["data"]
        assert responses.shape == (1000, 5)
        np.testing.assert_allclose(
            responses.mean(axis=0), [0.924, 0.709, 0.553, 0.763, 0.870]
        )
        patterns, counts = np.unique(responses, axis=0, return_counts=True)
        # expand.table(LSAT6) in R mirt: 30 observed patterns.
        expected = {
            "00000": 3, "00001": 6, "00010": 2, "00011": 11, "00100": 1,
            "00101": 1, "00110": 3, "00111": 4, "01000": 1, "01001": 8,
            "01011": 16, "01101": 3, "01110": 2, "01111": 15, "10000": 10,
            "10001": 29, "10010": 14, "10011": 81, "10100": 3, "10101": 28,
            "10110": 15, "10111": 80, "11000": 16, "11001": 56, "11010": 21,
            "11011": 173, "11100": 11, "11101": 61, "11110": 28, "11111": 298,
        }  # fmt: skip
        observed = {
            "".join(str(value) for value in pattern): int(count)
            for pattern, count in zip(patterns, counts, strict=True)
        }
        assert observed == expected

        result = fit_mirt(responses, "2PL", n_quadpts=61, tol=1e-8, max_iter=2000)
        slopes = result.model.parameters["discrimination"].ravel()
        intercepts = -slopes * result.model.parameters["difficulty"]
        # mirt::mirt(expand.table(LSAT6), 1) in R: logLik -2466.653.
        np.testing.assert_allclose(
            slopes, [0.825, 0.723, 0.890, 0.689, 0.658], atol=3e-3
        )
        np.testing.assert_allclose(
            intercepts, [2.773, 0.990, 0.249, 1.285, 2.054], atol=3e-3
        )
        assert result.log_likelihood == pytest.approx(-2466.654, abs=1e-2)

    def test_simulated_stand_ins_are_labelled(self):
        """Datasets named after R mirt data say they are simulated stand-ins."""
        from mirt import describe_dataset

        published = {"LSAT6", "LSAT7"}
        r_mirt_names = {"SAT12", "Science", "Bock1997", "deAyala", "SLF"}
        for name in list_datasets():
            metadata = describe_dataset(name)
            data = load_dataset(name, copy=False)["data"]
            assert (metadata["n_persons"], metadata["n_items"]) == data.shape
            if name in published:
                assert metadata["simulated"] is False
                continue
            assert metadata["simulated"] is True
            assert metadata["description"].startswith("Simulated stand-in")
            if name in r_mirt_names:
                assert "not the R mirt data" in metadata["description"]
                assert str(data.shape[0]) in metadata["description"]

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
