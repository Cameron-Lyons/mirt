"""Tests for CAT result structures."""

import numpy as np
import pytest

from mirt.cat.results import CATResult, CATState, MCATResult, MCATState
from mirt.exceptions import MirtValidationError


class TestCATState:
    """Tests for CATState dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        state = CATState(
            theta=0.5,
            standard_error=0.3,
            items_administered=[0, 1, 2],
            responses=[1, 0, 1],
            n_items=3,
            is_complete=False,
            next_item=3,
        )

        assert state.theta == 0.5
        assert state.standard_error == 0.3
        assert state.items_administered == [0, 1, 2]
        assert state.responses == [1, 0, 1]
        assert state.n_items == 3
        assert state.is_complete is False
        assert state.next_item == 3

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        state = CATState(theta=0.0, standard_error=1.0)

        assert state.items_administered == []
        assert state.responses == []
        assert state.n_items == 0
        assert state.is_complete is False
        assert state.next_item is None

    def test_repr(self):
        """Test __repr__ method."""
        state = CATState(
            theta=0.5,
            standard_error=0.3,
            n_items=5,
            is_complete=True,
        )

        repr_str = repr(state)

        assert "CATState" in repr_str
        assert "0.5" in repr_str or "0.500" in repr_str
        assert "0.3" in repr_str or "0.300" in repr_str
        assert "5" in repr_str


class TestCATResult:
    """Tests for CATResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample CAT result."""
        return CATResult(
            theta=0.75,
            standard_error=0.28,
            items_administered=[0, 3, 5, 2, 7],
            responses=np.array([1, 1, 0, 1, 0]),
            n_items_administered=5,
            stopping_reason="SE threshold reached",
            theta_history=[0.0, 0.5, 0.6, 0.7, 0.75],
            se_history=[1.0, 0.5, 0.4, 0.32, 0.28],
            item_info_history=[1.0, 0.8, 0.9, 0.7, 0.6],
        )

    def test_initialization(self, sample_result):
        """Test basic initialization."""
        assert sample_result.theta == 0.75
        assert sample_result.standard_error == 0.28
        assert sample_result.items_administered == [0, 3, 5, 2, 7]
        assert len(sample_result.responses) == 5
        assert sample_result.n_items_administered == 5
        assert sample_result.stopping_reason == "SE threshold reached"

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        result = CATResult(
            theta=0.0,
            standard_error=0.5,
            items_administered=[0],
            responses=np.array([1]),
            n_items_administered=1,
            stopping_reason="Max items",
        )

        assert result.theta_history == []
        assert result.se_history == []
        assert result.item_info_history == []

    def test_summary(self, sample_result):
        """Test summary method."""
        summary = sample_result.summary()

        assert "CAT Result" in summary
        assert "0.75" in summary or "theta" in summary.lower()
        assert "0.28" in summary or "error" in summary.lower()
        assert "5" in summary

    def test_to_array(self, sample_result):
        """Test to_array method."""
        arr = sample_result.to_array()

        assert arr.shape == (5, 4)
        assert arr[0, 0] == 0
        assert arr[1, 0] == 3
        assert arr[2, 0] == 5

    def test_to_array_without_history(self):
        """Test to_array without history."""
        result = CATResult(
            theta=0.0,
            standard_error=0.5,
            items_administered=[0, 1],
            responses=np.array([1, 0]),
            n_items_administered=2,
            stopping_reason="Max items",
        )

        arr = result.to_array()

        assert arr.shape == (2, 4)

    def test_repr(self, sample_result):
        """Test __repr__ method."""
        repr_str = repr(sample_result)

        assert "CATResult" in repr_str
        assert "theta" in repr_str.lower()
        assert "5" in repr_str

    def test_dict_round_trip(self, sample_result):
        """Portable dictionaries reconstruct the full administration."""
        payload = sample_result.to_dict()

        restored = CATResult.from_dict(payload)

        assert restored.to_dict() == payload
        assert isinstance(restored.responses, np.ndarray)
        payload["items_administered"][0] = 99
        assert restored.items_administered[0] == 0

    def test_json_round_trip(self, sample_result):
        """JSON text and bytes reconstruct the full administration."""
        text = sample_result.to_json(indent=2)

        restored_from_text = CATResult.from_json(text)
        restored_from_bytes = CATResult.from_json(text.encode())

        assert restored_from_text.to_dict() == sample_result.to_dict()
        assert restored_from_bytes.to_dict() == sample_result.to_dict()

    def test_from_dict_rejects_unknown_and_missing_fields(self, sample_result):
        """Malformed mappings do not silently lose result data."""
        unknown = sample_result.to_dict()
        unknown["stopping_reazon"] = unknown["stopping_reason"]
        with pytest.raises(MirtValidationError, match="unknown fields"):
            CATResult.from_dict(unknown)

        missing = sample_result.to_dict()
        del missing["responses"]
        with pytest.raises(MirtValidationError, match="missing required fields"):
            CATResult.from_dict(missing)

    def test_from_json_rejects_invalid_values(self):
        """Invalid JSON inputs use the package validation exception."""
        with pytest.raises(MirtValidationError, match="valid JSON object"):
            CATResult.from_json("{not valid")
        with pytest.raises(MirtValidationError, match="string or bytes"):
            CATResult.from_json(123)  # type: ignore[arg-type]

    def test_constructor_normalizes_and_copies_arrays(self):
        """Caller-owned arrays cannot mutate a constructed result."""
        responses = np.array([1, 0])
        theta_history = np.array([0.1, 0.2])
        result = CATResult(
            theta=np.float64(0.2),
            standard_error=np.float64(0.4),
            items_administered=[2, 4],
            responses=responses,
            n_items_administered=np.int64(2),
            stopping_reason="  complete  ",
            theta_history=theta_history,
        )

        responses[0] = 0
        theta_history[0] = 99.0
        assert result.responses.tolist() == [1, 0]
        assert result.theta_history == [0.1, 0.2]
        assert result.stopping_reason == "complete"

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"responses": np.array([1])}, "one value per administered item"),
            ({"n_items_administered": 4}, "must match"),
            ({"standard_error": -0.1}, "cannot be negative"),
            ({"theta_history": [0.1] * 6}, "cannot be longer"),
        ],
    )
    def test_constructor_rejects_inconsistent_results(
        self,
        sample_result,
        changes,
        match,
    ):
        """Counts, uncertainty, and histories must describe one administration."""
        values = sample_result.to_dict()
        values.update(changes)

        with pytest.raises(MirtValidationError, match=match):
            CATResult.from_dict(values)


class TestMCATState:
    """Tests for MCATState dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        theta = np.array([0.5, -0.3])
        cov = np.array([[0.25, 0.05], [0.05, 0.30]])
        se = np.array([0.5, 0.55])

        state = MCATState(
            theta=theta,
            covariance=cov,
            standard_error=se,
            items_administered=[0, 1, 2],
            responses=[1, 0, 1],
            n_items=3,
            is_complete=False,
            next_item=3,
        )

        assert np.array_equal(state.theta, theta)
        assert np.array_equal(state.covariance, cov)
        assert np.array_equal(state.standard_error, se)
        assert state.n_items == 3

    def test_n_factors_property(self):
        """Test n_factors property."""
        theta = np.array([0.5, -0.3, 0.1])
        cov = np.eye(3)
        se = np.array([0.5, 0.5, 0.5])

        state = MCATState(theta=theta, covariance=cov, standard_error=se)

        assert state.n_factors == 3

    def test_trace_covariance_property(self):
        """Test trace_covariance property."""
        theta = np.array([0.0, 0.0])
        cov = np.array([[0.25, 0.0], [0.0, 0.36]])
        se = np.array([0.5, 0.6])

        state = MCATState(theta=theta, covariance=cov, standard_error=se)

        assert state.trace_covariance == pytest.approx(0.61)

    def test_det_covariance_property(self):
        """Test det_covariance property."""
        theta = np.array([0.0, 0.0])
        cov = np.array([[0.25, 0.0], [0.0, 0.36]])
        se = np.array([0.5, 0.6])

        state = MCATState(theta=theta, covariance=cov, standard_error=se)

        assert state.det_covariance == pytest.approx(0.09)

    def test_repr(self):
        """Test __repr__ method."""
        state = MCATState(
            theta=np.array([0.5, -0.3]),
            covariance=np.eye(2),
            standard_error=np.array([0.5, 0.5]),
            n_items=5,
            is_complete=True,
        )

        repr_str = repr(state)

        assert "MCATState" in repr_str
        assert "0.5" in repr_str or "0.500" in repr_str


class TestMCATResult:
    """Tests for MCATResult dataclass."""

    @pytest.fixture
    def sample_mcat_result(self):
        """Create a sample MCAT result."""
        return MCATResult(
            theta=np.array([0.75, -0.5]),
            covariance=np.array([[0.16, 0.02], [0.02, 0.25]]),
            standard_error=np.array([0.4, 0.5]),
            items_administered=[0, 3, 5, 2, 7],
            responses=np.array([1, 1, 0, 1, 0]),
            n_items_administered=5,
            stopping_reason="Trace threshold reached",
            theta_history=[np.array([0.0, 0.0]), np.array([0.5, -0.3])],
            se_history=[np.array([1.0, 1.0]), np.array([0.6, 0.7])],
            covariance_history=[np.eye(2), np.eye(2) * 0.5],
            item_info_history=[1.0, 0.8],
        )

    def test_initialization(self, sample_mcat_result):
        """Test basic initialization."""
        assert sample_mcat_result.theta.shape == (2,)
        assert sample_mcat_result.covariance.shape == (2, 2)
        assert sample_mcat_result.n_items_administered == 5

    def test_n_factors_property(self, sample_mcat_result):
        """Test n_factors property."""
        assert sample_mcat_result.n_factors == 2

    def test_summary(self, sample_mcat_result):
        """Test summary method."""
        summary = sample_mcat_result.summary()

        assert "MCAT" in summary
        assert "Dimension" in summary
        assert "5" in summary

    def test_repr(self, sample_mcat_result):
        """Test __repr__ method."""
        repr_str = repr(sample_mcat_result)

        assert "MCATResult" in repr_str
        assert "theta" in repr_str.lower()

    def test_dict_round_trip(self, sample_mcat_result):
        """Portable dictionaries preserve multidimensional histories."""
        payload = sample_mcat_result.to_dict()

        restored = MCATResult.from_dict(payload)

        assert restored.to_dict() == payload
        assert all(isinstance(value, np.ndarray) for value in restored.theta_history)
        payload["covariance"][0][0] = 99.0
        assert restored.covariance[0, 0] == pytest.approx(0.16)

    def test_json_round_trip(self, sample_mcat_result):
        """JSON preserves multidimensional estimates and covariance histories."""
        restored = MCATResult.from_json(sample_mcat_result.to_json(indent=2))

        assert restored.to_dict() == sample_mcat_result.to_dict()
        np.testing.assert_array_equal(restored.theta, sample_mcat_result.theta)
        np.testing.assert_array_equal(
            restored.covariance_history[1],
            sample_mcat_result.covariance_history[1],
        )

    def test_from_dict_validates_factor_metadata(self, sample_mcat_result):
        """Derived dimensional metadata must agree with the arrays."""
        payload = sample_mcat_result.to_dict()
        payload["n_factors"] = 3

        with pytest.raises(MirtValidationError, match="n_factors does not match"):
            MCATResult.from_dict(payload)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"theta": []}, "at least one factor"),
            ({"covariance": [[1.0]]}, "one row and column per factor"),
            (
                {"covariance": [[1.0, 0.4], [0.1, 1.0]]},
                "must be symmetric",
            ),
            ({"standard_error": [0.4]}, "one value per factor"),
            ({"standard_error": [0.4, -0.1]}, "negative values"),
            ({"theta_history": [[0.1]]}, "final result shape"),
        ],
    )
    def test_constructor_rejects_inconsistent_dimensions(
        self,
        sample_mcat_result,
        changes,
        match,
    ):
        """Every multidimensional array must follow the final factor shape."""
        values = sample_mcat_result.to_dict()
        values.update(changes)

        with pytest.raises(MirtValidationError, match=match):
            MCATResult.from_dict(values)


class TestCATResultToDataFrame:
    """Tests for to_dataframe method (requires pandas/polars)."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample CAT result."""
        return CATResult(
            theta=0.75,
            standard_error=0.28,
            items_administered=[0, 3, 5],
            responses=np.array([1, 1, 0]),
            n_items_administered=3,
            stopping_reason="SE threshold reached",
            theta_history=[0.0, 0.5, 0.75],
            se_history=[1.0, 0.5, 0.28],
            item_info_history=[1.0, 0.8, 0.9],
        )

    def test_to_dataframe(self, sample_result):
        """Test to_dataframe conversion."""
        try:
            df = sample_result.to_dataframe()
            assert len(df) == 3
            assert "step" in df.columns
            assert "item" in df.columns
            assert "response" in df.columns
        except ImportError:
            pytest.skip("pandas/polars not available")


class TestCATResultPlotConvergence:
    """Tests for plot_convergence method."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample CAT result."""
        return CATResult(
            theta=0.75,
            standard_error=0.28,
            items_administered=[0, 3, 5],
            responses=np.array([1, 1, 0]),
            n_items_administered=3,
            stopping_reason="SE threshold reached",
            theta_history=[0.0, 0.5, 0.75],
            se_history=[1.0, 0.5, 0.28],
            item_info_history=[1.0, 0.8, 0.9],
        )

    @pytest.mark.skip(reason="Requires display for matplotlib")
    def test_plot_convergence(self, sample_result):
        """Test plot_convergence method."""
        try:
            fig = sample_result.plot_convergence()
            assert fig is not None
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")
