"""Tests for Differential Item Functioning (DIF) analysis."""

from types import SimpleNamespace

import numpy as np
import pytest

from mirt.diagnostics.dif import _ets_classify, compute_dif, flag_dif_items
from mirt.models.dichotomous import TwoParameterLogistic

try:
    import pandas as pandas

    HAS_DATAFRAME = True
except ImportError:
    try:
        import polars as polars

        HAS_DATAFRAME = True
    except ImportError:
        HAS_DATAFRAME = False


class TestDIF:
    """Tests for DIF detection functions."""

    def test_basic_dif_likelihood_ratio(self, rng):
        """Test basic DIF analysis with likelihood ratio method."""
        n_per_group = 100
        n_items = 5

        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        difficulty = rng.normal(0, 1, n_items)

        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty)))
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty)))

        responses1 = (rng.random((n_per_group, n_items)) < probs1).astype(int)
        responses2 = (rng.random((n_per_group, n_items)) < probs2).astype(int)

        data = np.vstack([responses1, responses2])
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
            p_adjust="holm",
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "p_value_adjusted" in result
        assert "effect_size" in result
        assert "classification" in result
        assert "adjustment" in result

        assert len(result["statistic"]) == n_items
        assert len(result["p_value"]) == n_items
        assert np.all(result["p_value"] >= 0) and np.all(result["p_value"] <= 1)
        assert np.all(result["p_value_adjusted"] >= result["p_value"])
        assert np.all(result["adjustment"] == "holm")
        expected_classes = np.where(
            (result["p_value_adjusted"] > 0.05)
            | (np.abs(result["effect_size"]) < 0.426),
            "A",
            np.where(np.abs(result["effect_size"]) < 0.638, "B", "C"),
        )
        np.testing.assert_array_equal(result["classification"], expected_classes)

    def test_dif_wald_method(self, rng):
        """Test DIF analysis with Wald method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="wald", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "p_value" in result
        assert len(result["statistic"]) == n_items

    def test_dif_lord_method(self, rng):
        """Test DIF analysis with Lord's chi-square method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="lord", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "p_value" in result

    def test_dif_raju_method(self, rng):
        """Test DIF analysis with Raju's area method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="raju", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "effect_size" in result

    def test_dif_detects_biased_item(self, rng):
        """Test that DIF detects an item with large difficulty difference."""
        n_per_group = 150
        n_items = 5

        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        difficulty = np.zeros(n_items)
        difficulty_group2 = difficulty.copy()
        difficulty_group2[0] = 2.0

        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty)))
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty_group2)))

        responses1 = (rng.random((n_per_group, n_items)) < probs1).astype(int)
        responses2 = (rng.random((n_per_group, n_items)) < probs2).astype(int)

        data = np.vstack([responses1, responses2])
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        assert result["effect_size"][0] > np.mean(result["effect_size"][1:])

    def test_ets_classification(self, rng):
        """Test ETS A/B/C classification."""
        n_per_group = 100
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        valid_classes = {"A", "B", "C"}
        for c in result["classification"]:
            assert c in valid_classes

    def test_flag_dif_items(self, rng):
        """Test flag_dif_items helper function."""
        dif_results = {
            "statistic": np.array([10.0, 2.0, 15.0, 1.0]),
            "p_value": np.array([0.001, 0.15, 0.0001, 0.3]),
            "effect_size": np.array([0.8, 0.2, 1.0, 0.1]),
            "classification": np.array(["C", "A", "C", "A"]),
        }

        flags = flag_dif_items(dif_results)
        assert flags.dtype == bool
        assert len(flags) == 4

        assert flags[0]
        assert flags[2]
        assert not flags[1]
        assert not flags[3]

    def test_flag_dif_by_classification(self, rng):
        """Test flagging by ETS classification."""
        dif_results = {
            "statistic": np.array([10.0, 5.0, 15.0, 1.0]),
            "p_value": np.array([0.001, 0.02, 0.0001, 0.3]),
            "effect_size": np.array([0.8, 0.5, 1.0, 0.1]),
            "classification": np.array(["C", "B", "C", "A"]),
        }

        flags_c = flag_dif_items(dif_results, classification="C")
        assert np.sum(flags_c) <= np.sum(dif_results["classification"] == "C")

        flags_b = flag_dif_items(dif_results, classification="B")
        assert np.sum(flags_b) <= np.sum(
            (dif_results["classification"] == "B")
            | (dif_results["classification"] == "C")
        )

    def test_flag_dif_applies_multiple_testing_adjustment(self):
        dif_results = {
            "p_value": np.array([0.01, 0.02, 0.04, 0.20]),
            "effect_size": np.full(4, 0.8),
            "classification": np.full(4, "C"),
        }

        unadjusted = flag_dif_items(dif_results)
        adjusted = flag_dif_items(dif_results, p_adjust="bonferroni")

        np.testing.assert_array_equal(unadjusted, [True, True, True, False])
        np.testing.assert_array_equal(adjusted, [True, False, False, False])

    def test_flag_dif_classifies_signed_effects_by_magnitude(self):
        dif_results = {
            "p_value": np.array([0.001, 0.001]),
            "effect_size": np.array([-0.8, -0.5]),
            "classification": np.array(["C", "B"]),
        }

        np.testing.assert_array_equal(
            flag_dif_items(dif_results, classification="B"),
            [True, True],
        )
        np.testing.assert_array_equal(
            flag_dif_items(dif_results, classification="C"),
            [True, False],
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": "invalid"}, "alpha"),
            ({"min_effect_size": -0.1}, "min_effect_size"),
            ({"classification": "A"}, "classification"),
            ({"p_adjust": "unknown"}, "p_adjust"),
        ],
    )
    def test_flag_dif_validates_options(self, kwargs, message):
        dif_results = {
            "p_value": np.array([0.01, 0.20]),
            "effect_size": np.array([0.8, 0.2]),
            "classification": np.array(["C", "A"]),
        }

        with pytest.raises(ValueError, match=message):
            flag_dif_items(dif_results, **kwargs)

    def test_compute_dif_rejects_adjustment_before_fitting(self):
        with pytest.raises(ValueError, match="p_adjust"):
            compute_dif(
                np.zeros((1, 1), dtype=int),
                np.zeros(1, dtype=int),
                p_adjust="unknown",  # type: ignore[arg-type]
            )

    def test_requires_two_groups(self, rng):
        """Test that DIF requires exactly 2 groups."""
        n_persons = 100
        n_items = 5
        data = rng.integers(0, 2, size=(n_persons, n_items))

        groups = np.zeros(n_persons)
        with pytest.raises(ValueError, match="Expected 2 groups"):
            compute_dif(data, groups)

        groups = np.array([0] * 33 + [1] * 33 + [2] * 34)
        with pytest.raises(ValueError, match="Expected 2 groups"):
            compute_dif(data, groups)

    def test_string_group_labels(self, rng):
        """Test DIF with string group labels."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array(["male"] * n_per_group + ["female"] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="wald", n_quadpts=11, max_iter=30
        )

        assert len(result["statistic"]) == n_items

    def test_focal_group_specification(self, rng):
        """Test specifying focal group."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="wald",
            focal_group=0,
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_invalid_focal_group(self, rng):
        """Test error for invalid focal group."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        with pytest.raises(ValueError, match="not found in groups"):
            compute_dif(data, groups, focal_group=99)

    def test_1pl_model(self, rng):
        """Test DIF with 1PL model."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="1PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_polytomous_grm(self, rng):
        """Test DIF with polytomous GRM model."""
        n_per_group = 60
        n_items = 4
        n_categories = 4

        data = rng.integers(0, n_categories, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="GRM",
            n_categories=n_categories,
            method="raju",
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_invalid_method(self, rng):
        """Test error for invalid DIF method."""
        n_per_group = 30
        n_items = 3

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        with pytest.raises(ValueError, match="Unknown DIF method"):
            compute_dif(data, groups, method="invalid")


class TestDIFIntegration:
    """Integration tests for mirt.dif() function."""

    def test_dif_function_import(self):
        """Test that dif function is importable from mirt."""
        import mirt

        assert hasattr(mirt, "dif")

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_dif_returns_dataframe(self, rng):
        """Test that mirt.dif() returns a DataFrame."""
        import mirt

        n_per_group = 50
        n_items = 4

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = mirt.dif(
            data,
            groups,
            model="2PL",
            method="wald",
            n_quadpts=11,
            max_iter=30,
            p_adjust="fdr_bh",
        )

        assert hasattr(result, "columns") or hasattr(result, "schema")
        columns = (
            result.columns if hasattr(result, "columns") else result.schema.names()
        )
        assert "p_value_adjusted" in columns
        assert "adjustment" in columns


def _simulate_two_groups(
    seed: int,
    *,
    n_per_group: int,
    n_items: int,
    shift: float = 1.2,
    impact: float = -0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """2PL responses with uniform DIF on item 0 and a lower focal mean."""
    rng = np.random.default_rng(seed)
    discrimination = np.linspace(0.9, 1.8, n_items)
    difficulty = np.linspace(-1.2, 1.0, n_items)
    focal_difficulty = difficulty.copy()
    focal_difficulty[0] += shift

    def responses(theta: np.ndarray, locations: np.ndarray) -> np.ndarray:
        logits = discrimination * (theta[:, None] - locations)
        return (rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

    data = np.vstack(
        [
            responses(rng.normal(0.0, 1.0, n_per_group), difficulty),
            responses(rng.normal(impact, 1.0, n_per_group), focal_difficulty),
        ]
    )
    return data, np.repeat([0, 1], n_per_group)


def _impact_calibrations(dif_shift: float = 0.0):
    """Reference 2PL and the separate focal calibration under impact.

    Focal abilities follow N(-1, 1.2**2); calibrating the focal group alone
    standardizes them, giving ``a * 1.2`` and ``(b + 1) / 1.2``. Item 0 is
    ``dif_shift`` harder for the focal group on the common scale.
    """
    discrimination = np.array([0.8, 1.0, 1.3, 1.6, 2.0, 1.1])
    difficulty = np.array([-1.5, -0.8, -0.2, 0.3, 0.9, 1.4])
    focal_difficulty = difficulty.copy()
    focal_difficulty[0] += dif_shift
    reference = TwoParameterLogistic(n_items=6)
    reference.set_parameters(discrimination=discrimination, difficulty=difficulty)
    focal = TwoParameterLogistic(n_items=6)
    focal.set_parameters(
        discrimination=1.2 * discrimination,
        difficulty=(focal_difficulty + 1.0) / 1.2,
    )
    return reference, focal


def _install_calibrations(monkeypatch, reference, focal, errors=(None, None)):
    """Replace the separate group calibrations used by Wald and Raju DIF."""
    options: list[dict] = []

    def fake_fit(ref_data, focal_data, model="2PL", **kwargs):
        options.append(kwargs)
        return tuple(
            SimpleNamespace(
                model=fitted, converged=True, standard_errors=group_errors or {}
            )
            for fitted, group_errors in zip((reference, focal), errors, strict=True)
        )

    monkeypatch.setattr("mirt.diagnostics._utils.fit_group_models", fake_fit)
    return options


def _calibration_data():
    return np.zeros((8, 6), dtype=int), np.repeat([0, 1], 4)


def _calibration_errors():
    return (
        {"discrimination": np.full(6, 0.1), "difficulty": np.full(6, 0.15)},
        {"discrimination": np.full(6, 0.12), "difficulty": np.full(6, 0.125)},
    )


class TestCommonScaleDIF:
    """DIF statistics compare groups on one latent scale."""

    def test_wald_uses_linked_estimates_and_rescaled_errors(self, monkeypatch):
        reference, focal = _impact_calibrations(dif_shift=0.6)
        options = _install_calibrations(
            monkeypatch, reference, focal, _calibration_errors()
        )
        data, groups = _calibration_data()

        result = compute_dif(
            data, groups, method="wald", anchors=[1, 2, 3, 4, 5], n_quadpts=11
        )

        A, B = result["linking_constants"]
        assert A == pytest.approx(1.2, rel=1e-5)
        assert B == pytest.approx(-1.0, abs=1e-5)
        # a: 0 / (0.1**2 + (0.12 / 1.2)**2); b: 0.6**2 / (0.15**2 + (1.2 * 0.125)**2)
        assert result["statistic"][0] == pytest.approx(8.0, rel=1e-4)
        assert result["df"][0] == 2
        assert result["p_value"][0] == pytest.approx(np.exp(-4.0), rel=1e-3)
        assert result["effect_size"][0] == pytest.approx(0.6, abs=1e-4)
        assert result["classification"][0] == "B"
        assert np.all(np.isnan(result["statistic"][1:]))
        np.testing.assert_array_equal(result["tested"], [True] + [False] * 5)
        assert result["anchors"] == [1, 2, 3, 4, 5]
        assert options[0]["compute_standard_errors"] is True

    def test_lord_is_an_alias_of_wald(self, monkeypatch):
        reference, focal = _impact_calibrations(dif_shift=0.6)
        _install_calibrations(monkeypatch, reference, focal, _calibration_errors())
        data, groups = _calibration_data()

        wald = compute_dif(data, groups, method="wald")
        lord = compute_dif(data, groups, method="lord")

        for key in ("statistic", "p_value", "effect_size"):
            np.testing.assert_array_equal(wald[key], lord[key])

    def test_raju_areas_ignore_pure_impact_and_report_no_p_values(self, monkeypatch):
        reference, focal = _impact_calibrations()
        options = _install_calibrations(monkeypatch, reference, focal)
        data, groups = _calibration_data()

        result = compute_dif(data, groups, method="raju", p_adjust="holm")

        np.testing.assert_allclose(result["statistic"], 0.0, atol=1e-4)
        assert np.all(np.isnan(result["p_value"]))
        assert np.all(np.isnan(result["p_value_adjusted"]))
        assert np.all(result["classification"] == "A")
        assert result["method"] == "raju"
        assert options[0]["compute_standard_errors"] is False

    def test_raju_signed_area_measures_the_location_shift(self, monkeypatch):
        reference, focal = _impact_calibrations(dif_shift=0.6)
        _install_calibrations(monkeypatch, reference, focal)
        data, groups = _calibration_data()

        result = compute_dif(data, groups, method="raju", anchors=[1, 2, 3, 4, 5])

        # Areas between the true curves (b = -1.5 versus -0.9, a = 0.8) over
        # the reference-scale range [-4, 4].
        theta = np.linspace(-4.0, 4.0, 100)
        difference = 1.0 / (1.0 + np.exp(-0.8 * (theta + 1.5))) - 1.0 / (
            1.0 + np.exp(-0.8 * (theta + 0.9))
        )
        expected = np.trapezoid(difference, theta)
        assert result["effect_size"][0] == pytest.approx(expected, abs=1e-4)
        assert result["statistic"][0] == pytest.approx(expected, abs=1e-4)
        assert result["classification"][0] == "B"
        assert np.all(np.isnan(result["effect_size"][1:]))
        flags = flag_dif_items(result, min_effect_size=0.5)
        np.testing.assert_array_equal(flags, [True] + [False] * 5)

    def test_linking_methods_need_two_anchors(self, monkeypatch):
        def unexpected_fit(*args, **kwargs):
            raise AssertionError("fit should not run")

        monkeypatch.setattr("mirt.diagnostics._utils.fit_group_models", unexpected_fit)
        data, groups = _calibration_data()

        with pytest.raises(ValueError, match="at least 2"):
            compute_dif(data, groups, method="wald", anchors=[1])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"model": "NRM"}, "model must be one of"),
            ({"scheme": "forward"}, "scheme must be one of"),
            ({"anchors": [0, 0]}, "duplicate"),
            ({"n_jobs": 0}, "n_jobs"),
        ],
    )
    def test_options_are_validated(self, kwargs, message):
        data, groups = _calibration_data()
        with pytest.raises(ValueError, match=message):
            compute_dif(data, groups, **kwargs)

    def test_likelihood_ratio_maps_multigroup_rows(self, monkeypatch):
        from mirt.multigroup import dif as multigroup_dif_module
        from mirt.multigroup.dif import _DIFTestRow, _DIFTestTable

        calls: list[dict] = []

        def fake_run(data, groups, model, **kwargs):
            calls.append(kwargs)
            reference = ["x", "y"].index(kwargs["reference_group"])
            parameters = [{"difficulty": 0.2}, {"difficulty": 0.2}]
            parameters[1 - reference] = {"difficulty": 1.0}
            row = _DIFTestRow(
                item=0,
                chi2=12.0,
                df=2.0,
                p_value=0.002,
                p_value_adjusted=0.004,
                delta_aic=8.0,
                delta_bic=1.0,
                converged=True,
                round=1,
                group_parameters=tuple(parameters),
            )
            return _DIFTestTable(
                rows=[row],
                item_names=["a", "b", "c"],
                group_labels=["x", "y"],
                reference_group=reference,
                n_categories=None,
                alpha=0.05,
                p_adjust=kwargs["p_adjust"],
            )

        monkeypatch.setattr(multigroup_dif_module, "run_multigroup_dif", fake_run)
        data = np.zeros((4, 3), dtype=int)
        groups = np.array(["x", "x", "y", "y"])

        result = compute_dif(
            data,
            groups,
            focal_group="x",
            anchors=[1, 2],
            scheme="add",
            p_adjust="holm",
            n_jobs=2,
        )

        # The reference group goes by label, never by an ambiguous index.
        assert calls[0]["reference_group"] == "y"
        assert calls[0]["anchors"] == [1, 2]
        assert calls[0]["scheme"] == "add"
        assert calls[0]["p_adjust"] == "holm"
        assert calls[0]["n_jobs"] == 2
        np.testing.assert_allclose(result["statistic"], [12.0, np.nan, np.nan])
        np.testing.assert_allclose(result["p_value_adjusted"], [0.004, np.nan, np.nan])
        assert result["effect_size"][0] == pytest.approx(0.8)
        assert list(result["classification"]) == ["C", "A", "A"]
        assert result["linking_constants"] is None
        np.testing.assert_array_equal(result["tested"], [True, False, False])
        # Untested anchors report the convergence of the whole analysis.
        np.testing.assert_array_equal(result["converged"], [True, True, True])

    def test_likelihood_ratio_detects_dif_despite_impact(self):
        data, groups = _simulate_two_groups(1, n_per_group=250, n_items=8, impact=-0.5)

        result = compute_dif(
            data,
            groups,
            method="likelihood_ratio",
            n_quadpts=9,
            tol=1e-3,
            p_adjust="holm",
        )

        assert result["p_value"][0] < 1e-4
        np.testing.assert_array_equal(
            np.flatnonzero(result["p_value_adjusted"] < 0.05), [0]
        )
        assert result["effect_size"][0] == pytest.approx(1.2, abs=0.4)
        assert result["classification"][0] == "C"
        assert np.all(result["df"] == 2)

    def test_wald_detects_dif_despite_impact(self):
        """Regression: unlinked fits without SEs gave p = 1 for every item."""
        data, groups = _simulate_two_groups(1, n_per_group=600, n_items=10)

        result = compute_dif(data, groups, method="wald", p_adjust="holm")

        assert result["p_value"][0] < 1e-4
        assert not np.allclose(result["p_value"], 1.0)
        assert result["classification"][0] == "C"
        assert result["effect_size"][0] == pytest.approx(1.2, abs=0.4)

    def test_raju_does_not_report_impact_as_dif(self):
        """Regression: an invented SE flagged every item under impact."""
        data, groups = _simulate_two_groups(
            2, n_per_group=600, n_items=10, shift=0.0, impact=-1.0
        )

        result = compute_dif(data, groups, method="raju")

        assert np.all(result["classification"] == "A")
        assert np.all(np.isnan(result["p_value"]))
        assert np.max(np.abs(result["effect_size"])) < 0.3

    def test_ets_classes_need_significance_unless_descriptive(self):
        effects = np.array([0.8, 0.5, 0.2, np.nan, -0.7])
        p_values = np.array([0.01, 0.01, 0.01, 0.01, np.nan])

        np.testing.assert_array_equal(
            _ets_classify(effects, p_values), ["C", "B", "A", "A", "A"]
        )
        np.testing.assert_array_equal(
            _ets_classify(effects, None), ["C", "B", "A", "A", "C"]
        )

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_dataframe_wrapper_drops_metadata(self, monkeypatch):
        import mirt

        reference, focal = _impact_calibrations(dif_shift=0.6)
        _install_calibrations(monkeypatch, reference, focal)
        data, groups = _calibration_data()

        frame = mirt.dif(data, groups, method="raju", anchors=[1, 2, 3, 4, 5])

        assert {"statistic", "df", "tested", "converged"} <= set(frame.columns)
        assert not {"method", "anchors", "linking_constants"} & set(frame.columns)


@pytest.mark.slow
def test_wald_impact_only_holm_false_positives_stay_rare():
    for seed in range(3):
        data, groups = _simulate_two_groups(
            10 + seed, n_per_group=800, n_items=10, shift=0.0, impact=-1.0
        )
        result = compute_dif(data, groups, method="wald", p_adjust="holm")
        assert np.count_nonzero(result["p_value_adjusted"] < 0.05) <= 1


def test_likelihood_ratio_entry_points_share_the_adjustment_default():
    import inspect

    import mirt
    from mirt.multigroup import multigroup_dif, select_dif_anchors

    defaults = {
        function.__name__: inspect.signature(function).parameters["p_adjust"].default
        for function in (compute_dif, mirt.dif, multigroup_dif, select_dif_anchors)
    }

    assert set(defaults.values()) == {"none"}, defaults


def test_compute_dif_and_multigroup_dif_flag_the_same_items():
    from mirt.multigroup import multigroup_dif

    data, groups = _simulate_two_groups(3, n_per_group=200, n_items=6)
    options = {"model": "1PL", "n_quadpts": 7, "tol": 1e-2}

    result = compute_dif(data, groups, method="likelihood_ratio", **options)
    table = multigroup_dif(data, groups, **options)

    np.testing.assert_allclose(result["p_value"], table["p_value"])
    np.testing.assert_array_equal(
        result["p_value_adjusted"] < 0.05, np.asarray(table["flagged"])
    )


def test_anchors_may_be_item_names(monkeypatch):
    from mirt.multigroup import dif as multigroup_dif_module

    calls: list[dict] = []

    def fake_run(data, groups, model, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after the anchors are resolved")

    monkeypatch.setattr(multigroup_dif_module, "run_multigroup_dif", fake_run)
    data, groups = _simulate_two_groups(4, n_per_group=60, n_items=5)

    with pytest.raises(RuntimeError, match="stop"):
        compute_dif(data, groups, anchors=["Item_3", 1])
    assert calls[0]["anchors"] == [1, 3]
    with pytest.raises(ValueError, match="unknown item name"):
        compute_dif(data, groups, anchors=["Item_9"])

    linked = compute_dif(data, groups, method="raju", anchors=["Item_2", "Item_4"])
    assert linked["anchors"] == [2, 4]


@pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
def test_anchor_names_follow_dataframe_columns():
    from mirt.utils.dataframe import create_dataframe

    data, groups = _simulate_two_groups(4, n_per_group=60, n_items=5)
    frame = create_dataframe({f"q{item}": data[:, item] for item in range(5)})

    result = compute_dif(frame, groups, method="raju", anchors=["q0", "q2", "q3"])

    assert result["anchors"] == [0, 2, 3]


def test_likelihood_ratio_passes_integer_labels_unambiguously(monkeypatch):
    from mirt.multigroup import dif as multigroup_dif_module

    calls: list[dict] = []

    def fake_run(data, groups, model, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(multigroup_dif_module, "run_multigroup_dif", fake_run)
    data, _ = _simulate_two_groups(4, n_per_group=60, n_items=5)
    # Labels 1 and 2: the reference group is label 2, which is index 1.
    groups = np.repeat([1, 2], 60)

    with pytest.raises(RuntimeError):
        compute_dif(data, groups, focal_group=1)
    assert calls[0]["reference_group"] == "2"


def test_integer_reference_group_that_names_another_group_is_rejected():
    from mirt.multigroup import _prepare_multigroup

    data, _ = _simulate_two_groups(4, n_per_group=30, n_items=4)
    groups = np.repeat([1, 2], 30)
    prepare = {"n_categories": None, "item_names": None}

    with pytest.raises(ValueError, match="also the label of group 0"):
        _prepare_multigroup(data, groups, "2PL", reference_group=1, **prepare)
    by_label = _prepare_multigroup(data, groups, "2PL", reference_group="1", **prepare)
    by_index = _prepare_multigroup(data, groups, "2PL", reference_group=0, **prepare)
    assert by_label[2] == by_index[2] == 0
    labeled = np.repeat([0, 1], 30)
    same = _prepare_multigroup(data, labeled, "2PL", reference_group=1, **prepare)
    assert same[2] == 1


def test_default_reference_group_is_the_first_group_for_any_labels():
    import mirt
    from mirt.multigroup import _prepare_multigroup

    data, _ = _simulate_two_groups(4, n_per_group=30, n_items=4)
    # Label 0 sorts second, so an explicit 0 is ambiguous, but the default
    # still selects the first group.
    groups = np.repeat([-1, 0], 30)
    prepare = {"n_categories": None, "item_names": None}

    default = _prepare_multigroup(data, groups, "2PL", reference_group=None, **prepare)
    assert default[2] == 0
    with pytest.raises(ValueError, match="reference_group='-1' for group index 0"):
        _prepare_multigroup(data, groups, "2PL", reference_group=0, **prepare)
    result = mirt.fit_multigroup(data, groups, n_quadpts=7, max_iter=2)
    assert result.group_labels == ["-1", "0"]
