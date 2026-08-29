import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt.diagnostics.ld as ld_module
from mirt.diagnostics.ld import (
    LDResult,
    compute_ld_chi2,
    compute_ld_statistics,
    flag_ld_pairs,
    ld_summary_table,
)


def _chi2_matrix_from_p_values(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    n_items = 3
    rows, columns = np.triu_indices(n_items, k=1)
    matrix = np.full((n_items, n_items), np.nan)
    values = stats.chi2.isf(p_values, df=1)
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


class MinimalModel:
    is_polytomous = True
    item_names = ["one", "two", "three"]


def test_compute_ld_statistics_stores_raw_and_adjusted_pair_probabilities(monkeypatch):
    raw_p_values = np.array([0.01, 0.03, 0.20])
    chi2 = _chi2_matrix_from_p_values(raw_p_values)
    q3 = np.zeros((3, 3))
    monkeypatch.setattr(
        ld_module,
        "_compute_residuals_and_positive_probabilities",
        lambda model, responses, theta: (
            np.zeros_like(responses, dtype=np.float64),
            np.full_like(responses, 0.5, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(ld_module, "_compute_q3", lambda residuals, responses: q3)
    monkeypatch.setattr(
        ld_module,
        "_compute_ld_chi2_g2",
        lambda model, responses, theta, n_quadpts, **kwargs: (chi2, chi2.copy()),
    )

    result = compute_ld_statistics(
        MinimalModel(),
        np.zeros((12, 3), dtype=np.int64),
        theta=np.zeros(12),
        p_adjust="bonferroni",
    )

    rows, columns = np.triu_indices(3, k=1)
    assert_allclose(result.chi2_p_value_matrix[rows, columns], raw_p_values)
    assert_allclose(
        result.chi2_adjusted_p_value_matrix[rows, columns],
        [0.03, 0.09, 0.60],
    )
    assert result.p_adjustment == "bonferroni"
    assert len(result.chi2_flagged) == 1
    assert result.chi2_flagged[0][:3] == pytest.approx((0, 1, chi2[0, 1]))
    assert result.chi2_flagged[0][3] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", [0.01, 0.03, 0.20]),
        ("bonferroni", [0.03, 0.09, 0.60]),
        ("holm", [0.03, 0.06, 0.20]),
        ("fdr_bh", [0.03, 0.045, 0.20]),
    ],
)
def test_compute_ld_chi2_returns_requested_adjustment(monkeypatch, method, expected):
    chi2 = _chi2_matrix_from_p_values([0.01, 0.03, 0.20])
    monkeypatch.setattr(
        ld_module,
        "_compute_ld_chi2_g2",
        lambda model, responses, theta, n_quadpts: (chi2, chi2.copy()),
    )

    _, p_value_matrix = compute_ld_chi2(
        MinimalModel(),
        np.zeros((12, 3), dtype=np.int64),
        theta=np.zeros(12),
        p_adjust=method,
    )

    rows, columns = np.triu_indices(3, k=1)
    assert_allclose(p_value_matrix[rows, columns], expected)
    assert_allclose(p_value_matrix, p_value_matrix.T)


def test_flag_ld_pairs_uses_stored_adjustment_and_all_pair_tests():
    raw_p_values = np.array([0.01, 0.03, 0.20])
    chi2 = _chi2_matrix_from_p_values(raw_p_values)
    q3 = np.zeros_like(chi2)
    result = LDResult(
        q3,
        chi2,
        chi2.copy(),
        q3.copy(),
        [],
        [],
        p_adjustment="bonferroni",
    )

    assert flag_ld_pairs(result, method="chi2") == [(0, 1)]
    assert flag_ld_pairs(result, method="chi2", p_adjust="none") == [
        (0, 1),
        (0, 2),
    ]


def test_summary_reports_adjusted_pair_probabilities():
    chi2 = _chi2_matrix_from_p_values([0.01, 0.03, 0.20])
    q3 = np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.1], [0.2, 0.1, 0.0]])
    adjusted = np.full((3, 3), np.nan)
    rows, columns = np.triu_indices(3, k=1)
    adjusted[rows, columns] = [0.03, 0.09, 0.60]
    adjusted[columns, rows] = [0.03, 0.09, 0.60]
    result = LDResult(
        q3,
        chi2,
        chi2.copy(),
        q3.copy(),
        [],
        [(0, 1, chi2[0, 1], 0.03)],
        ["one", "two", "three"],
        chi2_adjusted_p_value_matrix=adjusted,
        p_adjustment="bonferroni",
    )

    summary = result.summary()
    table = ld_summary_table(result, top_n=1)

    assert "bonferroni-adjusted p" in summary
    assert "adj p" in table
    assert "0.0300" in table


@pytest.mark.parametrize("method", ["bad", "BH", ""])
def test_rejects_unknown_pair_adjustment(method):
    with pytest.raises(ValueError, match="p_adjust"):
        compute_ld_statistics(
            MinimalModel(),
            np.zeros((2, 3), dtype=np.int64),
            theta=np.zeros(2),
            p_adjust=method,
        )

    matrix = np.zeros((2, 2))
    result = LDResult(matrix, matrix, matrix, matrix, [], [])
    with pytest.raises(ValueError, match="p_adjust"):
        flag_ld_pairs(result, method="chi2", p_adjust=method)
