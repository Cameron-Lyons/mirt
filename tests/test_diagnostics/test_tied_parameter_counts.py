"""Fit diagnostics count coordinates tied by equality constraints once."""

import numpy as np
import pytest

import mirt
from mirt.diagnostics.itemfit import _sx2_parameter_counts, compute_itemfit
from mirt.diagnostics.modelfit import compute_m2
from mirt.models.dichotomous import TwoParameterLogistic

TIED = [{"parameter": "discrimination", "items": [0, 1, 2]}]


@pytest.fixture(scope="module")
def tied_fit():
    rng = np.random.default_rng(17)
    n_items = 6
    slopes = np.array([1.2, 1.2, 1.2, 0.8, 1.6, 1.0])
    difficulty = np.linspace(-1.0, 1.0, n_items)
    theta = rng.normal(size=(1500, 1))
    probability = 1.0 / (1.0 + np.exp(-slopes * (theta - difficulty)))
    responses = (rng.random(probability.shape) < probability).astype(int)
    result = mirt.fit_mirt(
        responses, model="2PL", constraints=TIED, verbose=False, n_quadpts=21
    )
    return result, responses


def test_item_parameter_shares_split_a_tied_group_over_its_items():
    model = TwoParameterLogistic(4)

    counts = _sx2_parameter_counts(model, None, constraints=TIED[:1])

    np.testing.assert_allclose(counts[:3], 1.0 + 1.0 / 3.0)
    assert counts[3] == 2.0
    # The shares add up to the parameters the constrained fit estimates.
    assert counts.sum() == pytest.approx(model.n_parameters - 2)
    np.testing.assert_array_equal(_sx2_parameter_counts(model, None), 2.0)
    # Explicit counts take precedence.
    explicit = np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(
        _sx2_parameter_counts(model, explicit, constraints=TIED), explicit
    )


def test_sx2_degrees_of_freedom_count_tied_slopes_once(tied_fit):
    result, responses = tied_fit

    # An empty sequence counts every free coordinate separately.
    free = compute_itemfit(result, responses, ["S_X2"], constraints=[])
    tied = compute_itemfit(result, responses, ["S_X2"], constraints=TIED)
    top_level = mirt.itemfit(result, responses, ["S_X2"], constraints=TIED)

    np.testing.assert_allclose(tied["S_X2"], free["S_X2"])
    np.testing.assert_allclose(tied["df"][:3], free["df"][:3] + 2.0 / 3.0)
    np.testing.assert_allclose(tied["df"][3:], free["df"][3:])
    assert np.all(tied["p_value"][:3] > free["p_value"][:3])
    np.testing.assert_allclose(np.asarray(top_level["df"]), tied["df"])


def test_m2_projects_one_direction_per_tied_group(tied_fit):
    result, responses = tied_fit
    n_items = responses.shape[1]
    moments = n_items + n_items * (n_items - 1) // 2

    free = compute_m2(result, responses, constraints=[])
    tied = compute_m2(result, responses, constraints=TIED)

    assert free["df"] == moments - 2 * n_items
    # Three tied slopes are one parameter, so two degrees of freedom return.
    assert tied["df"] == free["df"] + 2
    # The tied tangent spans a subspace of the free one, so less is removed.
    assert tied["M2"] >= free["M2"] - 1e-8
    indices = mirt.compute_fit_indices(result, responses, constraints=TIED)
    assert indices["M2_df"] == tied["df"]
    assert indices["M2"] == pytest.approx(tied["M2"])


def test_fit_result_supplies_its_own_constraints(tied_fit):
    """A constrained fit counts its tied coordinates without re-passing them."""
    result, responses = tied_fit
    model = result.model

    assert compute_m2(result, responses) == compute_m2(
        result, responses, constraints=TIED
    )
    # A bare model carries no constraints.
    assert compute_m2(model, responses) == compute_m2(result, responses, constraints=[])
    indices = mirt.compute_fit_indices(result, responses)
    assert indices["M2_df"] == compute_m2(result, responses, constraints=TIED)["df"]

    tied = compute_itemfit(result, responses, ["S_X2", "X2"], constraints=TIED)
    for fit_stats in (
        compute_itemfit(result, responses, ["S_X2", "X2"]),
        mirt.itemfit(result, responses, ["S_X2", "X2"]),
    ):
        for name in ("df", "X2_df", "p_value", "X2_p"):
            np.testing.assert_allclose(np.asarray(fit_stats[name]), tied[name])
    bare = compute_itemfit(model, responses, ["S_X2"])
    np.testing.assert_allclose(bare["df"][:3], tied["df"][:3] - 2.0 / 3.0)


def test_constrain_statement_is_counted_like_constraints():
    data = mirt.simdata("2PL", n_persons=800, n_items=6, seed=3)
    result = mirt.fit_mirt(data, spec="F = 1-6\nCONSTRAIN = (1-3, a1)", verbose=False)

    fitted = compute_m2(result, data)
    free = compute_m2(result.model, data)
    assert fitted["df"] == free["df"] + 2
    assert fitted == compute_m2(result.model, data, constraints=TIED)
