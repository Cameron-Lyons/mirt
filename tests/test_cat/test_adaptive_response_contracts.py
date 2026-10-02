"""Invalid submissions must leave the disclosed adaptive item usable."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat import CATEngine, MCATEngine
from mirt.models import GradedResponseModel, MultidimensionalModel, TwoParameterLogistic


def _engine(kind):
    if kind == "cat":
        model = TwoParameterLogistic(n_items=5)
        engine_type = CATEngine
    elif kind == "mcat":
        model = MultidimensionalModel(n_items=5, n_factors=2)
        engine_type = MCATEngine
    else:
        dimensions = 1 if kind == "ordinal_cat" else 2
        model = GradedResponseModel(
            n_items=5, n_factors=dimensions, n_categories=[2, 4, 3, 2, 5]
        )
        engine_type = CATEngine if dimensions == 1 else MCATEngine
    model._is_fitted = True
    return engine_type(model, min_items=5, max_items=5)


@pytest.mark.parametrize("kind", ["cat", "mcat", "ordinal_cat", "ordinal_mcat"])
@pytest.mark.parametrize(
    "response", [2.75, -1, np.nan, np.inf, -np.inf, "1", None, [1], 1 + 0j]
)
def test_invalid_response_preserves_state_and_pending_item(kind, response):
    engine = _engine(kind)
    before = engine.get_current_state()
    item = before.next_item
    available = engine._available_items.copy()

    with pytest.raises(ValueError, match="response"):
        engine.administer_item(response)

    after = engine.get_current_state()
    assert after.next_item == item
    assert after.items_administered == before.items_administered == []
    assert after.responses == []
    assert_allclose(after.theta, before.theta)
    assert_allclose(after.standard_error, before.standard_error)
    assert engine._available_items == available
    assert engine._theta_history == engine._se_history == engine._info_history == []
    assert not after.is_complete

    retried = engine.administer_item(1)
    assert retried.items_administered == [item]
    assert retried.responses == [1]


@pytest.mark.parametrize("kind", ["cat", "mcat", "ordinal_cat", "ordinal_mcat"])
def test_response_range_is_for_the_disclosed_item(kind):
    engine = _engine(kind)
    item = engine.select_next_item()
    categories = engine.model.n_categories[item] if engine.model.is_polytomous else 2
    with pytest.raises(ValueError, match=f"item {item}"):
        engine.administer_item(categories)
    assert engine.select_next_item() == item
    state = engine.administer_item(categories - 1)
    assert state.responses == [categories - 1]


@pytest.mark.parametrize("response", [True, np.int32(1), np.uint64(1), 1.0])
def test_valid_numeric_response_is_stored_as_an_integer(response):
    engine = _engine("cat")
    state = engine.administer_item(response)
    assert state.responses == [1]
    assert type(state.responses[0]) is int
