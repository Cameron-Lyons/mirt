"""CAT and MCAT share content, exposure, and strategy dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from mirt.cat import (
    CATEngine,
    DOptimality,
    MCATEngine,
    ProgressiveRestricted,
    Randomesque,
)
from mirt.cat.mcat_selection import RandomMCATSelection
from mirt.cat.selection import (
    KullbackLeibler,
    MaxExpectedInformation,
    MaxFisherInformation,
    RandomSelection,
    UrryRule,
)
from mirt.models import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    MultidimensionalModel,
    TwoParameterLogistic,
)


def _mirt_model(n_items: int = 60) -> MultidimensionalModel:
    rng = np.random.default_rng(5)
    model = MultidimensionalModel(n_items=n_items, n_factors=2)
    model.set_parameters(
        slopes=rng.uniform(0.4, 2.0, (n_items, 2)), intercepts=rng.normal(size=n_items)
    )
    model._is_fitted = True
    return model


def _two_pl(n_items: int = 60) -> TwoParameterLogistic:
    rng = np.random.default_rng(6)
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=rng.uniform(0.5, 2.0, n_items),
        difficulty=rng.normal(size=n_items),
    )
    model._is_fitted = True
    return model


def _top_items(criteria: dict[int, float], k: int) -> set[int]:
    ranked = sorted(criteria.items(), key=lambda pair: (-pair[1], pair[0]))
    return {item for item, _ in ranked[:k]}


def test_mcat_randomesque_draws_first_items_from_top_d_optimal_items():
    model = _mirt_model()
    engine = MCATEngine(
        model, exposure_control=Randomesque(k=5, seed=0), max_items=3, seed=1
    )
    top = _top_items(
        DOptimality().get_item_criteria(
            model, np.zeros(2), np.eye(2), set(range(model.n_items))
        ),
        5,
    )

    first_items = [
        engine.run_simulation(np.array([0.2, -0.4])).items_administered[0]
        for _ in range(30)
    ]

    assert len(set(first_items)) > 1
    assert set(first_items) <= top


@pytest.mark.parametrize("kind", ["CAT", "MCAT"])
def test_top_one_randomesque_reproduces_strategy_choices(kind):
    model = _two_pl() if kind == "CAT" else _mirt_model()
    engine_class = CATEngine if kind == "CAT" else MCATEngine
    true_theta = 0.4 if kind == "CAT" else np.array([0.4, -0.3])

    plain = engine_class(model, max_items=8, seed=3).run_simulation(true_theta)
    randomesque = engine_class(
        model, max_items=8, seed=3, exposure_control=Randomesque(k=1, seed=9)
    ).run_simulation(true_theta)

    assert randomesque.items_administered == plain.items_administered


@pytest.mark.parametrize("kind", ["CAT", "MCAT"])
def test_randomesque_ties_rank_by_item_index(kind):
    # Identical items tie on every criterion.
    if kind == "CAT":
        model = TwoParameterLogistic(n_items=12)
        engine_class = CATEngine
    else:
        model = MultidimensionalModel(n_items=12, n_factors=2)
        engine_class = MCATEngine
    model._is_fitted = True
    engine = engine_class(model, exposure_control=Randomesque(k=1))
    engine._available_items = {11, 7, 3, 9}

    assert engine.select_next_item() == 3


@pytest.mark.parametrize(
    "strategy",
    [MaxFisherInformation(), KullbackLeibler(), MaxExpectedInformation(), UrryRule()],
    ids=["MFI", "KL", "MEI", "Urry"],
)
def test_progressive_control_supports_ordinal_cat(strategy):
    model = GradedResponseModel(n_items=10, n_categories=4)
    model._is_fitted = True
    control = ProgressiveRestricted(window_size=0.3, seed=2)
    engine = CATEngine(
        model, item_selection=strategy, exposure_control=control, max_items=5, seed=4
    )

    first = engine.select_next_item()
    information = np.array(
        [
            model.information(np.array([[0.0]]), item_idx=item).sum()
            for item in range(model.n_items)
        ]
    )
    result = engine.run_simulation(0.3)

    assert information[first] >= information.max() - control.window_size
    assert result.n_items_administered == 5
    assert control.max_information_seen


def test_progressive_control_supports_multidimensional_ordinal_mcat():
    model = GeneralizedPartialCredit(n_items=8, n_factors=2, n_categories=3)
    model._is_fitted = True

    result = MCATEngine(
        model, exposure_control="progressive", max_items=4, seed=2
    ).run_simulation(np.array([0.2, -0.1]))

    assert result.n_items_administered == 4


@pytest.mark.parametrize("kind", ["CAT", "MCAT"])
def test_randomesque_over_random_selection_is_a_seeded_pool_draw(kind):
    if kind == "CAT":
        model = _two_pl()
        engine = CATEngine(
            model,
            item_selection=RandomSelection(seed=4),
            exposure_control=Randomesque(k=3, seed=5),
            max_items=1,
        )
        theta = 0.0
    else:
        model = _mirt_model()
        engine = MCATEngine(
            model,
            item_selection=RandomMCATSelection(seed=4),
            exposure_control=Randomesque(k=3, seed=5),
            max_items=1,
        )
        theta = np.zeros(2)

    first_items = {
        engine.run_simulation(theta).items_administered[0] for _ in range(20)
    }

    # Constant criteria used to confine every draw to the lowest indices.
    assert not first_items <= {0, 1, 2}
    assert len(first_items) > 5


@pytest.mark.parametrize(
    "strategy, arguments",
    [
        (RandomSelection, (0.0,)),
        (RandomMCATSelection, (np.zeros(2), np.eye(2))),
    ],
)
def test_random_strategies_rank_by_seeded_uniform_scores(strategy, arguments):
    model = _mirt_model(10)
    first = strategy(seed=8).get_item_criteria(model, *arguments, {9, 2, 5})
    second = strategy(seed=8).get_item_criteria(model, *arguments, {5, 9, 2})

    assert list(first) == [2, 5, 9]
    assert first == second
    assert all(0.0 <= value < 1.0 for value in first.values())
