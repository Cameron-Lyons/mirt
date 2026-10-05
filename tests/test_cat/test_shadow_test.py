"""Shadow-test CAT enforces assembly constraints during adaptive selection."""

import numpy as np
import pytest

from mirt.cat import (
    CATEngine,
    ContentArea,
    ContentBlueprint,
    FormAssemblyResult,
    ShadowTestSelection,
)
from mirt.cat._lockstep import supports_lockstep
from mirt.cat.exposure import (
    ExposureControl,
    ProgressiveRestricted,
    Randomesque,
    SympsonHetter,
)
from mirt.models import GradedResponseModel, TwoParameterLogistic

N_ITEMS = 90
LENGTH = 12


@pytest.fixture(scope="module")
def model() -> TwoParameterLogistic:
    rng = np.random.default_rng(11)
    pool = TwoParameterLogistic(n_items=N_ITEMS)
    pool.set_parameters(
        discrimination=rng.lognormal(0.0, 0.4, N_ITEMS),
        difficulty=rng.normal(0.0, 1.0, N_ITEMS),
    )
    pool._is_fitted = True
    return pool


@pytest.fixture(scope="module")
def constraints(model) -> dict:
    information = np.asarray(model.information(np.array([[0.0]]))).ravel()
    ranked = [int(item) for item in np.argsort(-information)]
    rng = np.random.default_rng(5)
    return {
        "blueprint": ContentBlueprint(
            [
                ContentArea("A", items=set(range(30)), min_items=4, max_items=4),
                ContentArea("B", items=set(range(30, 60)), min_items=4, max_items=4),
                ContentArea("C", items=set(range(60, 90)), min_items=4, max_items=4),
            ]
        ),
        # The most informative items are enemies, bundled, or costly, so an
        # unconstrained selection would violate every constraint.
        "enemy_pairs": {(ranked[0], ranked[1]), (ranked[2], ranked[3])},
        "item_bundles": [{ranked[4], ranked[5]}, {ranked[6], ranked[7], 88}],
        "item_costs": np.where(
            np.isin(np.arange(N_ITEMS), ranked[8:12]), 5.0, rng.uniform(0.5, 1.0, 90)
        ),
        "max_cost": 14.0,
    }


def _assert_constraints(items: list[int], constraints: dict) -> None:
    administered = set(items)
    assert len(administered) == len(items) == LENGTH
    blueprint = constraints["blueprint"]
    assert blueprint.get_area_counts(items) == {"A": 4, "B": 4, "C": 4}
    for first, second in constraints["enemy_pairs"]:
        assert not (first in administered and second in administered)
    for bundle in constraints["item_bundles"]:
        assert bundle <= administered or not bundle & administered
    assert constraints["item_costs"][items].sum() <= constraints["max_cost"] + 1e-9


def _fixed_length_engine(model, selection, **options) -> CATEngine:
    return CATEngine(
        model,
        item_selection=selection,
        se_threshold=1e-6,
        max_items=LENGTH,
        min_items=LENGTH,
        seed=3,
        **options,
    )


def test_completed_sessions_satisfy_every_constraint(model, constraints):
    selection = ShadowTestSelection(**constraints)
    engine = _fixed_length_engine(model, selection)

    for theta in (-2.5, -0.5, 0.0, 1.0, 2.5):
        result = engine.run_simulation(theta)

        _assert_constraints(result.items_administered, constraints)
        assert result.stopping_reason == f"Maximum items reached ({LENGTH})"


def test_unconstrained_selection_follows_maximum_information(model):
    for theta in (-1.5, 0.3, 2.0):
        shadow = CATEngine(
            model, item_selection=ShadowTestSelection(), max_items=15, seed=7
        ).run_simulation(theta)
        reference = CATEngine(
            model, item_selection="MFI", max_items=15, seed=7
        ).run_simulation(theta)

        assert shadow.items_administered == reference.items_administered
        np.testing.assert_allclose(shadow.theta_history, reference.theta_history)


def test_shadow_test_contains_administered_items_and_spans_engine_horizon(
    model, constraints
):
    selection = ShadowTestSelection(**constraints)
    engine = _fixed_length_engine(model, selection)

    for _ in range(5):
        engine.administer_item(1)
    pending = engine.select_next_item()

    shadow = selection.last_shadow_test
    assert isinstance(shadow, FormAssemblyResult)
    assert shadow.n_items == LENGTH
    assert set(engine.get_current_state().items_administered) < set(
        shadow.selected_items.tolist()
    )
    assert pending in shadow.selected_items


def test_infeasible_constraints_fail_at_first_selection(model):
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items=set(range(30)), min_items=8),
            ContentArea("B", items=set(range(30, 60)), min_items=8),
        ]
    )
    engine = _fixed_length_engine(model, ShadowTestSelection(blueprint=blueprint))

    with pytest.raises(
        RuntimeError,
        match="shadow test assembly after 0 administered items failed for a 12-item",
    ):
        engine.select_next_item()


def test_unbounded_engine_reports_pool_length_shadow_test(model, constraints):
    # Without max_items the shadow test spans the pool, which the blueprint
    # maxima cannot allow; the error names the length to fix.
    engine = CATEngine(
        model,
        item_selection=ShadowTestSelection(blueprint=constraints["blueprint"]),
        seed=0,
    )

    with pytest.raises(RuntimeError, match=f"{N_ITEMS}-item shadow test"):
        engine.select_next_item()


def test_explicit_length_shorter_than_session_reports_completion(model):
    engine = CATEngine(
        model,
        item_selection=ShadowTestSelection(test_length=3),
        se_threshold=1e-6,
        max_items=5,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="3-item shadow test is complete"):
        engine.run_simulation(0.0)


def test_randomesque_draws_within_the_shadow_test(model, constraints):
    paths = set()
    for seed in range(4):
        engine = _fixed_length_engine(
            model,
            ShadowTestSelection(**constraints),
            exposure_control=Randomesque(k=4, seed=seed),
        )
        result = engine.run_simulation(0.0)
        _assert_constraints(result.items_administered, constraints)
        paths.add(tuple(result.items_administered))

    assert len(paths) > 1


def test_seeded_sessions_are_reproducible(model, constraints):
    def run() -> list[list[int]]:
        engine = _fixed_length_engine(
            model,
            ShadowTestSelection(**constraints),
            exposure_control=Randomesque(k=3, seed=21),
        )
        return [engine.run_simulation(theta).items_administered for theta in (0, 1)]

    assert run() == run()


def test_exposure_exclusions_are_relaxed_when_constraints_require(model):
    # Area B consists of two items and both are required, but exposure
    # control never makes item 30 eligible.
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items=set(range(30)), min_items=4, max_items=4),
            ContentArea("B", items={30, 31}, min_items=2, max_items=2),
        ]
    )
    exposure = SympsonHetter(exposure_params={30: 0.0}, seed=0)
    engine = CATEngine(
        model,
        item_selection=ShadowTestSelection(blueprint=blueprint),
        se_threshold=1e-6,
        max_items=6,
        min_items=6,
        exposure_control=exposure,
        seed=0,
    )

    result = engine.run_simulation(0.5)

    assert blueprint.get_area_counts(result.items_administered) == {"A": 4, "B": 2}
    assert 30 in result.items_administered


class _BlockAfterAdministration(ExposureControl):
    """Make ``blocked`` ineligible once ``trigger`` has been administered."""

    def __init__(self, trigger: int, blocked: int) -> None:
        self.trigger = trigger
        self.blocked = blocked
        self.triggered = False

    def filter_items(self, available_items, model, theta):
        excluded = {self.blocked} if self.triggered else set()
        return set(available_items) - excluded

    def update(self, selected_item: int) -> None:
        self.triggered = self.triggered or selected_item == self.trigger

    def reset(self) -> None:
        self.triggered = False


def test_bundle_member_ineligible_mid_test_is_still_completed(model):
    information = np.asarray(model.information(np.array([[0.0]]))).ravel()
    first, second = (int(item) for item in np.argsort(-information)[:2])
    engine = CATEngine(
        model,
        item_selection=ShadowTestSelection(item_bundles=[{first, second}]),
        se_threshold=1e-6,
        max_items=5,
        min_items=5,
        exposure_control=_BlockAfterAdministration(first, second),
        seed=1,
    )

    result = engine.run_simulation(0.0)

    assert result.items_administered[0] == first
    assert second in result.items_administered


def test_one_shot_constraint_iterators_bind_every_selection(model, constraints):
    # Generators used to be exhausted by the first shadow test, silently
    # dropping enemy pairs and bundles from every later selection.
    options = dict(constraints)
    options["enemy_pairs"] = (pair for pair in constraints["enemy_pairs"])
    options["item_bundles"] = (bundle for bundle in constraints["item_bundles"])
    engine = _fixed_length_engine(model, ShadowTestSelection(**options))

    for theta in (-0.5, 0.0, 1.0):
        _assert_constraints(
            engine.run_simulation(theta).items_administered, constraints
        )


def test_constraints_bind_against_unconstrained_selection(model, constraints):
    reference = _fixed_length_engine(model, "MFI").run_simulation(0.0)

    with pytest.raises(AssertionError):
        _assert_constraints(reference.items_administered, constraints)


def test_batch_simulation_runs_independent_sessions(model, constraints):
    engine = _fixed_length_engine(model, ShadowTestSelection(**constraints))

    assert not supports_lockstep(engine)
    results = engine.run_batch_simulation([-1.0, 1.0], n_replications=2)

    assert len(results) == 4
    for result in results:
        _assert_constraints(result.items_administered, constraints)
        assert len(result.theta_history) == LENGTH


def test_progressive_exposure_is_rejected(model):
    with pytest.raises(ValueError, match="progressive exposure"):
        CATEngine(
            model,
            item_selection=ShadowTestSelection(),
            exposure_control=ProgressiveRestricted(),
        )


def test_polytomous_pool():
    rng = np.random.default_rng(2)
    pool = GradedResponseModel(n_items=24, n_categories=4)
    pool.set_parameters(
        discrimination=rng.lognormal(0.0, 0.3, 24),
        thresholds=np.sort(rng.normal(0.0, 1.0, (24, 3)), axis=1),
    )
    pool._is_fitted = True
    blueprint = ContentBlueprint(
        [
            ContentArea("A", items=set(range(12)), min_items=3, max_items=3),
            ContentArea("B", items=set(range(12, 24)), min_items=3, max_items=3),
        ]
    )
    engine = CATEngine(
        pool,
        item_selection=ShadowTestSelection(blueprint=blueprint),
        se_threshold=1e-6,
        max_items=6,
        min_items=6,
        seed=4,
    )

    result = engine.run_simulation(1.0)

    assert blueprint.get_area_counts(result.items_administered) == {"A": 3, "B": 3}


def test_empty_available_items():
    selection = ShadowTestSelection()

    assert selection.get_item_criteria(object(), 0.0, set()) == {}
    with pytest.raises(ValueError, match="No available items"):
        selection.select_item(object(), 0.0, set())


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"test_length": 0}, ValueError, "test_length"),
        ({"test_length": 2.5}, ValueError, "test_length"),
        ({"test_length": True}, ValueError, "test_length"),
        ({"blueprint": [ContentArea("A", items={0})]}, TypeError, "ContentBlueprint"),
        ({"solver_options": [("time_limit", 1)]}, TypeError, "mapping"),
    ],
)
def test_rejects_invalid_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        ShadowTestSelection(**kwargs)
