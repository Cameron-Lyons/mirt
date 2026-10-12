"""Shadow-test item selection for constrained adaptive testing."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from mirt.cat.assembly import (
    FormAssemblyResult,
    _information_matrix,
    _validate_blueprint,
    _validate_costs,
    assemble_form,
)
from mirt.cat.content import ContentBlueprint
from mirt.cat.selection import ItemSelectionStrategy, MaxFisherInformation

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


class ShadowTestSelection(ItemSelectionStrategy):
    """Shadow-test item selection (van der Linden & Reese, 1998).

    Before every selection, a full-length *shadow test* is assembled with
    :func:`~mirt.cat.assembly.assemble_form`. It contains every administered
    item, satisfies all constraints, and maximizes Fisher information at the
    current ability estimate. The most informative unadministered item of the
    shadow test is administered next, with ties going to the lowest index.

    Each shadow test is a feasible completion of the items already given, so
    a session that reaches ``test_length`` items satisfies every constraint
    exactly: content minima and maxima, enemy pairs, all-or-none bundles, and
    the cost budget. Sessions that stop earlier satisfy the maxima, enemy
    pairs, and budget, but may fall short of content minima and incomplete
    bundles. Without constraints, selection is identical to maximum Fisher
    information.

    Items removed by the engine's exposure control (or content control,
    although content belongs in the shadow test's ``blueprint``) are excluded
    from the shadow test. When that leaves no feasible shadow test, the
    exclusions are relaxed as in van der Linden and Veldkamp (2004): the
    shadow test is assembled from every unadministered item, and its most
    informative eligible free item is administered, or its most informative
    free item when none is eligible. Constraints therefore take precedence
    over exposure control. Progressive exposure control, which selects items
    without the strategy, is rejected by the engine.

    Shadow testing requires a unidimensional model. Batch simulation runs
    shadow-test sessions independently, never on the native or lock-step
    paths.

    Shadow tests that only content minima and maxima constrain (or nothing
    at all) are separable: the optimum takes each content area's most
    informative items up to its minimum, then the most informative
    remaining items within the maxima. Those shadow tests are built with
    this exact greedy allocation instead of the mixed-integer solver, which
    handles enemy pairs, bundles, and cost budgets. Ties between equally
    informative items may then resolve differently.

    Parameters
    ----------
    test_length : int, optional
        Number of items in every shadow test. By default the engine uses
        the length its stopping rules enforce (``max_items``, a
        ``MaxItemsStop`` in ``stopping_rule``, or else the pool size). The
        engine must stop by this length.
    blueprint : ContentBlueprint, optional
        Content-area minimum and maximum item counts.
    enemy_pairs : collection of tuple[int, int], optional
        Item pairs that may not both be administered.
    item_bundles : collection of collections of int, optional
        All-or-none item sets, such as items sharing a passage. Bundle
        members need not be administered consecutively.
    item_costs : array-like, optional
        Non-negative cost, for example expected response time, of every item.
    max_cost : float, optional
        Maximum total cost of a test.
    solver_options : mapping, optional
        Options forwarded to :func:`scipy.optimize.milp`, such as
        ``time_limit``. A feasible shadow test found within a limit is used.

    Attributes
    ----------
    last_shadow_test : FormAssemblyResult | None
        Shadow test assembled for the most recent selection.

    References
    ----------
    van der Linden, W. J., & Reese, L. M. (1998). A model for optimal
    constrained adaptive testing. Applied Psychological Measurement, 22(3),
    259-270.

    van der Linden, W. J., & Veldkamp, B. P. (2004). Constraining item
    exposure in computerized adaptive testing with shadow tests. Journal of
    Educational and Behavioral Statistics, 29(3), 273-291.

    Examples
    --------
    >>> blueprint = ContentBlueprint([
    ...     ContentArea("Algebra", items=set(range(50)), min_items=10, max_items=10),
    ...     ContentArea("Geometry", items=set(range(50, 100)), min_items=10,
    ...                 max_items=10),
    ... ])
    >>> selection = ShadowTestSelection(blueprint=blueprint, enemy_pairs={(3, 4)})
    >>> engine = CATEngine(model, item_selection=selection, max_items=20,
    ...                    min_items=20)
    """

    def __init__(
        self,
        test_length: int | None = None,
        *,
        blueprint: ContentBlueprint | None = None,
        enemy_pairs: Collection[tuple[int, int]] | None = None,
        item_bundles: Collection[Collection[int]] | None = None,
        item_costs: ArrayLike | None = None,
        max_cost: float | None = None,
        solver_options: Mapping[str, bool | int | float] | None = None,
    ) -> None:
        if test_length is not None and (
            isinstance(test_length, (bool, np.bool_))
            or not isinstance(test_length, Integral)
            or test_length < 1
        ):
            raise ValueError("test_length must be a positive integer or None")
        if blueprint is not None and not isinstance(blueprint, ContentBlueprint):
            raise TypeError("blueprint must be a ContentBlueprint")
        if solver_options is not None and not isinstance(solver_options, Mapping):
            raise TypeError("solver_options must be a mapping")

        self.test_length: int | None = None if test_length is None else int(test_length)
        self.blueprint = blueprint
        # Every selection reassembles the shadow test, so one-shot iterators
        # are stored as lists instead of being exhausted by the first one.
        self.enemy_pairs = (
            list(enemy_pairs) if isinstance(enemy_pairs, Iterator) else enemy_pairs
        )
        self.item_bundles = (
            list(item_bundles) if isinstance(item_bundles, Iterator) else item_bundles
        )
        self.item_costs = item_costs
        self.max_cost = max_cost
        self.solver_options = None if solver_options is None else dict(solver_options)
        self.last_shadow_test: FormAssemblyResult | None = None

    def select_item(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
        *,
        test_length: int | None = None,
    ) -> int:
        if not available_items:
            raise ValueError("No available items to select from")

        criteria = self.get_item_criteria(
            model,
            theta,
            available_items,
            administered_items,
            responses,
            test_length=test_length,
        )
        return max(criteria, key=criteria.__getitem__)

    def get_item_criteria(
        self,
        model: BaseItemModel,
        theta: float,
        available_items: set[int],
        administered_items: list[int] | None = None,
        responses: list[int] | None = None,
        *,
        test_length: int | None = None,
    ) -> dict[int, float]:
        """Return Fisher information of the free shadow-test items.

        Only unadministered items of the current shadow test are returned,
        restricted to ``available_items`` unless none of them is available.
        Ranking-based exposure control such as randomesque selection
        therefore stays within the shadow test. Items are returned in
        increasing index order.

        Raises
        ------
        RuntimeError
            If no shadow test is feasible or the shadow test is complete.
        """
        if not available_items:
            return {}
        given = set(administered_items or [])
        length = min(self.test_length or test_length or model.n_items, model.n_items)
        if len(given) >= length:
            raise RuntimeError(
                f"the {length}-item shadow test is complete; stop the test by "
                f"{length} items, for example with max_items"
            )

        pool = set(range(model.n_items))
        eligible = set(available_items) - given
        try:
            shadow = self._assemble(model, theta, length, eligible | given, given)
        except (RuntimeError, ValueError):
            if eligible | given == pool:
                raise
            # Relax exposure and content exclusions (van der Linden & Veldkamp).
            shadow = self._assemble(model, theta, length, pool, given)
        self.last_shadow_test = shadow

        free = set(shadow.selected_items.tolist()) - given
        return MaxFisherInformation().get_item_criteria(
            model, theta, free & eligible or free
        )

    def _is_separable(self) -> bool:
        """Whether only disjoint content-area counts constrain the shadow test."""
        if self.enemy_pairs or self.item_bundles or self.max_cost is not None:
            return False
        if self.blueprint is None:
            return True
        areas = [area.items for area in self.blueprint.areas]
        return sum(map(len, areas)) == len(set().union(*areas))

    def _assemble_greedy(
        self,
        model: BaseItemModel,
        theta: float,
        length: int,
        candidates: set[int],
        required: set[int],
    ) -> FormAssemblyResult:
        """Build the optimal separable shadow test by greedy allocation.

        The required items are kept; each content area then receives its most
        informative candidates up to its minimum, and the remaining places go
        to the most informative candidates whose area is below its maximum.
        Information is the separable objective, and each area's best items
        dominate its other items, so the allocation is optimal.
        """
        pool_size = model.n_items
        blueprint = _validate_blueprint(self.blueprint, pool_size)
        costs, _ = _validate_costs(self.item_costs, pool_size, None)
        if len(candidates) < length:
            raise RuntimeError(
                f"only {len(candidates)} candidate items for a {length}-item test"
            )
        indices = np.asarray(sorted(candidates), dtype=np.intp)
        theta_values = np.array([float(theta)])
        information = _information_matrix(model, theta_values, pool_size, indices)[0]
        utility = dict(zip(indices.tolist(), information.tolist(), strict=True))
        # Most informative first; ties go to the lowest index.
        ranked = sorted(set(candidates) - required, key=lambda j: (-utility[j], j))

        areas = blueprint.areas if blueprint is not None else []
        area_of = {item: area for area in areas for item in area.items}
        counts = {area.name: len(area.items & required) for area in areas}
        selected = set(required)
        for area in areas:
            if counts[area.name] > area.max_items:
                raise RuntimeError(
                    f"required items exceed the maximum of {area.name!r}"
                )
            needed = area.min_items - counts[area.name]
            if needed > 0:
                members = [item for item in ranked if area_of.get(item) is area]
                if len(members) < needed:
                    raise RuntimeError(f"too few candidate items for {area.name!r}")
                selected.update(members[:needed])
                counts[area.name] += needed
        if len(selected) > length:
            raise RuntimeError("content minima exceed the test length")
        for item in ranked:
            if len(selected) == length:
                break
            if item in selected:
                continue
            area = area_of.get(item)
            if area is not None:
                if counts[area.name] >= area.max_items:
                    continue
                counts[area.name] += 1
            selected.add(item)
        if len(selected) < length:
            raise RuntimeError("content maxima leave too few items for the test")

        items = np.asarray(sorted(selected), dtype=np.intp)
        total = float(sum(utility[int(item)] for item in items))
        return FormAssemblyResult(
            selected_items=items,
            theta=theta_values,
            information=np.array([total]),
            objective_value=total,
            method="maximize_information",
            content_counts={area.name: counts[area.name] for area in areas},
            total_cost=None if costs is None else float(np.sum(costs[items])),
            solver_message="optimal greedy allocation (separable constraints)",
        )

    def _assemble(
        self,
        model: BaseItemModel,
        theta: float,
        length: int,
        candidates: set[int],
        required: set[int],
    ) -> FormAssemblyResult:
        """Assemble one shadow test, naming the step when it is infeasible."""
        try:
            if self._is_separable():
                return self._assemble_greedy(model, theta, length, candidates, required)
            return assemble_form(
                model,
                length,
                [theta],
                blueprint=self.blueprint,
                candidate_items=candidates,
                required_items=required,
                enemy_pairs=self.enemy_pairs,
                item_bundles=self.item_bundles,
                item_costs=self.item_costs,
                max_cost=self.max_cost,
                solver_options=self.solver_options,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"shadow test assembly after {len(required)} administered items "
                f"failed for a {length}-item shadow test: {error}"
            ) from error
