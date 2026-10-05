"""Shadow-test item selection for constrained adaptive testing."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from mirt.cat.assembly import FormAssemblyResult, assemble_form
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

    Parameters
    ----------
    test_length : int, optional
        Number of items in every shadow test. By default the engine's
        maximum test length (``max_items``, or the pool size) is used. The
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
