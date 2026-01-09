"""Computerized Adaptive Testing (CAT) module for mirt.

This module provides comprehensive CAT functionality including:
- Item selection strategies (MFI, MEI, KL, etc.)
- Stopping rules (SE threshold, max items, etc.)
- Exposure control (Sympson-Hetter, randomesque)
- Content balancing (test blueprints)
- CAT engine for orchestrating adaptive tests

Examples
--------
Basic CAT session:

>>> from mirt import fit_mirt
>>> from mirt.cat import CATEngine
>>> result = fit_mirt(data, model="2PL")
>>> cat = CATEngine(result.model, se_threshold=0.3, max_items=20)
>>> state = cat.get_current_state()
>>> while not state.is_complete:
...     # Display item state.next_item to examinee
...     response = get_response()  # 0 or 1
...     state = cat.administer_item(response)
>>> print(cat.get_result().summary())

Simulation study:

>>> cat = CATEngine(model, se_threshold=0.3)
>>> results = cat.run_batch_simulation(
...     true_thetas=np.linspace(-2, 2, 11),
...     n_replications=100,
... )

With content balancing:

>>> from mirt.cat import CATEngine, ContentBlueprint, ContentArea
>>> blueprint = ContentBlueprint([
...     ContentArea("Algebra", items={0, 1, 2, 3}, min_items=2),
...     ContentArea("Geometry", items={4, 5, 6}, min_items=1),
... ])
>>> cat = CATEngine(model, content_constraint=blueprint)
"""

from mirt.cat.content import (
    ContentArea,
    ContentBlueprint,
    ContentConstraint,
    NoContentConstraint,
    WeightedContent,
    create_content_constraint,
)
from mirt.cat.engine import CATEngine
from mirt.cat.exposure import (
    ExposureControl,
    NoExposureControl,
    ProgressiveRestricted,
    Randomesque,
    SympsonHetter,
    create_exposure_control,
)
from mirt.cat.results import CATResult, CATState
from mirt.cat.selection import (
    AStratified,
    ItemSelectionStrategy,
    KullbackLeibler,
    MaxExpectedInformation,
    MaxFisherInformation,
    RandomSelection,
    UrryRule,
    create_selection_strategy,
)
from mirt.cat.stopping import (
    ClassificationStop,
    CombinedStop,
    MaxItemsStop,
    MinItemsStop,
    StandardErrorStop,
    StoppingRule,
    ThetaChangeStop,
    create_stopping_rule,
)

__all__ = [
    "CATEngine",
    "CATResult",
    "CATState",
    "ItemSelectionStrategy",
    "MaxFisherInformation",
    "MaxExpectedInformation",
    "KullbackLeibler",
    "UrryRule",
    "RandomSelection",
    "AStratified",
    "create_selection_strategy",
    "StoppingRule",
    "StandardErrorStop",
    "MaxItemsStop",
    "MinItemsStop",
    "ThetaChangeStop",
    "ClassificationStop",
    "CombinedStop",
    "create_stopping_rule",
    "ExposureControl",
    "NoExposureControl",
    "SympsonHetter",
    "Randomesque",
    "ProgressiveRestricted",
    "create_exposure_control",
    "ContentConstraint",
    "NoContentConstraint",
    "ContentBlueprint",
    "ContentArea",
    "WeightedContent",
    "create_content_constraint",
]
