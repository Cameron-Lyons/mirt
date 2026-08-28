"""Computerized Adaptive Testing (CAT) module for mirt.

This module provides comprehensive CAT functionality including:
- Item selection strategies (MFI, MEI, KL, etc.)
- Stopping rules (SE threshold, max items, etc.)
- Exposure control (Sympson-Hetter, randomesque)
- Content balancing (test blueprints)
- CAT engine for orchestrating adaptive tests
- Multidimensional CAT (MCAT) support

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

Multidimensional CAT:

>>> from mirt import fit_mirt
>>> from mirt.cat import MCATEngine
>>> result = fit_mirt(data, model="MIRT", n_factors=2)
>>> mcat = MCATEngine(result.model, trace_threshold=0.5, max_items=30)
>>> state = mcat.get_current_state()
>>> while not state.is_complete:
...     response = get_response(state.next_item)
...     state = mcat.administer_item(response)
>>> print(mcat.get_result().summary())
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "CATEngine": "mirt.cat.engine",
    "CATResult": "mirt.cat.results",
    "CATState": "mirt.cat.results",
    "ItemSelectionStrategy": "mirt.cat.selection",
    "MaxFisherInformation": "mirt.cat.selection",
    "MaxExpectedInformation": "mirt.cat.selection",
    "KullbackLeibler": "mirt.cat.selection",
    "UrryRule": "mirt.cat.selection",
    "RandomSelection": "mirt.cat.selection",
    "AStratified": "mirt.cat.selection",
    "create_selection_strategy": "mirt.cat.selection",
    "StoppingRule": "mirt.cat.stopping",
    "StandardErrorStop": "mirt.cat.stopping",
    "MaxItemsStop": "mirt.cat.stopping",
    "MinItemsStop": "mirt.cat.stopping",
    "ThetaChangeStop": "mirt.cat.stopping",
    "ClassificationStop": "mirt.cat.stopping",
    "CombinedStop": "mirt.cat.stopping",
    "create_stopping_rule": "mirt.cat.stopping",
    "MCATEngine": "mirt.cat.mcat_engine",
    "MCATResult": "mirt.cat.results",
    "MCATState": "mirt.cat.results",
    "MCATSelectionStrategy": "mirt.cat.mcat_selection",
    "DOptimality": "mirt.cat.mcat_selection",
    "AOptimality": "mirt.cat.mcat_selection",
    "COptimality": "mirt.cat.mcat_selection",
    "KullbackLeiblerMCAT": "mirt.cat.mcat_selection",
    "BayesianMCAT": "mirt.cat.mcat_selection",
    "RandomMCATSelection": "mirt.cat.mcat_selection",
    "create_mcat_selection_strategy": "mirt.cat.mcat_selection",
    "MCATStoppingRule": "mirt.cat.mcat_stopping",
    "CovarianceTraceStop": "mirt.cat.mcat_stopping",
    "CovarianceDeterminantStop": "mirt.cat.mcat_stopping",
    "MaxSEStop": "mirt.cat.mcat_stopping",
    "AvgSEStop": "mirt.cat.mcat_stopping",
    "MaxItemsMCATStop": "mirt.cat.mcat_stopping",
    "ThetaChangeMCATStop": "mirt.cat.mcat_stopping",
    "CombinedMCATStop": "mirt.cat.mcat_stopping",
    "create_mcat_stopping_rule": "mirt.cat.mcat_stopping",
    "ExposureControl": "mirt.cat.exposure",
    "NoExposureControl": "mirt.cat.exposure",
    "SympsonHetter": "mirt.cat.exposure",
    "Randomesque": "mirt.cat.exposure",
    "ProgressiveRestricted": "mirt.cat.exposure",
    "create_exposure_control": "mirt.cat.exposure",
    "ContentConstraint": "mirt.cat.content",
    "NoContentConstraint": "mirt.cat.content",
    "ContentBlueprint": "mirt.cat.content",
    "ContentArea": "mirt.cat.content",
    "WeightedContent": "mirt.cat.content",
    "create_content_constraint": "mirt.cat.content",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public CAT symbol on first access."""
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return loaded attributes together with deferred public symbols."""
    return sorted(set(globals()) | set(__all__))
