"""Test equating and linking module for IRT models.

This module provides comprehensive IRT linking and score equating
functionality including:

- Linear linking methods (mean/sigma, mean/mean, Stocking-Lord, Haebara)
- Robust linking (bisector, orthogonal regression)
- Polytomous model linking (GRM, GPCM, NRM)
- Multidimensional linking (Procrustes rotation)
- Score equating (true score, observed score)
- Item parameter drift detection
- Chain linking across multiple time points

Examples
--------
Basic linking between two test forms:

>>> from mirt.equating import link
>>> result = link(model_old, model_new, [0, 1, 2], [0, 1, 2])
>>> print(f"A = {result.constants.A:.3f}, B = {result.constants.B:.3f}")

True score equating:

>>> from mirt.equating import true_score_equating
>>> eq_result = true_score_equating(model_old, model_new)
>>> print(eq_result.new_scores)

Drift detection:

>>> from mirt.equating import detect_drift
>>> drift = detect_drift(model_old, model_new, [0, 1, 2], [0, 1, 2])
>>> print(f"Flagged items: {drift.flagged_items}")
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "link": "mirt.equating.linking",
    "transform_parameters": "mirt.equating.linking",
    "LinkingConstants": "mirt.equating.linking",
    "LinkingResult": "mirt.equating.linking",
    "LinkingFitStatistics": "mirt.equating.linking",
    "AnchorDiagnostics": "mirt.equating.linking",
    "detect_drift": "mirt.equating.drift",
    "purify_anchors": "mirt.equating.drift",
    "signed_area_difference": "mirt.equating.drift",
    "DriftResult": "mirt.equating.drift",
    "bootstrap_linking_se": "mirt.equating.diagnostics",
    "delta_method_se": "mirt.equating.diagnostics",
    "compute_linking_fit": "mirt.equating.diagnostics",
    "linking_summary": "mirt.equating.diagnostics",
    "compare_linking_methods": "mirt.equating.diagnostics",
    "parameter_recovery_summary": "mirt.equating.diagnostics",
    "link_grm": "mirt.equating.polytomous",
    "link_gpcm": "mirt.equating.polytomous",
    "link_nrm": "mirt.equating.polytomous",
    "transform_polytomous_parameters": "mirt.equating.polytomous",
    "PolytomousLinkingResult": "mirt.equating.polytomous",
    "link_mirt": "mirt.equating.multidimensional",
    "orthogonal_procrustes_rotation": "mirt.equating.multidimensional",
    "oblique_procrustes_rotation": "mirt.equating.multidimensional",
    "transform_mirt_parameters": "mirt.equating.multidimensional",
    "transform_mirt_theta": "mirt.equating.multidimensional",
    "factor_congruence_coefficient": "mirt.equating.multidimensional",
    "match_factors": "mirt.equating.multidimensional",
    "compute_mirt_linking_fit": "mirt.equating.multidimensional",
    "target_rotation": "mirt.equating.multidimensional",
    "mirt_linking_summary": "mirt.equating.multidimensional",
    "ProcrustesResult": "mirt.equating.multidimensional",
    "true_score_equating": "mirt.equating.score_equating",
    "observed_score_equating": "mirt.equating.score_equating",
    "lord_wingersky_recursion": "mirt.equating.score_equating",
    "equipercentile_equating": "mirt.equating.score_equating",
    "score_to_theta": "mirt.equating.score_equating",
    "theta_to_score": "mirt.equating.score_equating",
    "score_equating_summary": "mirt.equating.score_equating",
    "compute_see": "mirt.equating.score_equating",
    "ScoreEquatingResult": "mirt.equating.score_equating",
    "chain_link": "mirt.equating.chain",
    "accumulate_constants": "mirt.equating.chain",
    "transform_to_reference": "mirt.equating.chain",
    "transform_theta_to_reference": "mirt.equating.chain",
    "concurrent_link": "mirt.equating.chain",
    "chain_linking_summary": "mirt.equating.chain",
    "detect_longitudinal_drift": "mirt.equating.chain",
    "ChainLinkingResult": "mirt.equating.chain",
    "TimePointModel": "mirt.equating.chain",
    "vertical_scale": "mirt.equating.vertical",
    "compute_vertical_diagnostics": "mirt.equating.vertical",
    "vertical_scale_summary": "mirt.equating.vertical",
    "plot_vertical_scale": "mirt.equating.vertical",
    "GradeData": "mirt.equating.vertical",
    "VerticalScaleResult": "mirt.equating.vertical",
    "VerticalScaleDiagnostics": "mirt.equating.vertical",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public equating symbol on first access."""
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
