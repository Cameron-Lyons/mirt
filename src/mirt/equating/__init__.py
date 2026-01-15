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

from mirt.equating.chain import (
    ChainLinkingResult,
    TimePointModel,
    accumulate_constants,
    chain_link,
    chain_linking_summary,
    concurrent_link,
    detect_longitudinal_drift,
    transform_theta_to_reference,
    transform_to_reference,
)
from mirt.equating.diagnostics import (
    bootstrap_linking_se,
    compare_linking_methods,
    compute_linking_fit,
    delta_method_se,
    linking_summary,
    parameter_recovery_summary,
)
from mirt.equating.drift import (
    DriftResult,
    detect_drift,
    purify_anchors,
    signed_area_difference,
)
from mirt.equating.linking import (
    AnchorDiagnostics,
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
    link,
    transform_parameters,
)
from mirt.equating.multidimensional import (
    ProcrustesResult,
    compute_mirt_linking_fit,
    factor_congruence_coefficient,
    link_mirt,
    match_factors,
    mirt_linking_summary,
    oblique_procrustes_rotation,
    orthogonal_procrustes_rotation,
    target_rotation,
    transform_mirt_parameters,
)
from mirt.equating.polytomous import (
    PolytomousLinkingResult,
    link_gpcm,
    link_grm,
    link_nrm,
    transform_polytomous_parameters,
)
from mirt.equating.score_equating import (
    ScoreEquatingResult,
    compute_see,
    equipercentile_equating,
    lord_wingersky_recursion,
    observed_score_equating,
    score_equating_summary,
    score_to_theta,
    theta_to_score,
    true_score_equating,
)

__all__ = [
    # Core linking
    "link",
    "transform_parameters",
    "LinkingConstants",
    "LinkingResult",
    "LinkingFitStatistics",
    "AnchorDiagnostics",
    # Drift detection
    "detect_drift",
    "purify_anchors",
    "signed_area_difference",
    "DriftResult",
    # Diagnostics
    "bootstrap_linking_se",
    "delta_method_se",
    "compute_linking_fit",
    "linking_summary",
    "compare_linking_methods",
    "parameter_recovery_summary",
    # Polytomous linking
    "link_grm",
    "link_gpcm",
    "link_nrm",
    "transform_polytomous_parameters",
    "PolytomousLinkingResult",
    # Multidimensional linking
    "link_mirt",
    "orthogonal_procrustes_rotation",
    "oblique_procrustes_rotation",
    "transform_mirt_parameters",
    "factor_congruence_coefficient",
    "match_factors",
    "compute_mirt_linking_fit",
    "target_rotation",
    "mirt_linking_summary",
    "ProcrustesResult",
    # Score equating
    "true_score_equating",
    "observed_score_equating",
    "lord_wingersky_recursion",
    "equipercentile_equating",
    "score_to_theta",
    "theta_to_score",
    "score_equating_summary",
    "compute_see",
    "ScoreEquatingResult",
    # Chain linking
    "chain_link",
    "accumulate_constants",
    "transform_to_reference",
    "transform_theta_to_reference",
    "concurrent_link",
    "chain_linking_summary",
    "detect_longitudinal_drift",
    "ChainLinkingResult",
    "TimePointModel",
]
