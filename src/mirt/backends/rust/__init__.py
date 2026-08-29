"""Python interface to the Rust backend for MIRT.

This package wraps the ``mirt.mirt_rs`` extension. Each submodule declares
``FALLBACK_MODE`` as ``"numpy"``, ``"optional"``, ``"required"``, or
``"mixed"`` — see :mod:`mirt.backends.rust._helpers` for the contract.

Global ``mirt.set_backend("numpy")`` disables Rust dispatch even when the
extension is installed.
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "_helpers": (
        "PreparedArrays",
        "RUST_AVAILABLE",
        "is_rust_available",
        "rust_enabled",
    ),
    "likelihood": (
        "compute_log_likelihoods_2pl",
        "compute_log_likelihoods_3pl",
        "compute_log_likelihoods_mirt",
    ),
    "estep": (
        "e_step_complete",
        "compute_expected_counts",
        "compute_expected_counts_polytomous",
        "compute_expected_counts_parallel",
    ),
    "scoring": ("compute_eap_scores", "compute_wle_scores"),
    "estimation": (
        "em_fit_2pl",
        "gibbs_sample_2pl",
        "mhrm_fit_2pl",
        "bootstrap_fit_2pl",
        "em_iteration_2pl",
        "em_iteration_3pl",
    ),
    "mstep": ("m_step_dichotomous_parallel",),
    "diagnostics": (
        "sibtest_compute_beta",
        "sibtest_all_items",
        "compute_standardized_residuals",
        "compute_q3_matrix",
        "compute_ld_chi2_matrix",
        "compute_item_se_parallel",
        "compute_hessian_block_diagonal",
        "compute_fit_statistics",
        "compute_probabilities_batch",
        "compute_probabilities_batch_3pl",
        "compute_expected_variance_batch",
    ),
    "simulation": (
        "simulate_grm",
        "simulate_gpcm",
        "simulate_dichotomous",
    ),
    "plausible": (
        "generate_plausible_values_posterior",
        "generate_plausible_values_mcmc",
        "compute_observed_margins",
        "compute_expected_margins",
        "generate_bootstrap_indices",
        "resample_responses",
        "impute_from_probabilities",
        "multiple_imputation",
    ),
    "cat": (
        "cat_compute_item_info",
        "cat_select_max_info",
        "cat_eap_update",
        "cat_simulate_batch",
        "cat_simulate_batch_full",
        "cat_conditional_mse",
    ),
    "eapsum": (
        "lord_wingersky_recursion",
        "lord_wingersky_polytomous",
        "eapsum_from_distribution",
    ),
    "calibration": (
        "fixed_calib_em",
        "stocking_lord_criterion",
    ),
    "polytomous": (
        "compute_log_likelihoods_grm",
        "compute_log_likelihoods_gpcm",
        "compute_alpha_if_deleted",
    ),
    "multigroup": (
        "multigroup_e_step_grm",
        "multigroup_e_step_gpcm",
        "multigroup_e_step_2pl",
        "multigroup_e_step_3pl",
        "multigroup_e_step_nrm",
        "multigroup_expected_counts",
    ),
    "gvem": (
        "gvem_e_step",
        "gvem_m_step",
        "gvem_compute_elbo",
    ),
    "regularized": ("coordinate_descent_mstep_regularized",),
    "response_time": (
        "rt_joint_log_likelihood",
        "rt_accept_person_proposals",
    ),
    "dynamic": (
        "bkt_forward",
        "bkt_backward",
        "bkt_forward_backward_batch",
        "bkt_viterbi",
        "bkt_ffbs",
        "bkt_ffbs_batch",
        "longitudinal_log_likelihood",
        "compute_growth_trajectory",
    ),
    "equating": ("observed_score_distribution_2pl",),
}

_LAZY_IMPORTS = {
    symbol: (f"mirt.backends.rust.{module_name}", symbol)
    for module_name, symbols in _MODULE_EXPORTS.items()
    for symbol in symbols
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.backends.rust' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
