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
    "_helpers": ("RUST_AVAILABLE", "is_rust_available", "rust_enabled"),
    "estimation": (
        "mhrm_fit_2pl",
        "bootstrap_fit_2pl",
        "em_iteration_3pl",
    ),
    "simulation": ("simulate_grm", "simulate_gpcm", "simulate_dichotomous"),
    "cat": ("cat_simulate_batch_full", "cat_conditional_mse"),
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
    ),
    "regularized": ("coordinate_descent_mstep_regularized",),
    "response_time": ("rt_joint_log_likelihood", "rt_accept_person_proposals"),
    "dynamic": (
        "bkt_forward",
        "bkt_backward",
        "bkt_forward_backward_batch",
        "bkt_viterbi",
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
