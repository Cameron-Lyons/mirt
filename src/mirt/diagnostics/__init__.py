import importlib
from typing import Any

__all__ = [
    "compute_itemfit",
    "compute_personfit",
    "compute_dif",
    "compute_grdif",
    "compute_pairwise_rdif",
    "grdif_effect_size",
    "flag_dif_items",
    "compute_ld_statistics",
    "compute_q3",
    "compute_ld_chi2",
    "flag_ld_pairs",
    "ld_summary_table",
    "LDResult",
    "compute_residuals",
    "analyze_residuals",
    "compute_outfit_infit",
    "identify_misfitting_patterns",
    "ResidualAnalysisResult",
    "psis_loo",
    "waic",
    "dic",
    "posterior_predictive_check",
    "compute_pointwise_log_lik",
    "compare_models",
    "PSISResult",
    "WAICResult",
    "PPCResult",
]

_LAZY_IMPORTS = {
    "compute_itemfit": ("mirt.diagnostics.itemfit", "compute_itemfit"),
    "compute_personfit": ("mirt.diagnostics.personfit", "compute_personfit"),
    "compute_dif": ("mirt.diagnostics.dif", "compute_dif"),
    "compute_grdif": ("mirt.diagnostics.dif", "compute_grdif"),
    "compute_pairwise_rdif": ("mirt.diagnostics.dif", "compute_pairwise_rdif"),
    "grdif_effect_size": ("mirt.diagnostics.dif", "grdif_effect_size"),
    "flag_dif_items": ("mirt.diagnostics.dif", "flag_dif_items"),
    "compute_ld_statistics": ("mirt.diagnostics.ld", "compute_ld_statistics"),
    "compute_q3": ("mirt.diagnostics.ld", "compute_q3"),
    "compute_ld_chi2": ("mirt.diagnostics.ld", "compute_ld_chi2"),
    "flag_ld_pairs": ("mirt.diagnostics.ld", "flag_ld_pairs"),
    "ld_summary_table": ("mirt.diagnostics.ld", "ld_summary_table"),
    "LDResult": ("mirt.diagnostics.ld", "LDResult"),
    "compute_residuals": ("mirt.diagnostics.residuals", "compute_residuals"),
    "analyze_residuals": ("mirt.diagnostics.residuals", "analyze_residuals"),
    "compute_outfit_infit": ("mirt.diagnostics.residuals", "compute_outfit_infit"),
    "identify_misfitting_patterns": (
        "mirt.diagnostics.residuals",
        "identify_misfitting_patterns",
    ),
    "ResidualAnalysisResult": (
        "mirt.diagnostics.residuals",
        "ResidualAnalysisResult",
    ),
    "psis_loo": ("mirt.diagnostics.bayesian", "psis_loo"),
    "waic": ("mirt.diagnostics.bayesian", "waic"),
    "dic": ("mirt.diagnostics.bayesian", "dic"),
    "posterior_predictive_check": (
        "mirt.diagnostics.bayesian",
        "posterior_predictive_check",
    ),
    "compute_pointwise_log_lik": ("mirt.diagnostics.bayesian", "compute_pointwise_log_lik"),
    "compare_models": ("mirt.diagnostics.bayesian", "compare_models"),
    "PSISResult": ("mirt.diagnostics.bayesian", "PSISResult"),
    "WAICResult": ("mirt.diagnostics.bayesian", "WAICResult"),
    "PPCResult": ("mirt.diagnostics.bayesian", "PPCResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mirt.diagnostics' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
