import importlib
from collections.abc import Sequence
from typing import Any

__all__ = [
    "compute_itemfit",
    "compute_s_x2",
    "compute_personfit",
    "compute_personfit_significance",
    "compute_dif",
    "compute_grdif",
    "compute_pairwise_rdif",
    "grdif_effect_size",
    "flag_dif_items",
    "adjust_p_values",
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
    "posterior_predictive_checks",
    "compute_pointwise_log_lik",
    "compare_models",
    "loo_compare",
    "PSISResult",
    "WAICResult",
    "PPCResult",
]

_LAZY_IMPORTS = {
    "compute_itemfit": ("mirt.diagnostics.itemfit", "compute_itemfit"),
    "compute_s_x2": ("mirt.diagnostics.itemfit", "compute_s_x2"),
    "compute_personfit": ("mirt.diagnostics.personfit", "compute_personfit"),
    "compute_personfit_significance": (
        "mirt.diagnostics.personfit",
        "compute_personfit_significance",
    ),
    "compute_dif": ("mirt.diagnostics.dif", "compute_dif"),
    "compute_grdif": ("mirt.diagnostics.dif", "compute_grdif"),
    "compute_pairwise_rdif": ("mirt.diagnostics.dif", "compute_pairwise_rdif"),
    "grdif_effect_size": ("mirt.diagnostics.dif", "grdif_effect_size"),
    "flag_dif_items": ("mirt.diagnostics.dif", "flag_dif_items"),
    "adjust_p_values": (
        "mirt.diagnostics.multiple_testing",
        "adjust_p_values",
    ),
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
    "posterior_predictive_checks": (
        "mirt.diagnostics.bayesian",
        "posterior_predictive_checks",
    ),
    "compute_pointwise_log_lik": (
        "mirt.diagnostics.bayesian",
        "compute_pointwise_log_lik",
    ),
    "loo_compare": ("mirt.diagnostics.bayesian", "loo_compare"),
    "PSISResult": ("mirt.diagnostics.bayesian", "PSISResult"),
    "WAICResult": ("mirt.diagnostics.bayesian", "WAICResult"),
    "PPCResult": ("mirt.diagnostics.bayesian", "PPCResult"),
}


def compare_models(
    results: Any,
    *more_results: Any,
    criteria: Sequence[str] | None = None,
    names: list[str] | None = None,
) -> Any:
    """Compare fitted models by information criteria.

    This is :func:`mirt.compare_models`: ``results`` is a list of
    ``FitResult`` objects and ``criteria`` names the information criteria.

    .. deprecated::
        Passing ``PSISResult`` or ``WAICResult`` objects, as the earlier
        ``mirt.diagnostics.compare_models`` did, still returns its formatted
        table but warns. Use :func:`loo_compare` for them, which returns a
        DataFrame with the standard errors of the elpd differences.

    Parameters
    ----------
    results : list of FitResult
        Fitted model results to compare.
    criteria : list of str, optional
        Information criteria, as in :func:`mirt.compare_models`. It may also
        be passed positionally after ``results``.
    names : list of str, optional
        Model names of the deprecated PSIS-LOO and WAIC table.

    Returns
    -------
    DataFrame or str
        The information-criterion table, or the deprecated formatted table.
    """
    from mirt.diagnostics.bayesian import (
        PSISResult,
        WAICResult,
        _loo_comparison_text,
        _warn_loo_table_deprecated,
    )
    from mirt.diagnostics.comparison import compare_models as compare_fits

    if isinstance(results, (PSISResult, WAICResult)):
        if criteria is not None:
            raise TypeError("criteria apply only to FitResult comparisons")
        _warn_loo_table_deprecated(stacklevel=3)
        return _loo_comparison_text((results, *more_results), names)
    if names is not None:
        raise TypeError("names apply only to PSIS-LOO and WAIC results")
    if more_results:
        if len(more_results) > 1 or criteria is not None:
            raise TypeError(
                "compare_models takes a list of FitResult objects and criteria"
            )
        criteria = more_results[0]
    return compare_fits(results, criteria)


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
