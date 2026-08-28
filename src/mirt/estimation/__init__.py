"""Public estimation API with on-demand imports.

Estimator implementations and numerical dependencies remain deferred until a
public symbol is accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "BaseEstimator": "mirt.estimation.base",
    "EMEstimator": "mirt.estimation.em",
    "GVEMEstimator": "mirt.estimation.gvem",
    "IRTreeEMEstimator": "mirt.estimation.irtree_em",
    "IRTreeResult": "mirt.estimation.irtree_em",
    "GaussHermiteQuadrature": "mirt.estimation.quadrature",
    "MCEMEstimator": "mirt.estimation.mcem",
    "QMCEMEstimator": "mirt.estimation.mcem",
    "SparseBayesianEstimator": "mirt.estimation.sparse_bayesian",
    "SparseBayesianResult": "mirt.estimation.sparse_bayesian",
    "SpikeSlabLassoPrior": "mirt.estimation.sparse_bayesian",
    "StochasticEMEstimator": "mirt.estimation.mcem",
    "WeightedEMEstimator": "mirt.estimation.weighted",
    "compute_effective_sample_size": "mirt.estimation.weighted",
    "compute_design_effect": "mirt.estimation.weighted",
    "compute_observed_information": "mirt.estimation.standard_errors",
    "compute_sandwich_se": "mirt.estimation.standard_errors",
    "compute_oakes_se": "mirt.estimation.standard_errors",
    "compute_sem_se": "mirt.estimation.standard_errors",
    "Prior": "mirt.estimation.priors",
    "NormalPrior": "mirt.estimation.priors",
    "TruncatedNormalPrior": "mirt.estimation.priors",
    "LogNormalPrior": "mirt.estimation.priors",
    "BetaPrior": "mirt.estimation.priors",
    "UniformPrior": "mirt.estimation.priors",
    "GammaPrior": "mirt.estimation.priors",
    "CustomPrior": "mirt.estimation.priors",
    "PriorSpecification": "mirt.estimation.priors",
    "default_priors": "mirt.estimation.priors",
    "weakly_informative_priors": "mirt.estimation.priors",
    "compute_prior_log_pdf": "mirt.estimation.priors",
    "LatentDensity": "mirt.estimation.latent_density",
    "GaussianDensity": "mirt.estimation.latent_density",
    "EmpiricalHistogram": "mirt.estimation.latent_density",
    "DavidianCurve": "mirt.estimation.latent_density",
    "MixtureDensity": "mirt.estimation.latent_density",
    "CustomDensity": "mirt.estimation.latent_density",
    "create_density": "mirt.estimation.latent_density",
    "RTModelPriors": "mirt.estimation.rt_gibbs",
    "ResponseTimeGibbsSampler": "mirt.estimation.rt_gibbs",
    "ParameterConstraint": "mirt.estimation.constraints",
    "FixedConstraint": "mirt.estimation.constraints",
    "EqualityConstraint": "mirt.estimation.constraints",
    "BoundConstraint": "mirt.estimation.constraints",
    "LinearConstraint": "mirt.estimation.constraints",
    "CustomConstraint": "mirt.estimation.constraints",
    "ConstraintSet": "mirt.estimation.constraints",
    "fix_parameter": "mirt.estimation.constraints",
    "equal_parameters": "mirt.estimation.constraints",
    "bound_parameter": "mirt.estimation.constraints",
    "mean_constraint": "mirt.estimation.constraints",
    "create_1pl_constraints": "mirt.estimation.constraints",
    "create_rasch_constraints": "mirt.estimation.constraints",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public estimation symbol on first access."""
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
