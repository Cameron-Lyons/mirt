"""Python interface to the Rust backend for MIRT.

This package wraps the ``mirt.mirt_rs`` extension. Each submodule declares
``FALLBACK_MODE`` as ``\"numpy\"``, ``\"optional\"``, ``\"required\"``, or
``\"mixed\"`` — see :mod:`mirt.backends.rust._helpers` for the contract.

Global ``mirt.set_backend(\"numpy\")`` disables Rust dispatch even when the
extension is installed.
"""

from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    PreparedArrays,
    is_rust_available,
    rust_enabled,
)
from mirt.backends.rust.calibration import (
    fixed_calib_em,
    stocking_lord_criterion,
)
from mirt.backends.rust.cat import (
    cat_compute_item_info,
    cat_conditional_mse,
    cat_eap_update,
    cat_select_max_info,
    cat_simulate_batch,
)
from mirt.backends.rust.diagnostics import (
    compute_expected_variance_batch,
    compute_fit_statistics,
    compute_hessian_block_diagonal,
    compute_item_se_parallel,
    compute_ld_chi2_matrix,
    compute_probabilities_batch,
    compute_probabilities_batch_3pl,
    compute_q3_matrix,
    compute_standardized_residuals,
    sibtest_all_items,
    sibtest_compute_beta,
)
from mirt.backends.rust.dynamic import (
    bkt_backward,
    bkt_ffbs,
    bkt_ffbs_batch,
    bkt_forward,
    bkt_forward_backward_batch,
    bkt_viterbi,
    compute_growth_trajectory,
    longitudinal_log_likelihood,
)
from mirt.backends.rust.eapsum import (
    eapsum_from_distribution,
    lord_wingersky_polytomous,
    lord_wingersky_recursion,
)
from mirt.backends.rust.estep import (
    compute_expected_counts,
    compute_expected_counts_parallel,
    compute_expected_counts_polytomous,
    e_step_complete,
)
from mirt.backends.rust.estimation import (
    bootstrap_fit_2pl,
    em_fit_2pl,
    em_iteration_2pl,
    em_iteration_3pl,
    gibbs_sample_2pl,
    mhrm_fit_2pl,
)
from mirt.backends.rust.gvem import (
    gvem_compute_elbo,
    gvem_e_step,
    gvem_m_step,
)
from mirt.backends.rust.likelihood import (
    compute_log_likelihoods_2pl,
    compute_log_likelihoods_3pl,
    compute_log_likelihoods_mirt,
)
from mirt.backends.rust.mstep import (
    m_step_dichotomous_parallel,
)
from mirt.backends.rust.multigroup import (
    multigroup_e_step_2pl,
    multigroup_e_step_3pl,
    multigroup_e_step_gpcm,
    multigroup_e_step_grm,
    multigroup_e_step_nrm,
    multigroup_expected_counts,
)
from mirt.backends.rust.plausible import (
    compute_expected_margins,
    compute_observed_margins,
    generate_bootstrap_indices,
    generate_plausible_values_mcmc,
    generate_plausible_values_posterior,
    impute_from_probabilities,
    multiple_imputation,
    resample_responses,
)
from mirt.backends.rust.polytomous import (
    compute_alpha_if_deleted,
    compute_log_likelihoods_gpcm,
    compute_log_likelihoods_grm,
)
from mirt.backends.rust.regularized import (
    coordinate_descent_mstep_regularized,
)
from mirt.backends.rust.scoring import (
    compute_eap_scores,
)
from mirt.backends.rust.simulation import (
    simulate_dichotomous,
    simulate_gpcm,
    simulate_grm,
)

__all__ = [
    "PreparedArrays",
    "RUST_AVAILABLE",
    "is_rust_available",
    "rust_enabled",
    "compute_log_likelihoods_2pl",
    "compute_log_likelihoods_3pl",
    "compute_log_likelihoods_mirt",
    "e_step_complete",
    "compute_expected_counts",
    "compute_expected_counts_polytomous",
    "compute_expected_counts_parallel",
    "compute_eap_scores",
    "em_fit_2pl",
    "gibbs_sample_2pl",
    "mhrm_fit_2pl",
    "bootstrap_fit_2pl",
    "em_iteration_2pl",
    "em_iteration_3pl",
    "m_step_dichotomous_parallel",
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
    "simulate_grm",
    "simulate_gpcm",
    "simulate_dichotomous",
    "generate_plausible_values_posterior",
    "generate_plausible_values_mcmc",
    "compute_observed_margins",
    "compute_expected_margins",
    "generate_bootstrap_indices",
    "resample_responses",
    "impute_from_probabilities",
    "multiple_imputation",
    "cat_compute_item_info",
    "cat_select_max_info",
    "cat_eap_update",
    "cat_simulate_batch",
    "cat_conditional_mse",
    "lord_wingersky_recursion",
    "lord_wingersky_polytomous",
    "eapsum_from_distribution",
    "fixed_calib_em",
    "stocking_lord_criterion",
    "compute_log_likelihoods_grm",
    "compute_log_likelihoods_gpcm",
    "compute_alpha_if_deleted",
    "multigroup_e_step_grm",
    "multigroup_e_step_gpcm",
    "multigroup_e_step_2pl",
    "multigroup_e_step_3pl",
    "multigroup_e_step_nrm",
    "multigroup_expected_counts",
    "gvem_e_step",
    "gvem_m_step",
    "gvem_compute_elbo",
    "coordinate_descent_mstep_regularized",
    "bkt_forward",
    "bkt_backward",
    "bkt_forward_backward_batch",
    "bkt_viterbi",
    "bkt_ffbs",
    "bkt_ffbs_batch",
    "longitudinal_log_likelihood",
    "compute_growth_trajectory",
]
