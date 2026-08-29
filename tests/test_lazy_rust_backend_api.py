"""Regression tests for the lightweight public Rust backend namespace."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

EXPECTED_EXPORTS = (
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
    "compute_wle_scores",
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
    "rt_joint_log_likelihood",
    "rt_accept_person_proposals",
    "bkt_forward",
    "bkt_backward",
    "bkt_forward_backward_batch",
    "bkt_viterbi",
    "bkt_ffbs",
    "bkt_ffbs_batch",
    "longitudinal_log_likelihood",
    "compute_growth_trajectory",
    "observed_score_distribution_2pl",
)


def _run_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_public_backend_export_order_is_unchanged() -> None:
    import mirt.backends.rust as rust_backend

    assert rust_backend.__all__ == list(EXPECTED_EXPORTS)


def test_plain_backend_import_defers_dependencies_and_modules() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.backends.rust as rust_backend

        print(json.dumps({
            "numpy_loaded": "numpy" in sys.modules,
            "native_loaded": "mirt.mirt_rs" in sys.modules,
            "backend_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.backends.rust.")
            ),
            "export_count": len(rust_backend.__all__),
            "exports_visible": all(
                name in dir(rust_backend) for name in rust_backend.__all__
            ),
        }))
        """
    )

    assert result == {
        "numpy_loaded": False,
        "native_loaded": False,
        "backend_submodules": [],
        "export_count": 76,
        "exports_visible": True,
    }


def test_backend_symbol_resolution_is_cached_and_scoped() -> None:
    result = _run_probe(
        """
        import json
        import sys

        import mirt.backends.rust as rust_backend

        deferred_before_access = "cat_conditional_mse" not in rust_backend.__dict__
        symbol = rust_backend.cat_conditional_mse

        from mirt.backends.rust.cat import cat_conditional_mse

        feature_modules = (
            "calibration",
            "cat",
            "diagnostics",
            "dynamic",
            "eapsum",
            "equating",
            "estep",
            "estimation",
            "gvem",
            "likelihood",
            "mstep",
            "multigroup",
            "plausible",
            "polytomous",
            "regularized",
            "response_time",
            "scoring",
            "simulation",
        )

        print(json.dumps({
            "deferred_before_access": deferred_before_access,
            "cached": rust_backend.__dict__["cat_conditional_mse"] is symbol,
            "same_symbol": symbol is cat_conditional_mse,
            "loaded_features": [
                name for name in feature_modules
                if f"mirt.backends.rust.{name}" in sys.modules
            ],
            "backend_submodules": sorted(
                name for name in sys.modules
                if name.startswith("mirt.backends.rust.")
            ),
        }))
        """
    )

    assert result == {
        "deferred_before_access": True,
        "cached": True,
        "same_symbol": True,
        "loaded_features": ["cat"],
        "backend_submodules": [
            "mirt.backends.rust._helpers",
            "mirt.backends.rust.cat",
        ],
    }


def test_star_import_resolves_every_public_backend_export() -> None:
    result = _run_probe(
        """
        import json

        import mirt.backends.rust as rust_backend

        namespace = {}
        exec("from mirt.backends.rust import *", namespace)
        exported = {name for name in namespace if not name.startswith("__")}

        print(json.dumps({
            "matches_all": exported == set(rust_backend.__all__),
            "export_count": len(exported),
            "all_cached": all(
                name in rust_backend.__dict__ for name in rust_backend.__all__
            ),
        }))
        """
    )

    assert result == {
        "matches_all": True,
        "export_count": 76,
        "all_cached": True,
    }


def test_submodule_import_fallback_and_unknown_attribute_error() -> None:
    result = _run_probe(
        """
        import json

        import mirt.backends.rust as rust_backend
        from mirt.backends.rust import simulation

        try:
            rust_backend.NotABackendFunction
        except AttributeError as error:
            message = str(error)
        else:
            message = None

        print(json.dumps({
            "submodule_name": simulation.__name__,
            "error": message,
        }))
        """
    )

    assert result == {
        "submodule_name": "mirt.backends.rust.simulation",
        "error": "module 'mirt.backends.rust' has no attribute 'NotABackendFunction'",
    }
