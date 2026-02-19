"""Regression tests for documentation examples and symbol references."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mirt


def test_docs_do_not_reference_removed_model_aliases() -> None:
    """Documentation should only reference currently supported model symbols."""
    docs_files = (
        Path("docs/index.rst"),
        Path("docs/quickstart.rst"),
        Path("docs/api/index.rst"),
    )
    removed_symbols = ("Model1PL", "Model2PL", "Model3PL")

    for doc_file in docs_files:
        text = doc_file.read_text(encoding="utf-8")
        for symbol in removed_symbols:
            assert symbol not in text, f"{symbol} found in {doc_file}"


def test_api_reference_symbols_exist() -> None:
    """Symbols listed in docs/api/index.rst should resolve from top-level mirt."""
    documented_symbols = (
        "fit_mirt",
        "fscores",
        "itemfit",
        "personfit",
        "dif",
        "load_dataset",
        "list_datasets",
        "OneParameterLogistic",
        "TwoParameterLogistic",
        "ThreeParameterLogistic",
        "FourParameterLogistic",
        "GradedResponseModel",
        "GeneralizedPartialCredit",
        "PartialCreditModel",
        "NominalResponseModel",
        "EMEstimator",
        "MHRMEstimator",
        "GibbsSampler",
        "BLEstimator",
        "compute_fit_indices",
        "compare_models",
        "anova_irt",
        "compute_dtf",
        "compute_drf",
        "sibtest",
    )
    missing = [symbol for symbol in documented_symbols if not hasattr(mirt, symbol)]
    assert not missing, f"Undocumented or missing symbols: {missing}"


def test_quickstart_fit_and_score_smoke() -> None:
    """Core quickstart flow should run end-to-end on bundled sample data."""
    responses = mirt.load_dataset("LSAT7")["data"][:100]

    result = mirt.fit_mirt(
        responses,
        model="2PL",
        n_quadpts=11,
        max_iter=30,
        tol=1e-3,
    )
    scores = mirt.fscores(result, responses, method="EAP", n_quadpts=21)

    assert result.model.model_name == "2PL"
    assert scores.theta.shape[0] == responses.shape[0]
    assert np.all(np.isfinite(scores.theta))
