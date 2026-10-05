//! High-performance Rust backend for MIRT (Multidimensional Item Response Theory).
//!
//! This crate provides optimized implementations of IRT algorithms including:
//! - Log-likelihood computations for 2PL, 3PL, and multidimensional IRT models
//! - E-step and M-step algorithms for EM estimation
//! - SIBTEST for differential item functioning analysis
//! - Response simulation for various IRT models
//! - Plausible values generation
//! - Parameter estimation (EM, Gibbs, MHRM)
//! - Diagnostic statistics (Q3, LD chi-square, fit statistics)
//! - Person scoring (EAP, WLE)
//! - Bootstrap and imputation methods
//! - CAT (Computerized Adaptive Testing) functions
//! - EAPsum scoring with Lord-Wingersky recursion

use pyo3::prelude::*;

mod bayesian_diagnostics;
mod bootstrap;
mod calibration;
mod cat;
mod counts;
mod diagnostics;
mod dynamic;
mod eapsum;
mod equating;
mod estep;
mod estimation;
mod explanatory;
mod gvem;
mod likelihood;
mod likelihood_cache;
mod mstep;
mod multigroup;
mod optimization_scoring;
mod patterns;
mod plausible;
mod polytomous;
mod polytomous_mstep;
mod posterior;
mod regularized;
mod response_time;
mod scoring;
mod sibtest;
mod simulation;
mod special;
mod standard_errors;
mod utils;

/// Python module for mirt_rs
#[pymodule]
fn mirt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bayesian_diagnostics::register(m)?;
    likelihood::register(m)?;
    estep::register(m)?;
    sibtest::register(m)?;
    simulation::register(m)?;
    plausible::register(m)?;
    estimation::register(m)?;
    diagnostics::register(m)?;
    scoring::register(m)?;
    optimization_scoring::register(m)?;
    bootstrap::register(m)?;
    eapsum::register(m)?;
    cat::register(m)?;
    mstep::register(m)?;
    polytomous_mstep::register(m)?;
    standard_errors::register(m)?;
    calibration::register(m)?;
    polytomous::register(m)?;
    multigroup::register(m)?;
    gvem::register(m)?;
    regularized::register(m)?;
    response_time::register(m)?;
    dynamic::register(m)?;
    explanatory::register(m)?;
    equating::register(m)?;
    patterns::register(m)?;
    posterior::register(m)?;

    Ok(())
}
