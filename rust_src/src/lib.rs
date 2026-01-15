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

pub mod utils;

pub mod bayesian_diagnostics;
pub mod bootstrap;
pub mod calibration;
pub mod cat;
pub mod diagnostics;
pub mod dynamic;
pub mod eapsum;
pub mod equating;
pub mod estep;
pub mod estimation;
pub mod explanatory;
pub mod gvem;
pub mod irtree;
pub mod likelihood;
pub mod mfrm;
pub mod mirt_models;
pub mod mstep;
pub mod multigroup;
pub mod multilevel;
pub mod plausible;
pub mod polytomous;
pub mod regularized;
pub mod response_time;
pub mod scoring;
pub mod sibtest;
pub mod simulation;
pub mod standard_errors;

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
    bootstrap::register(m)?;
    mirt_models::register(m)?;
    eapsum::register(m)?;
    cat::register(m)?;
    mstep::register(m)?;
    standard_errors::register(m)?;
    calibration::register(m)?;
    polytomous::register(m)?;
    multigroup::register(m)?;
    gvem::register(m)?;
    multilevel::register(m)?;
    mfrm::register(m)?;
    regularized::register(m)?;
    irtree::register(m)?;
    response_time::register(m)?;
    dynamic::register(m)?;
    explanatory::register(m)?;
    equating::register(m)?;

    Ok(())
}
