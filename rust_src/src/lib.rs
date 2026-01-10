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

pub mod bootstrap;
pub mod calibration;
pub mod cat;
pub mod diagnostics;
pub mod eapsum;
pub mod estep;
pub mod estimation;
pub mod likelihood;
pub mod mirt_models;
pub mod mstep;
pub mod plausible;
pub mod scoring;
pub mod sibtest;
pub mod simulation;
pub mod standard_errors;

/// Python module for mirt_rs
#[pymodule]
fn mirt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    Ok(())
}
