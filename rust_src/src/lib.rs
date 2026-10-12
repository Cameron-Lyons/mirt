//! High-performance Rust backend for MIRT (Multidimensional Item Response Theory).
//!
//! This crate provides optimized implementations of IRT algorithms including:
//! - Log-likelihood computations for dichotomous and polytomous IRT models
//! - E-step and M-step algorithms for EM estimation
//! - Response simulation for various IRT models
//! - Parameter estimation (MHRM, bootstrap EM, fused 3PL EM iterations)
//! - Diagnostic statistics (pointwise 2PL log-likelihood, alpha if item deleted)
//! - Person scoring (optimized MAP/ML scoring)
//! - CAT (Computerized Adaptive Testing) simulation
//! - Observed-score distributions by the Lord-Wingersky recursion

use pyo3::prelude::*;

mod bayesian_diagnostics;
mod box_newton;
mod cat;
mod counts;
mod dynamic;
mod equating;
mod estimation;
mod explanatory;
mod item_prior;
mod likelihood;
mod likelihood_cache;
mod multigroup;
mod optimization_scoring;
mod patterns;
mod polytomous;
mod polytomous_mstep;
mod posterior;
mod regularized;
mod response_time;
mod simulation;
mod utils;

/// Python module for mirt_rs
#[pymodule]
fn mirt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bayesian_diagnostics::register(m)?;
    simulation::register(m)?;
    estimation::register(m)?;
    optimization_scoring::register(m)?;
    cat::register(m)?;
    polytomous_mstep::register(m)?;
    polytomous::register(m)?;
    multigroup::register(m)?;
    regularized::register(m)?;
    response_time::register(m)?;
    dynamic::register(m)?;
    explanatory::register(m)?;
    equating::register(m)?;
    patterns::register(m)?;
    posterior::register(m)?;
    likelihood_cache::register(m)?;

    Ok(())
}
