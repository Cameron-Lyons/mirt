//! Response time modeling support.
//!
//! This module provides functions for joint modeling of response accuracy
//! and response times using Van der Linden's hierarchical framework.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::EPSILON;

/// Sigmoid function with numerical stability.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Log-normal density.
#[inline]
fn log_lognormal_density(log_x: f64, mean: f64, precision: f64) -> f64 {
    let var = 1.0 / precision;
    -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * precision * (log_x - mean).powi(2)
}

/// Compute joint log-likelihood for response time model.
///
/// # Arguments
/// * `responses` - Binary responses (n_persons, n_items), -1 for missing
/// * `log_rt` - Log response times (n_persons, n_items), NaN for missing
/// * `theta` - Ability parameters (n_persons,)
/// * `tau` - Speed parameters (n_persons,)
/// * `discrimination` - Accuracy discrimination (n_items,)
/// * `difficulty` - Accuracy difficulty (n_items,)
/// * `time_discrimination` - RT discrimination/alpha (n_items,)
/// * `time_intensity` - RT intensity/beta (n_items,)
///
/// # Returns
/// Joint log-likelihood per person (n_persons,)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rt_joint_log_likelihood<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    log_rt: PyReadonlyArray2<f64>,
    theta: PyReadonlyArray1<f64>,
    tau: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    time_discrimination: PyReadonlyArray1<f64>,
    time_intensity: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let log_rt = log_rt.as_array();
    let theta = theta.as_array();
    let tau = tau.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();
    let time_disc = time_discrimination.as_array();
    let time_int = time_intensity.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let log_likes: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut ll = 0.0;
            let theta_i = theta[i];
            let tau_i = tau[i];

            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp >= 0 {
                    let z = disc[j] * (theta_i - diff[j]);
                    let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

                    if resp == 1 {
                        ll += p.ln();
                    } else {
                        ll += (1.0 - p).ln();
                    }
                }

                let rt = log_rt[[i, j]];
                if !rt.is_nan() {
                    let mean = time_int[j] - tau_i;
                    let precision = time_disc[j].powi(2);
                    ll += log_lognormal_density(rt, mean, precision);
                }
            }

            ll
        })
        .collect();

    Array1::from(log_likes).to_pyarray(py)
}

/// Compute joint log-likelihood for 3PL response time model.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rt_joint_log_likelihood_3pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    log_rt: PyReadonlyArray2<f64>,
    theta: PyReadonlyArray1<f64>,
    tau: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
    time_discrimination: PyReadonlyArray1<f64>,
    time_intensity: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let log_rt = log_rt.as_array();
    let theta = theta.as_array();
    let tau = tau.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();
    let guess = guessing.as_array();
    let time_disc = time_discrimination.as_array();
    let time_int = time_intensity.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let log_likes: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut ll = 0.0;
            let theta_i = theta[i];
            let tau_i = tau[i];

            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp >= 0 {
                    let z = disc[j] * (theta_i - diff[j]);
                    let p_star = sigmoid(z);
                    let p = (guess[j] + (1.0 - guess[j]) * p_star).clamp(EPSILON, 1.0 - EPSILON);

                    if resp == 1 {
                        ll += p.ln();
                    } else {
                        ll += (1.0 - p).ln();
                    }
                }

                let rt = log_rt[[i, j]];
                if !rt.is_nan() {
                    let mean = time_int[j] - tau_i;
                    let precision = time_disc[j].powi(2);
                    ll += log_lognormal_density(rt, mean, precision);
                }
            }

            ll
        })
        .collect();

    Array1::from(log_likes).to_pyarray(py)
}

/// Sample (theta, tau) for all persons via Metropolis-Hastings.
///
/// Uses a random walk proposal with bivariate normal.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn rt_sample_person_params<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    log_rt: PyReadonlyArray2<f64>,
    theta_current: PyReadonlyArray1<f64>,
    tau_current: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    time_discrimination: PyReadonlyArray1<f64>,
    time_intensity: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma_inv: PyReadonlyArray2<f64>,
    log_det_sigma: f64,
    proposal_sd: f64,
    seed: u64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
) {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};

    let responses = responses.as_array();
    let log_rt = log_rt.as_array();
    let theta = theta_current.as_array();
    let tau = tau_current.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();
    let time_disc = time_discrimination.as_array();
    let time_int = time_intensity.as_array();
    let mu = mu.as_array();
    let sigma_inv = sigma_inv.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let results: Vec<(f64, f64, i32)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i as u64);
            let normal = Normal::new(0.0, proposal_sd).unwrap();
            let uniform = Uniform::new(0.0f64, 1.0).unwrap();

            let theta_prop = theta[i] + normal.sample(&mut rng);
            let tau_prop = tau[i] + normal.sample(&mut rng);

            let log_prior_curr =
                log_mvn_density_single(theta[i], tau[i], mu[0], mu[1], &sigma_inv, log_det_sigma);
            let log_prior_prop = log_mvn_density_single(
                theta_prop,
                tau_prop,
                mu[0],
                mu[1],
                &sigma_inv,
                log_det_sigma,
            );

            let mut log_like_curr = 0.0;
            let mut log_like_prop = 0.0;

            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp >= 0 {
                    let z_curr = disc[j] * (theta[i] - diff[j]);
                    let z_prop = disc[j] * (theta_prop - diff[j]);

                    let p_curr = sigmoid(z_curr).clamp(EPSILON, 1.0 - EPSILON);
                    let p_prop = sigmoid(z_prop).clamp(EPSILON, 1.0 - EPSILON);

                    if resp == 1 {
                        log_like_curr += p_curr.ln();
                        log_like_prop += p_prop.ln();
                    } else {
                        log_like_curr += (1.0 - p_curr).ln();
                        log_like_prop += (1.0 - p_prop).ln();
                    }
                }

                let rt = log_rt[[i, j]];
                if !rt.is_nan() {
                    let precision = time_disc[j].powi(2);

                    let mean_curr = time_int[j] - tau[i];
                    let mean_prop = time_int[j] - tau_prop;

                    log_like_curr += log_lognormal_density(rt, mean_curr, precision);
                    log_like_prop += log_lognormal_density(rt, mean_prop, precision);
                }
            }

            let log_accept = (log_like_prop + log_prior_prop) - (log_like_curr + log_prior_curr);

            if uniform.sample(&mut rng).ln() < log_accept {
                (theta_prop, tau_prop, 1)
            } else {
                (theta[i], tau[i], 0)
            }
        })
        .collect();

    let mut new_theta = Array1::zeros(n_persons);
    let mut new_tau = Array1::zeros(n_persons);
    let mut accepted = Array1::zeros(n_persons);

    for (i, (t, s, a)) in results.into_iter().enumerate() {
        new_theta[i] = t;
        new_tau[i] = s;
        accepted[i] = a;
    }

    (
        new_theta.to_pyarray(py),
        new_tau.to_pyarray(py),
        accepted.to_pyarray(py),
    )
}

/// Log bivariate normal density for single observation.
#[inline]
fn log_mvn_density_single(
    x1: f64,
    x2: f64,
    mu1: f64,
    mu2: f64,
    sigma_inv: &ndarray::ArrayView2<f64>,
    log_det_sigma: f64,
) -> f64 {
    let d1 = x1 - mu1;
    let d2 = x2 - mu2;
    let maha = d1 * d1 * sigma_inv[[0, 0]]
        + 2.0 * d1 * d2 * sigma_inv[[0, 1]]
        + d2 * d2 * sigma_inv[[1, 1]];

    let log_norm = -0.5 * (2.0 * (2.0 * std::f64::consts::PI).ln() + log_det_sigma);
    log_norm - 0.5 * maha
}

/// Compute bivariate normal log-density for array of observations.
#[pyfunction]
pub fn rt_log_mvn_density<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    tau: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma_inv: PyReadonlyArray2<f64>,
    log_det_sigma: f64,
) -> Bound<'py, PyArray1<f64>> {
    let theta = theta.as_array();
    let tau = tau.as_array();
    let mu = mu.as_array();
    let sigma_inv = sigma_inv.as_array();

    let n_persons = theta.len();

    let log_densities: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| log_mvn_density_single(theta[i], tau[i], mu[0], mu[1], &sigma_inv, log_det_sigma))
        .collect();

    Array1::from(log_densities).to_pyarray(py)
}

/// Compute sufficient statistics for response time parameters.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn rt_time_sufficient_stats<'py>(
    py: Python<'py>,
    log_rt: PyReadonlyArray2<f64>,
    tau: PyReadonlyArray1<f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
) {
    let log_rt = log_rt.as_array();
    let tau = tau.as_array();

    let n_persons = log_rt.nrows();
    let n_items = log_rt.ncols();

    let stats: Vec<(f64, f64, i32)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut sum_residual = 0.0;
            let mut sum_sq_residual = 0.0;
            let mut count = 0;

            for i in 0..n_persons {
                let rt = log_rt[[i, j]];
                if !rt.is_nan() {
                    let residual = rt + tau[i];
                    sum_residual += residual;
                    sum_sq_residual += residual * residual;
                    count += 1;
                }
            }

            (sum_residual, sum_sq_residual, count)
        })
        .collect();

    let mut sum_residuals = Array1::zeros(n_items);
    let mut sum_sq_residuals = Array1::zeros(n_items);
    let mut counts = Array1::zeros(n_items);

    for (j, (sr, ssr, c)) in stats.into_iter().enumerate() {
        sum_residuals[j] = sr;
        sum_sq_residuals[j] = ssr;
        counts[j] = c;
    }

    (
        sum_residuals.to_pyarray(py),
        sum_sq_residuals.to_pyarray(py),
        counts.to_pyarray(py),
    )
}

/// Register response time functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rt_joint_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(rt_joint_log_likelihood_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(rt_sample_person_params, m)?)?;
    m.add_function(wrap_pyfunction!(rt_log_mvn_density, m)?)?;
    m.add_function(wrap_pyfunction!(rt_time_sufficient_stats, m)?)?;
    Ok(())
}
