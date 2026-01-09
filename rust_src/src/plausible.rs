//! Plausible values generation functions.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::utils::{EPSILON, log_likelihood_2pl_single};

/// Generate plausible values using posterior sampling
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn generate_plausible_values_posterior<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    jitter_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();

    let log_weights: Vec<f64> = weight_vec.iter().map(|&w| (w + EPSILON).ln()).collect();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let normal = Normal::new(0.0, jitter_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_likes: Vec<f64> = (0..n_quad)
                .map(|q| log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec))
                .collect();

            let log_posterior: Vec<f64> = log_likes
                .iter()
                .zip(log_weights.iter())
                .map(|(&ll, &lw)| ll + lw)
                .collect();

            let max_lp = log_posterior
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let posterior: Vec<f64> = log_posterior
                .iter()
                .map(|&lp| (lp - max_lp).exp())
                .collect();
            let sum: f64 = posterior.iter().sum();
            let posterior: Vec<f64> = posterior.iter().map(|&p| p / sum).collect();

            (0..n_plausible)
                .map(|_| {
                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    let mut idx = n_quad - 1;
                    for (q, &p) in posterior.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            idx = q;
                            break;
                        }
                    }
                    quad_vec[idx] + rng.sample(normal)
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Generate plausible values using MCMC
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn generate_plausible_values_mcmc<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    n_iter: usize,
    proposal_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let proposal_dist = Normal::new(0.0, proposal_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let mut theta = 0.0;
            let mut results = Vec::with_capacity(n_plausible);

            for _p in 0..n_plausible {
                for _ in 0..n_iter {
                    let proposal = theta + rng.sample(proposal_dist);

                    let ll_current =
                        log_likelihood_2pl_single(&resp_row, theta, &disc_vec, &diff_vec);
                    let ll_proposal =
                        log_likelihood_2pl_single(&resp_row, proposal, &disc_vec, &diff_vec);

                    let prior_current = -0.5 * theta * theta;
                    let prior_proposal = -0.5 * proposal * proposal;

                    let log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current);

                    let u: f64 = rng.random();
                    if u.ln() < log_alpha {
                        theta = proposal;
                    }
                }
                results.push(theta);
            }
            results
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Register plausible values functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_plausible_values_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(generate_plausible_values_mcmc, m)?)?;
    Ok(())
}
