//! E-step and expected counts computation functions.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{compute_log_weights, log_likelihood_2pl_single, logsumexp, LOG_2_PI};

/// Complete E-step computation with posterior weights
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn e_step_complete<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
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

    let log_prior: Vec<f64> = quad_vec
        .iter()
        .map(|&theta| {
            let z = (theta - prior_mean) / prior_var.sqrt();
            -0.5 * (LOG_2_PI + prior_var.ln() + z * z)
        })
        .collect();

    let log_weights = compute_log_weights(&weight_vec);

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let ll =
                        log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec);
                    ll + log_prior[q] + log_weights[q]
                })
                .collect();

            let log_marginal = logsumexp(&log_joint);

            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal.exp())
        })
        .collect();

    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (post, marg)) in results.iter().enumerate() {
        for (q, &p) in post.iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        marginal_ll[i] = *marg;
    }

    (posterior_weights.to_pyarray(py), marginal_ll.to_pyarray(py))
}

/// Compute r_k (expected counts) for dichotomous items
#[pyfunction]
pub fn compute_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 {
            continue;
        }
        for q in 0..n_quad {
            let w = posterior_weights[[i, q]];
            n_k[q] += w;
            if resp == 1 {
                r_k[q] += w;
            }
        }
    }

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

/// Compute r_kc (expected counts per category) for polytomous items
#[pyfunction]
pub fn compute_expected_counts_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    n_categories: usize,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_kc = Array2::zeros((n_quad, n_categories));

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 || resp as usize >= n_categories {
            continue;
        }
        for q in 0..n_quad {
            r_kc[[q, resp as usize]] += posterior_weights[[i, q]];
        }
    }

    r_kc.to_pyarray(py)
}

/// MCEM E-step using theta samples
#[pyfunction]
pub fn mcem_e_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_samples: PyReadonlyArray3<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let theta_samples = theta_samples.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_samples = theta_samples.shape()[1];

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_likes: Vec<f64> = (0..n_samples)
                .map(|s| {
                    let theta_s = theta_samples[[i, s, 0]];
                    log_likelihood_2pl_single(&resp_row, theta_s, &disc_vec, &diff_vec)
                })
                .collect();

            let max_ll = log_likes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = log_likes.iter().map(|&ll| (ll - max_ll).exp()).collect();
            let sum: f64 = weights.iter().sum();
            let normalized: Vec<f64> = weights.iter().map(|&w| w / sum).collect();

            let marginal = (sum / n_samples as f64) * max_ll.exp();

            (normalized, marginal)
        })
        .collect();

    let mut importance_weights = Array2::zeros((n_persons, n_samples));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (weights, marg)) in results.iter().enumerate() {
        for (s, &w) in weights.iter().enumerate() {
            importance_weights[[i, s]] = w;
        }
        marginal_ll[i] = *marg;
    }

    (
        importance_weights.to_pyarray(py),
        marginal_ll.to_pyarray(py),
    )
}

/// Weighted E-step for survey data
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn weighted_e_step<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    weights: PyReadonlyArray1<f64>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let responses = responses.as_array();
    let survey_weights = weights.as_array();
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

    let log_weights = compute_log_weights(&weight_vec);

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let ll =
                        log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec);
                    ll + log_weights[q]
                })
                .collect();

            let log_marginal = logsumexp(&log_joint);

            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal)
        })
        .collect();

    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut weighted_ll = 0.0;

    for (i, (post, log_marg)) in results.iter().enumerate() {
        for (q, &p) in post.iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        weighted_ll += survey_weights[i] * log_marg;
    }

    (posterior_weights.to_pyarray(py), weighted_ll)
}

/// Weighted expected counts for survey data
#[pyfunction]
pub fn weighted_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    survey_weights: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let survey_weights = survey_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 {
            continue;
        }
        let sw = survey_weights[i];
        for q in 0..n_quad {
            let w = posterior_weights[[i, q]] * sw;
            n_k[q] += w;
            if resp == 1 {
                r_k[q] += w;
            }
        }
    }

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

/// Register E-step functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(e_step_complete, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_polytomous, m)?)?;
    m.add_function(wrap_pyfunction!(mcem_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_expected_counts, m)?)?;
    Ok(())
}
