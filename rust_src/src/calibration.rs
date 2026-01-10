//! Fixed-item calibration functions for IRT models.
//!
//! This module provides parallelized fixed-item calibration using Rayon,
//! enabling efficient calibration of new items to an existing scale.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, compute_log_weights, log_sigmoid, logsumexp, sigmoid};

/// Compute log-likelihood matrix for anchor items across all persons and quadrature points.
fn compute_anchor_likelihood(
    anchor_responses: &[Vec<i32>],
    theta_grid: &[f64],
    anchor_disc: &[f64],
    anchor_diff: &[f64],
) -> Array2<f64> {
    let n_persons = anchor_responses.len();
    let n_quad = theta_grid.len();
    let n_anchor = anchor_disc.len();

    let results: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp = &anchor_responses[i];
            let mut ll_row = vec![0.0; n_quad];

            for (q, &theta) in theta_grid.iter().enumerate() {
                let mut ll = 0.0;
                for j in 0..n_anchor {
                    let r = resp[j];
                    if r < 0 {
                        continue;
                    }
                    let z = anchor_disc[j] * (theta - anchor_diff[j]);
                    if r == 1 {
                        ll += log_sigmoid(z);
                    } else {
                        ll += log_sigmoid(-z);
                    }
                }
                ll_row[q] = ll;
            }
            ll_row
        })
        .collect();

    let mut anchor_ll = Array2::zeros((n_persons, n_quad));
    for (i, row) in results.into_iter().enumerate() {
        for (q, val) in row.into_iter().enumerate() {
            anchor_ll[[i, q]] = val;
        }
    }
    anchor_ll
}

/// E-step: Compute posterior weights given current parameters.
fn e_step_fixed_calib(
    anchor_ll: &Array2<f64>,
    new_responses: &[Vec<i32>],
    theta_grid: &[f64],
    log_weights: &[f64],
    new_disc: &[f64],
    new_diff: &[f64],
) -> (Array2<f64>, f64) {
    let n_persons = new_responses.len();
    let n_quad = theta_grid.len();
    let n_new = new_disc.len();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp = &new_responses[i];

            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let theta = theta_grid[q];
                    let mut ll_new = 0.0;
                    for j in 0..n_new {
                        let r = resp[j];
                        if r < 0 {
                            continue;
                        }
                        let z = new_disc[j] * (theta - new_diff[j]);
                        if r == 1 {
                            ll_new += log_sigmoid(z);
                        } else {
                            ll_new += log_sigmoid(-z);
                        }
                    }
                    anchor_ll[[i, q]] + ll_new + log_weights[q]
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
    let mut total_ll = 0.0;

    for (i, (post, log_marg)) in results.into_iter().enumerate() {
        for (q, p) in post.into_iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        total_ll += log_marg;
    }

    (posterior_weights, total_ll)
}

/// M-step: Update parameters for new items using expected counts.
#[allow(clippy::too_many_arguments)]
fn m_step_fixed_calib(
    new_responses: &[Vec<i32>],
    posterior_weights: &Array2<f64>,
    theta_grid: &[f64],
    old_disc: &[f64],
    old_diff: &[f64],
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    prob_clamp: (f64, f64),
    min_count: f64,
    min_valid_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_persons = new_responses.len();
    let n_quad = theta_grid.len();
    let n_new = old_disc.len();

    let results: Vec<(f64, f64)> = (0..n_new)
        .into_par_iter()
        .map(|j| {
            let mut r_j = vec![0.0; n_quad];
            let mut n_j = vec![0.0; n_quad];

            for i in 0..n_persons {
                let r = new_responses[i][j];
                if r < 0 {
                    continue;
                }
                for q in 0..n_quad {
                    let w = posterior_weights[[i, q]];
                    n_j[q] += w;
                    if r == 1 {
                        r_j[q] += w;
                    }
                }
            }

            let mut valid_thetas = Vec::new();
            let mut valid_logits = Vec::new();
            let mut valid_weights = Vec::new();

            for q in 0..n_quad {
                if n_j[q] > min_count {
                    let p_j = (r_j[q] / n_j[q]).clamp(prob_clamp.0, prob_clamp.1);
                    let logit = (p_j / (1.0 - p_j)).ln();
                    valid_thetas.push(theta_grid[q]);
                    valid_logits.push(logit);
                    valid_weights.push(n_j[q]);
                }
            }

            if valid_thetas.len() < min_valid_points {
                return (old_disc[j], old_diff[j]);
            }

            let sum_w: f64 = valid_weights.iter().sum();
            let mean_theta: f64 = valid_thetas
                .iter()
                .zip(valid_weights.iter())
                .map(|(&t, &w)| t * w)
                .sum::<f64>()
                / sum_w;
            let mean_logit: f64 = valid_logits
                .iter()
                .zip(valid_weights.iter())
                .map(|(&l, &w)| l * w)
                .sum::<f64>()
                / sum_w;

            let var_theta: f64 = valid_thetas
                .iter()
                .zip(valid_weights.iter())
                .map(|(&t, &w)| w * (t - mean_theta).powi(2))
                .sum::<f64>()
                / sum_w;

            let cov_theta_logit: f64 = valid_thetas
                .iter()
                .zip(valid_logits.iter())
                .zip(valid_weights.iter())
                .map(|((&t, &l), &w)| w * (t - mean_theta) * (l - mean_logit))
                .sum::<f64>()
                / sum_w;

            if var_theta < EPSILON {
                return (old_disc[j], old_diff[j]);
            }

            let new_a = (cov_theta_logit / var_theta).clamp(disc_bounds.0, disc_bounds.1);
            let new_b = (mean_theta - mean_logit / new_a).clamp(diff_bounds.0, diff_bounds.1);

            (new_a, new_b)
        })
        .collect();

    let new_disc: Vec<f64> = results.iter().map(|(a, _)| *a).collect();
    let new_diff: Vec<f64> = results.iter().map(|(_, b)| *b).collect();

    (new_disc, new_diff)
}

/// Fixed-item calibration EM algorithm.
///
/// Calibrates new items to an existing scale defined by anchor items
/// with fixed parameters.
#[pyfunction]
#[pyo3(signature = (
    responses,
    anchor_items,
    new_items,
    anchor_disc,
    anchor_diff,
    theta_grid,
    quad_weights,
    max_iter,
    tol,
    disc_bounds = (0.2, 5.0),
    diff_bounds = (-5.0, 5.0),
    prob_clamp = (0.01, 0.99),
    init_disc = 1.0,
    init_diff = 0.0,
    min_count = 1.0,
    min_valid_points = 3,
))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn fixed_calib_em<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    anchor_items: Vec<usize>,
    new_items: Vec<usize>,
    anchor_disc: PyReadonlyArray1<f64>,
    anchor_diff: PyReadonlyArray1<f64>,
    theta_grid: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    prob_clamp: (f64, f64),
    init_disc: f64,
    init_diff: f64,
    min_count: f64,
    min_valid_points: usize,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    bool,
) {
    let responses = responses.as_array();
    let anchor_disc = anchor_disc.as_array();
    let anchor_diff = anchor_diff.as_array();
    let theta_grid = theta_grid.as_array();
    let quad_weights = quad_weights.as_array();

    let n_persons = responses.nrows();
    let n_quad = theta_grid.len();
    let n_new = new_items.len();

    let anchor_disc_vec: Vec<f64> = anchor_disc.to_vec();
    let anchor_diff_vec: Vec<f64> = anchor_diff.to_vec();
    let theta_grid_vec: Vec<f64> = theta_grid.to_vec();
    let log_weights = compute_log_weights(&quad_weights.to_vec());

    let anchor_responses: Vec<Vec<i32>> = (0..n_persons)
        .map(|i| anchor_items.iter().map(|&j| responses[[i, j]]).collect())
        .collect();

    let new_responses: Vec<Vec<i32>> = (0..n_persons)
        .map(|i| new_items.iter().map(|&j| responses[[i, j]]).collect())
        .collect();

    let anchor_ll = compute_anchor_likelihood(
        &anchor_responses,
        &theta_grid_vec,
        &anchor_disc_vec,
        &anchor_diff_vec,
    );

    let mut new_disc = vec![init_disc; n_new];
    let mut new_diff = vec![init_diff; n_new];

    let mut converged = false;
    let mut log_likelihood = f64::NEG_INFINITY;
    let mut final_iter = 0;
    let mut posterior_weights = Array2::zeros((n_persons, n_quad));

    for iteration in 0..max_iter {
        final_iter = iteration;

        let (new_posterior, new_ll) = e_step_fixed_calib(
            &anchor_ll,
            &new_responses,
            &theta_grid_vec,
            &log_weights,
            &new_disc,
            &new_diff,
        );
        posterior_weights = new_posterior;

        if (new_ll - log_likelihood).abs() < tol {
            log_likelihood = new_ll;
            converged = true;
            break;
        }
        log_likelihood = new_ll;

        let old_disc = new_disc.clone();
        let old_diff = new_diff.clone();

        let (updated_disc, updated_diff) = m_step_fixed_calib(
            &new_responses,
            &posterior_weights,
            &theta_grid_vec,
            &old_disc,
            &old_diff,
            disc_bounds,
            diff_bounds,
            prob_clamp,
            min_count,
            min_valid_points,
        );
        new_disc = updated_disc;
        new_diff = updated_diff;

        let max_disc_change = new_disc
            .iter()
            .zip(old_disc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let max_diff_change = new_diff
            .iter()
            .zip(old_diff.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        if max_disc_change + max_diff_change < tol {
            converged = true;
            break;
        }
    }

    let theta_est: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut theta_eap = 0.0;
            for q in 0..n_quad {
                theta_eap += posterior_weights[[i, q]] * theta_grid_vec[q];
            }
            theta_eap
        })
        .collect();

    let disc_arr: Array1<f64> = new_disc.into();
    let diff_arr: Array1<f64> = new_diff.into();
    let theta_arr: Array1<f64> = theta_est.into();

    (
        disc_arr.to_pyarray(py),
        diff_arr.to_pyarray(py),
        theta_arr.to_pyarray(py),
        log_likelihood,
        final_iter + 1,
        converged,
    )
}

/// Stocking-Lord equating criterion computation.
#[pyfunction]
pub fn stocking_lord_criterion(
    disc_old: PyReadonlyArray1<f64>,
    diff_old: PyReadonlyArray1<f64>,
    disc_new: PyReadonlyArray1<f64>,
    diff_new: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    theta_grid: PyReadonlyArray1<f64>,
) -> f64 {
    let disc_old = disc_old.as_array();
    let diff_old = diff_old.as_array();
    let disc_new = disc_new.as_array();
    let diff_new = diff_new.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc_old.len();
    let n_theta = theta_grid.len();

    (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut item_diff = 0.0;
            for q in 0..n_theta {
                let theta = theta_grid[q];
                let p_old = sigmoid(disc_old[j] * (theta - diff_old[j]));
                let theta_trans = a * theta + b;
                let p_new = sigmoid(disc_new[j] * (theta_trans - diff_new[j]));
                item_diff += (p_old - p_new).powi(2);
            }
            item_diff
        })
        .sum()
}

/// Register calibration functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fixed_calib_em, m)?)?;
    m.add_function(wrap_pyfunction!(stocking_lord_criterion, m)?)?;
    Ok(())
}
