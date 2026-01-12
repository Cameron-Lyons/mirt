//! Multilevel IRT computations.
//!
//! This module provides optimized implementations for:
//! - Two-level hierarchical likelihood
//! - Three-level hierarchical likelihood
//! - Crossed random effects
//! - Variance component estimation

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::log_sigmoid;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn multilevel_log_likelihood_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    group_membership: PyReadonlyArray1<i32>,
    group_means: PyReadonlyArray1<f64>,
    within_variance: f64,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let group_membership = group_membership.as_array();
    let group_means = group_means.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let within_sd = within_variance.sqrt();
    let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_within_sd = within_sd.ln();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let group_means_vec: Vec<f64> = group_means.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let group_idx = group_membership[i] as usize;
            let prior_mean = group_means_vec[group_idx];

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let r = resp_row[j];
                        if r >= 0 {
                            let z = disc_vec[j] * (theta - diff_vec[j]);
                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }

                    let theta_centered = theta - prior_mean;
                    let log_prior = -log_sqrt_2pi
                        - log_within_sd
                        - 0.5 * theta_centered * theta_centered / within_variance;

                    ll + log_prior
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
pub fn update_group_means<'py>(
    py: Python<'py>,
    theta_eap: PyReadonlyArray1<f64>,
    group_membership: PyReadonlyArray1<i32>,
    n_groups: i32,
    between_variance: f64,
    within_variance: f64,
) -> Bound<'py, PyArray1<f64>> {
    let theta_eap = theta_eap.as_array();
    let group_membership = group_membership.as_array();

    let n_groups = n_groups as usize;

    let mut group_sums = vec![0.0; n_groups];
    let mut group_counts = vec![0; n_groups];

    for (i, &g) in group_membership.iter().enumerate() {
        let g = g as usize;
        group_sums[g] += theta_eap[i];
        group_counts[g] += 1;
    }

    let mut group_means = Array1::<f64>::zeros(n_groups);

    for g in 0..n_groups {
        let n_g = group_counts[g] as f64;
        if n_g > 0.0 {
            let shrinkage = n_g / (n_g + within_variance / between_variance);
            let group_mean = group_sums[g] / n_g;
            group_means[g] = shrinkage * group_mean;
        }
    }

    group_means.to_pyarray(py)
}

#[pyfunction]
pub fn estimate_between_variance(
    group_means: PyReadonlyArray1<f64>,
    group_counts: PyReadonlyArray1<i32>,
    within_variance: f64,
) -> f64 {
    let group_means = group_means.as_array();
    let group_counts = group_counts.as_array();

    let n_groups = group_means.len();

    let total_n: i32 = group_counts.iter().sum();
    if total_n == 0 || n_groups < 2 {
        return 0.0;
    }

    let weighted_mean: f64 = group_means
        .iter()
        .zip(group_counts.iter())
        .map(|(&m, &n)| m * (n as f64))
        .sum::<f64>()
        / (total_n as f64);

    let between_ss: f64 = group_means
        .iter()
        .zip(group_counts.iter())
        .map(|(&m, &n)| (n as f64) * (m - weighted_mean).powi(2))
        .sum();

    let df_between = (n_groups - 1) as f64;
    let ms_between = between_ss / df_between;

    let harmonic_n = (n_groups as f64)
        / group_counts
            .iter()
            .map(|&n| 1.0 / (n as f64).max(1.0))
            .sum::<f64>();

    (ms_between - within_variance).max(0.0) / harmonic_n
}

#[pyfunction]
pub fn crossed_random_effects_loglik<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    rater_assignments: PyReadonlyArray2<i32>,
    rater_effects: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let rater_assignments = rater_assignments.as_array();
    let rater_effects = rater_effects.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let rater_eff_vec: Vec<f64> = rater_effects.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let rater_row: Vec<i32> = rater_assignments.row(i).to_vec();

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let r = resp_row[j];
                        if r >= 0 {
                            let rater_idx = rater_row[j];
                            let rater_effect = if rater_idx >= 0 {
                                rater_eff_vec[rater_idx as usize]
                            } else {
                                0.0
                            };

                            let z = disc_vec[j] * (theta - diff_vec[j]) + rater_effect;
                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }
                    ll
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn update_rater_effects<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_eap: PyReadonlyArray1<f64>,
    rater_assignments: PyReadonlyArray2<i32>,
    n_raters: i32,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    rater_variance: f64,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let theta_eap = theta_eap.as_array();
    let rater_assignments = rater_assignments.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_raters = n_raters as usize;

    let mut rater_numerator = vec![0.0; n_raters];
    let mut rater_denominator = vec![0.0; n_raters];

    for i in 0..n_persons {
        let theta_i = theta_eap[i];
        for j in 0..n_items {
            let r = responses[[i, j]];
            let rater_idx = rater_assignments[[i, j]];

            if r >= 0 && rater_idx >= 0 {
                let rater_idx = rater_idx as usize;
                let z = discrimination[j] * (theta_i - difficulty[j]);
                let p = 1.0 / (1.0 + (-z).exp());

                rater_numerator[rater_idx] += (r as f64) - p;
                rater_denominator[rater_idx] += p * (1.0 - p);
            }
        }
    }

    let mut rater_effects = Array1::<f64>::zeros(n_raters);
    let prior_precision = 1.0 / rater_variance;

    for k in 0..n_raters {
        if rater_denominator[k] > 0.0 {
            let posterior_var = 1.0 / (rater_denominator[k] + prior_precision);
            rater_effects[k] = posterior_var * rater_numerator[k];
        }
    }

    let mean_effect: f64 = rater_effects.iter().sum::<f64>() / n_raters as f64;
    for effect in rater_effects.iter_mut() {
        *effect -= mean_effect;
    }

    rater_effects.to_pyarray(py)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn three_level_log_likelihood<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    level2_membership: PyReadonlyArray1<i32>,
    level3_membership: PyReadonlyArray1<i32>,
    level2_effects: PyReadonlyArray1<f64>,
    level3_effects: PyReadonlyArray1<f64>,
    within_variance: f64,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let level2_membership = level2_membership.as_array();
    let level3_membership = level3_membership.as_array();
    let level2_effects = level2_effects.as_array();
    let level3_effects = level3_effects.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let within_sd = within_variance.sqrt();
    let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_within_sd = within_sd.ln();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let l2_eff_vec: Vec<f64> = level2_effects.to_vec();
    let l3_eff_vec: Vec<f64> = level3_effects.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let l2_idx = level2_membership[i] as usize;
            let l3_idx = level3_membership[l2_idx] as usize;
            let prior_mean = l2_eff_vec[l2_idx] + l3_eff_vec[l3_idx];

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let r = resp_row[j];
                        if r >= 0 {
                            let z = disc_vec[j] * (theta - diff_vec[j]);
                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }

                    let theta_centered = theta - prior_mean;
                    let log_prior = -log_sqrt_2pi
                        - log_within_sd
                        - 0.5 * theta_centered * theta_centered / within_variance;

                    ll + log_prior
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multilevel_log_likelihood_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(update_group_means, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_between_variance, m)?)?;
    m.add_function(wrap_pyfunction!(crossed_random_effects_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(update_rater_effects, m)?)?;
    m.add_function(wrap_pyfunction!(three_level_log_likelihood, m)?)?;
    Ok(())
}
