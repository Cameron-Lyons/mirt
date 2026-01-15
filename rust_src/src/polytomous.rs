//! Polytomous IRT model computations (GRM, GPCM).

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::grm_category_probability;

/// Compute log-likelihoods for GRM at all quadrature points
///
/// Parameters:
/// - responses: (n_persons, n_items) response matrix
/// - quad_points: (n_quad,) quadrature points
/// - discrimination: (n_items,) discrimination parameters
/// - thresholds: (n_items, n_categories-1) threshold parameters
/// - n_categories: (n_items,) number of categories per item
#[pyfunction]
pub fn compute_log_likelihoods_grm<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();
    let n_categories = n_categories.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let n_cat_vec: Vec<usize> = n_categories.iter().map(|&x| x as usize).collect();

    let thresh_vecs: Vec<Vec<f64>> = (0..n_items)
        .map(|j| {
            let n_thresh = n_cat_vec[j] - 1;
            (0..n_thresh).map(|k| thresholds[[j, k]]).collect()
        })
        .collect();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];
                    let mut ll = 0.0;

                    for j in 0..n_items {
                        let resp = resp_row[j];
                        if resp < 0 {
                            continue;
                        }

                        let prob = grm_category_probability(
                            theta,
                            disc_vec[j],
                            &thresh_vecs[j],
                            resp as usize,
                            n_cat_vec[j],
                        );
                        ll += prob.ln();
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

/// Compute log-likelihoods for GPCM at all quadrature points
#[pyfunction]
pub fn compute_log_likelihoods_gpcm<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    steps: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let steps = steps.as_array();
    let n_categories = n_categories.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let n_cat_vec: Vec<usize> = n_categories.iter().map(|&x| x as usize).collect();

    let step_vecs: Vec<Vec<f64>> = (0..n_items)
        .map(|j| {
            let n_steps = n_cat_vec[j];
            (0..n_steps).map(|k| steps[[j, k]]).collect()
        })
        .collect();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];
                    let mut ll = 0.0;

                    for j in 0..n_items {
                        let resp = resp_row[j];
                        if resp < 0 {
                            continue;
                        }

                        let a = disc_vec[j];
                        let n_cat = n_cat_vec[j];

                        let mut numerators = vec![0.0; n_cat];
                        numerators[0] = 0.0;
                        for k in 1..n_cat {
                            numerators[k] = numerators[k - 1] + a * (theta - step_vecs[j][k]);
                        }

                        let max_num = numerators.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let sum_exp: f64 = numerators.iter().map(|&x| (x - max_num).exp()).sum();
                        let log_denom = max_num + sum_exp.ln();

                        let prob = (numerators[resp as usize] - log_denom).exp().max(1e-10);
                        ll += prob.ln();
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

/// Compute classical test theory statistics efficiently
#[pyfunction]
pub fn compute_alpha_if_deleted<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<f64>,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let total_scores: Vec<f64> = (0..n_persons)
        .map(|i| responses.row(i).iter().filter(|x| !x.is_nan()).sum())
        .collect();

    let item_variances: Vec<f64> = (0..n_items)
        .map(|j| {
            let col: Vec<f64> = responses
                .column(j)
                .iter()
                .filter(|x| !x.is_nan())
                .cloned()
                .collect();
            if col.is_empty() {
                return 0.0;
            }
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (col.len() - 1).max(1) as f64
        })
        .collect();

    let alpha_if_deleted: Vec<f64> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let remaining_scores: Vec<f64> = (0..n_persons)
                .map(|i| {
                    total_scores[i]
                        - if responses[[i, j]].is_nan() {
                            0.0
                        } else {
                            responses[[i, j]]
                        }
                })
                .collect();

            let remaining_var_sum: f64 = (0..n_items)
                .filter(|&k| k != j)
                .map(|k| item_variances[k])
                .sum();

            let remaining_mean = remaining_scores.iter().sum::<f64>() / n_persons as f64;
            let remaining_total_var = remaining_scores
                .iter()
                .map(|x| (x - remaining_mean).powi(2))
                .sum::<f64>()
                / (n_persons - 1).max(1) as f64;

            let k = (n_items - 1) as f64;
            if remaining_total_var > 0.0 && k > 1.0 {
                (k / (k - 1.0)) * (1.0 - remaining_var_sum / remaining_total_var)
            } else {
                0.0
            }
        })
        .collect();

    numpy::PyArray1::from_vec(py, alpha_if_deleted)
}

/// Register polytomous functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_grm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_alpha_if_deleted, m)?)?;
    Ok(())
}
