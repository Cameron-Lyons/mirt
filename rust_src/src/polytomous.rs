//! Polytomous IRT model computations (GRM, GPCM).

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

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
#[pyo3(signature = (responses, quad_points, discrimination, thresholds, n_categories))]
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

    let disc_arc = Arc::new(discrimination.to_vec());
    let n_cat_arc = Arc::new(
        n_categories
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<usize>>(),
    );
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let responses_owned = responses.to_owned();

    let thresh_arc: Arc<Vec<Vec<f64>>> = Arc::new(
        (0..n_items)
            .map(|j| {
                let n_thresh = n_cat_arc[j] - 1;
                (0..n_thresh).map(|k| thresholds[[j, k]]).collect()
            })
            .collect(),
    );

    let result = py.detach(|| {
        let log_likes: Vec<Vec<f64>> = (0..n_persons)
            .into_par_iter()
            .map(|i| {
                let disc = Arc::clone(&disc_arc);
                let n_cat = Arc::clone(&n_cat_arc);
                let thresh = Arc::clone(&thresh_arc);
                let resp_row = responses_owned.row(i);
                (0..n_quad)
                    .map(|q| {
                        let theta = quad_vec[q];
                        let mut ll = 0.0;

                        for j in 0..n_items {
                            let resp = resp_row[j];
                            if resp < 0 {
                                continue;
                            }

                            let prob = grm_category_probability(
                                theta,
                                disc[j],
                                &thresh[j],
                                resp as usize,
                                n_cat[j],
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
        result
    });

    result.to_pyarray(py)
}

/// Compute log-likelihoods for GPCM at all quadrature points
#[pyfunction]
#[pyo3(signature = (responses, quad_points, discrimination, steps, n_categories))]
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

    let n_cat_tmp: Vec<usize> = n_categories.iter().map(|&x| x as usize).collect();
    let disc_arc = Arc::new(discrimination.to_vec());
    let n_cat_arc = Arc::new(n_cat_tmp.clone());
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let responses_owned = responses.to_owned();

    let step_arc: Arc<Vec<Vec<f64>>> = Arc::new(
        (0..n_items)
            .map(|j| {
                let n_steps = n_cat_tmp[j];
                (0..n_steps).map(|k| steps[[j, k]]).collect()
            })
            .collect(),
    );

    let result = py.detach(|| {
        let log_likes: Vec<Vec<f64>> = (0..n_persons)
            .into_par_iter()
            .map(|i| {
                let disc = Arc::clone(&disc_arc);
                let n_cat_v = Arc::clone(&n_cat_arc);
                let step_v = Arc::clone(&step_arc);
                let resp_row = responses_owned.row(i);
                (0..n_quad)
                    .map(|q| {
                        let theta = quad_vec[q];
                        let mut ll = 0.0;

                        for j in 0..n_items {
                            let resp = resp_row[j];
                            if resp < 0 {
                                continue;
                            }

                            let a = disc[j];
                            let n_cat = n_cat_v[j];

                            let mut numerators = vec![0.0; n_cat];
                            numerators[0] = 0.0;
                            for k in 1..n_cat {
                                numerators[k] = numerators[k - 1] + a * (theta - step_v[j][k]);
                            }

                            let max_num =
                                numerators.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let sum_exp: f64 =
                                numerators.iter().map(|&x| (x - max_num).exp()).sum();
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
        result
    });

    result.to_pyarray(py)
}

/// Compute classical test theory statistics efficiently
#[pyfunction]
#[pyo3(signature = (responses,))]
pub fn compute_alpha_if_deleted<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<f64>,
) -> Bound<'py, numpy::PyArray1<f64>> {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let responses_owned = responses.to_owned();

    let total_scores: Vec<f64> = (0..n_persons)
        .map(|i| responses_owned.row(i).iter().filter(|x| !x.is_nan()).sum())
        .collect();

    let item_variances: Vec<f64> = (0..n_items)
        .map(|j| {
            let col: Vec<f64> = responses_owned
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

    let alpha_if_deleted = py.detach(|| {
        (0..n_items)
            .into_par_iter()
            .map(|j| {
                let remaining_scores: Vec<f64> = (0..n_persons)
                    .map(|i| {
                        total_scores[i]
                            - if responses_owned[[i, j]].is_nan() {
                                0.0
                            } else {
                                responses_owned[[i, j]]
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
            .collect::<Vec<f64>>()
    });

    numpy::PyArray1::from_vec(py, alpha_if_deleted)
}

/// Register polytomous functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_grm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_alpha_if_deleted, m)?)?;
    Ok(())
}
