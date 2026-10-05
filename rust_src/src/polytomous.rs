//! Polytomous IRT model computations (GRM, GPCM).

use crate::likelihood_cache::cached_likelihoods;
use crate::utils::{EPSILON, grm_category_probability};
use numpy::ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Convert category counts to `usize`, mapping negative counts to zero so the
/// parameter check rejects them.
pub(crate) fn category_counts(n_categories: PyReadonlyArray1<i32>) -> Vec<usize> {
    n_categories
        .as_array()
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect()
}

/// Check one parameter row per item and enough columns for each item's
/// categories; `includes_zero` means column 0 belongs to category 0.
pub(crate) fn validate_category_parameters(
    n_items: usize,
    categories: &[usize],
    n_parameters: usize,
    parameters: ArrayView2<'_, f64>,
    includes_zero: bool,
) -> PyResult<()> {
    if categories.len() != n_items
        || n_parameters != n_items
        || parameters.nrows() != n_items
        || categories
            .iter()
            .any(|&k| k < 2 || k - usize::from(!includes_zero) > parameters.ncols())
    {
        return Err(PyValueError::new_err(
            "incompatible item parameters or category counts",
        ));
    }
    Ok(())
}

/// Reject observed responses at or above their item's category count.
pub(crate) fn validate_response_categories(
    responses: ArrayView2<'_, i32>,
    categories: &[usize],
) -> PyResult<()> {
    for row in responses.rows() {
        if row
            .iter()
            .zip(categories)
            .any(|(&r, &k)| r >= 0 && r as usize >= k)
        {
            return Err(PyIndexError::new_err(
                "response category is outside the item category range",
            ));
        }
    }
    Ok(())
}

fn validate_categories(
    responses: ArrayView2<'_, i32>,
    categories: &[usize],
    n_parameters: usize,
    parameters: ArrayView2<'_, f64>,
    includes_zero: bool,
) -> PyResult<()> {
    validate_category_parameters(
        responses.ncols(),
        categories,
        n_parameters,
        parameters,
        includes_zero,
    )?;
    validate_response_categories(responses, categories)
}

/// Fill `row` with the log GRM category probabilities of one item at `theta`.
pub(crate) fn grm_log_row(theta: f64, discrimination: f64, thresholds: &[f64], row: &mut [f64]) {
    let n_categories = row.len();
    for (k, value) in row.iter_mut().enumerate() {
        *value = grm_category_probability(theta, discrimination, thresholds, k, n_categories).ln();
    }
}

/// Turn adjacent-category increments into log category probabilities.
///
/// Category 0 has logit zero and category `k` the sum of `increment(1..=k)`.
/// Probabilities are floored at `EPSILON` before taking logs.
pub(crate) fn adjacent_log_row(row: &mut [f64], increment: impl Fn(usize) -> f64) {
    row[0] = 0.0;
    for k in 1..row.len() {
        row[k] = row[k - 1] + increment(k);
    }
    log_softmax_floor(row);
}

/// Replace logits with floored log-softmax probabilities in place.
pub(crate) fn log_softmax_floor(row: &mut [f64]) {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_denom = max + row.iter().map(|&v| (v - max).exp()).sum::<f64>().ln();
    for value in row {
        *value = (*value - log_denom).exp().max(EPSILON).ln();
    }
}

/// Compute GRM likelihoods using a bounded shared category-probability table.
#[pyfunction]
pub fn compute_log_likelihoods_grm<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array();
    let points = quad_points.as_array();
    let disc = discrimination.as_array();
    let thresholds = thresholds.as_array();
    let categories = category_counts(n_categories);
    validate_categories(responses, &categories, disc.len(), thresholds, false)?;
    let thresholds: Vec<Vec<f64>> = thresholds
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    let result = py.detach(|| {
        cached_likelihoods(responses, points.len(), &categories, false, |q, j, row| {
            grm_log_row(points[q], disc[j], &thresholds[j], row);
        })
    });
    Ok(result.into_pyarray(py))
}

/// Compute GPCM likelihoods; normalizers are shared across all respondents.
#[pyfunction]
pub fn compute_log_likelihoods_gpcm<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    steps: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array();
    let points = quad_points.as_array();
    let disc = discrimination.as_array();
    let steps = steps.as_array();
    let categories = category_counts(n_categories);
    validate_categories(responses, &categories, disc.len(), steps, true)?;
    let result = py.detach(|| {
        cached_likelihoods(responses, points.len(), &categories, false, |q, j, row| {
            adjacent_log_row(row, |k| disc[j] * (points[q] - steps[[j, k]]));
        })
    });
    Ok(result.into_pyarray(py))
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

    if n_items < 3 {
        return numpy::PyArray1::zeros(py, n_items, false);
    }

    let responses_owned = responses.to_owned();

    let total_scores: Vec<f64> = (0..n_persons)
        .map(|i| responses_owned.row(i).iter().filter(|x| !x.is_nan()).sum())
        .collect();
    let observed_counts: Vec<usize> = (0..n_persons)
        .map(|i| {
            responses_owned
                .row(i)
                .iter()
                .filter(|x| !x.is_nan())
                .count()
        })
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
    let item_variance_sum: f64 = item_variances.iter().sum();

    let alpha_if_deleted = py.detach(|| {
        (0..n_items)
            .into_par_iter()
            .map(|j| {
                let observed_after_deletion = |i: usize| {
                    observed_counts[i]
                        - if responses_owned[[i, j]].is_nan() {
                            0
                        } else {
                            1
                        }
                };
                let remaining_score = |i: usize| {
                    total_scores[i]
                        - if responses_owned[[i, j]].is_nan() {
                            0.0
                        } else {
                            responses_owned[[i, j]]
                        }
                };
                let valid_count = (0..n_persons)
                    .filter(|&i| observed_after_deletion(i) > 0)
                    .count();
                if valid_count < 2 {
                    return 0.0;
                }
                let remaining_mean = (0..n_persons)
                    .filter(|&i| observed_after_deletion(i) > 0)
                    .map(remaining_score)
                    .sum::<f64>()
                    / valid_count as f64;
                let remaining_total_var = (0..n_persons)
                    .filter(|&i| observed_after_deletion(i) > 0)
                    .map(|i| (remaining_score(i) - remaining_mean).powi(2))
                    .sum::<f64>()
                    / (valid_count - 1) as f64;
                let remaining_var_sum = item_variance_sum - item_variances[j];

                let k = (n_items - 1) as f64;
                if remaining_total_var > 0.0 && k > 1.0 {
                    let alpha = (k / (k - 1.0)) * (1.0 - remaining_var_sum / remaining_total_var);
                    if alpha.abs() <= 4.0 * f64::EPSILON {
                        0.0
                    } else {
                        alpha
                    }
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
