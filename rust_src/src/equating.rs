//! High-performance equating and linking functions.
//!
//! This module provides parallelized implementations of:
//! - Haebara and Stocking-Lord criterion computation
//! - ICC area difference calculations
//! - Lord-Wingersky recursion for observed score distributions

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::sigmoid;

/// Haebara equating criterion - item-level curve matching.
///
/// Computes sum of squared differences between item characteristic curves
/// after transformation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn haebara_criterion(
    disc_old: PyReadonlyArray1<f64>,
    diff_old: PyReadonlyArray1<f64>,
    disc_new: PyReadonlyArray1<f64>,
    diff_new: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    theta_grid: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
) -> f64 {
    let disc_old = disc_old.as_array();
    let diff_old = diff_old.as_array();
    let disc_new = disc_new.as_array();
    let diff_new = diff_new.as_array();
    let theta_grid = theta_grid.as_array();
    let weights = weights.as_array();

    let n_items = disc_old.len();
    let n_theta = theta_grid.len();

    (0..n_items)
        .into_par_iter()
        .map(|j| {
            let disc_trans = disc_new[j] / a;
            let diff_trans = a * diff_new[j] + b;

            (0..n_theta)
                .map(|q| {
                    let theta = theta_grid[q];
                    let p_old = sigmoid(disc_old[j] * (theta - diff_old[j]));
                    let p_new = sigmoid(disc_trans * (theta - diff_trans));
                    weights[q] * (p_old - p_new).powi(2)
                })
                .sum::<f64>()
        })
        .sum()
}

/// Compute unsigned area between ICCs for multiple items.
///
/// Returns array of area differences using trapezoidal integration.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn area_between_iccs_batch<'py>(
    py: Python<'py>,
    disc_old: PyReadonlyArray1<f64>,
    diff_old: PyReadonlyArray1<f64>,
    disc_new: PyReadonlyArray1<f64>,
    diff_new: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    theta_grid: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let disc_old = disc_old.as_array();
    let diff_old = diff_old.as_array();
    let disc_new = disc_new.as_array();
    let diff_new = diff_new.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc_old.len();
    let n_theta = theta_grid.len();

    let areas: Vec<f64> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let disc_trans = disc_new[j] / a;
            let diff_trans = a * diff_new[j] + b;

            let diffs: Vec<f64> = (0..n_theta)
                .map(|q| {
                    let theta = theta_grid[q];
                    let p_old = sigmoid(disc_old[j] * (theta - diff_old[j]));
                    let p_new = sigmoid(disc_trans * (theta - diff_trans));
                    (p_old - p_new).abs()
                })
                .collect();

            trapezoidal_integrate(&theta_grid.to_vec(), &diffs)
        })
        .collect();

    let arr: Array1<f64> = areas.into();
    arr.to_pyarray(py)
}

/// Lord-Wingersky recursion for observed score distribution.
///
/// Computes P(X = x | theta) for all possible sum scores x and all theta values.
/// Returns matrix of shape (n_theta, n_items + 1).
#[pyfunction]
pub fn lord_wingersky_recursion<'py>(
    py: Python<'py>,
    disc: PyReadonlyArray1<f64>,
    diff: PyReadonlyArray1<f64>,
    theta_grid: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let disc = disc.as_array();
    let diff = diff.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc.len();
    let n_theta = theta_grid.len();
    let max_score = n_items;

    let results: Vec<Vec<f64>> = (0..n_theta)
        .into_par_iter()
        .map(|q| {
            let theta = theta_grid[q];

            let probs: Vec<f64> = (0..n_items)
                .map(|j| sigmoid(disc[j] * (theta - diff[j])))
                .collect();

            let mut f_prev = vec![0.0; max_score + 1];
            f_prev[0] = 1.0;

            for (j, &p_j) in probs.iter().enumerate() {
                let mut f_curr = vec![0.0; max_score + 1];
                let q_j = 1.0 - p_j;

                for x in 0..=(j + 1) {
                    if x == 0 {
                        f_curr[x] = f_prev[x] * q_j;
                    } else if x == j + 1 {
                        f_curr[x] = f_prev[x - 1] * p_j;
                    } else {
                        f_curr[x] = f_prev[x] * q_j + f_prev[x - 1] * p_j;
                    }
                }
                f_prev = f_curr;
            }

            f_prev
        })
        .collect();

    let mut arr = Array2::zeros((n_theta, max_score + 1));
    for (q, row) in results.into_iter().enumerate() {
        for (x, val) in row.into_iter().enumerate() {
            arr[[q, x]] = val;
        }
    }

    arr.to_pyarray(py)
}

/// Compute test characteristic curves for old and new forms.
///
/// Returns tuple of (tcc_old, tcc_new) arrays of shape (n_theta,).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_tcc_pair<'py>(
    py: Python<'py>,
    disc_old: PyReadonlyArray1<f64>,
    diff_old: PyReadonlyArray1<f64>,
    disc_new: PyReadonlyArray1<f64>,
    diff_new: PyReadonlyArray1<f64>,
    a: f64,
    b: f64,
    theta_grid: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let disc_old = disc_old.as_array();
    let diff_old = diff_old.as_array();
    let disc_new = disc_new.as_array();
    let diff_new = diff_new.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc_old.len();
    let n_theta = theta_grid.len();

    let results: Vec<(f64, f64)> = (0..n_theta)
        .into_par_iter()
        .map(|q| {
            let theta = theta_grid[q];
            let mut tcc_old = 0.0;
            let mut tcc_new = 0.0;

            for j in 0..n_items {
                let p_old = sigmoid(disc_old[j] * (theta - diff_old[j]));
                tcc_old += p_old;

                let disc_trans = disc_new[j] / a;
                let diff_trans = a * diff_new[j] + b;
                let p_new = sigmoid(disc_trans * (theta - diff_trans));
                tcc_new += p_new;
            }

            (tcc_old, tcc_new)
        })
        .collect();

    let tcc_old: Array1<f64> = results.iter().map(|(o, _)| *o).collect();
    let tcc_new: Array1<f64> = results.iter().map(|(_, n)| *n).collect();

    (tcc_old.to_pyarray(py), tcc_new.to_pyarray(py))
}

/// Compute robust z-statistics for drift detection.
///
/// Returns array of z-scores based on MAD.
#[pyfunction]
pub fn robust_z_drift<'py>(
    py: Python<'py>,
    diff_a: PyReadonlyArray1<f64>,
    diff_b: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let diff_a = diff_a.as_array();
    let diff_b = diff_b.as_array();

    let n = diff_a.len();

    let combined: Vec<f64> = (0..n)
        .map(|i| (diff_a[i].powi(2) + diff_b[i].powi(2)).sqrt())
        .collect();

    let mut sorted = combined.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let mut abs_devs: Vec<f64> = combined.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if n.is_multiple_of(2) {
        (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
    } else {
        abs_devs[n / 2]
    };

    let mad_scaled = mad * 1.4826;

    let z_scores: Vec<f64> = if mad_scaled < 1e-10 {
        vec![0.0; n]
    } else {
        combined
            .iter()
            .map(|&x| (x - median) / mad_scaled)
            .collect()
    };

    let arr: Array1<f64> = z_scores.into();
    arr.to_pyarray(py)
}

/// Compute expected scores at each theta point.
#[pyfunction]
pub fn expected_scores<'py>(
    py: Python<'py>,
    disc: PyReadonlyArray1<f64>,
    diff: PyReadonlyArray1<f64>,
    theta_grid: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let disc = disc.as_array();
    let diff = diff.as_array();
    let theta_grid = theta_grid.as_array();

    let n_items = disc.len();
    let n_theta = theta_grid.len();

    let expected: Vec<f64> = (0..n_theta)
        .into_par_iter()
        .map(|q| {
            let theta = theta_grid[q];
            (0..n_items)
                .map(|j| sigmoid(disc[j] * (theta - diff[j])))
                .sum()
        })
        .collect();

    let arr: Array1<f64> = expected.into();
    arr.to_pyarray(py)
}

/// Trapezoidal integration helper.
fn trapezoidal_integrate(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    (0..n - 1)
        .map(|i| (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0)
        .sum()
}

/// Register equating functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haebara_criterion, m)?)?;
    m.add_function(wrap_pyfunction!(area_between_iccs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(lord_wingersky_recursion, m)?)?;
    m.add_function(wrap_pyfunction!(compute_tcc_pair, m)?)?;
    m.add_function(wrap_pyfunction!(robust_z_drift, m)?)?;
    m.add_function(wrap_pyfunction!(expected_scores, m)?)?;
    Ok(())
}
