//! Log-likelihood computation functions.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{log_likelihood_2pl_single, log_likelihood_3pl_single, log_sigmoid};

/// Compute log-likelihoods for all persons at all quadrature points (2PL)
#[pyfunction]
pub fn compute_log_likelihoods_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| log_likelihood_2pl_single(&resp_row, quad_points[q], &disc_vec, &diff_vec))
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

/// Compute log-likelihoods for all persons at all quadrature points (3PL)
#[pyfunction]
pub fn compute_log_likelihoods_3pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let guessing = guessing.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let guess_vec: Vec<f64> = guessing.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    log_likelihood_3pl_single(
                        &resp_row,
                        quad_points[q],
                        &disc_vec,
                        &diff_vec,
                        &guess_vec,
                    )
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

/// Compute log-likelihoods for multidimensional IRT
#[pyfunction]
pub fn compute_log_likelihoods_mirt<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.nrows();
    let n_items = responses.ncols();
    let n_factors = quad_points.ncols();

    let disc_sums: Vec<f64> = (0..n_items).map(|j| discrimination.row(j).sum()).collect();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    let theta_q = quad_points.row(q);
                    let mut ll = 0.0;
                    for (j, &resp) in resp_row.iter().enumerate() {
                        if resp < 0 {
                            continue;
                        }

                        let mut z = 0.0;
                        for f in 0..n_factors {
                            z += discrimination[[j, f]] * theta_q[f];
                        }
                        z -= disc_sums[j] * difficulty[j];

                        if resp == 1 {
                            ll += log_sigmoid(z);
                        } else {
                            ll += log_sigmoid(-z);
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

/// Register likelihood functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_mirt, m)?)?;
    Ok(())
}
