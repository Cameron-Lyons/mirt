//! Log-likelihood computation functions.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::likelihood_cache::cached_likelihoods;
use crate::utils::{EPSILON, log_sigmoid, sigmoid};

/// Fill a binary row with the 2PL log probabilities `[log(1 - p), log(p)]` of logit `z`.
#[inline]
pub(crate) fn log_2pl_row(z: f64, row: &mut [f64]) {
    row[0] = log_sigmoid(-z);
    row[1] = log_sigmoid(z);
}

/// Fill a binary row with clipped 3PL log probabilities for logit `z`.
#[inline]
pub(crate) fn log_3pl_row(z: f64, guessing: f64, row: &mut [f64]) {
    let p = (guessing + (1.0 - guessing) * sigmoid(z)).clamp(EPSILON, 1.0 - EPSILON);
    row[0] = (1.0 - p).ln();
    row[1] = p.ln();
}

/// Compute log-likelihoods for all persons at all quadrature points (2PL)
#[pyfunction]
#[pyo3(signature = (responses, quad_points, discrimination, difficulty))]
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

    let result = py.detach(|| {
        cached_likelihoods(
            responses,
            quad_points.len(),
            &vec![2; responses.ncols()],
            true,
            |q, j, row| log_2pl_row(discrimination[j] * (quad_points[q] - difficulty[j]), row),
        )
    });
    result.into_pyarray(py)
}

/// Compute log-likelihoods for all persons at all quadrature points (3PL)
#[pyfunction]
#[pyo3(signature = (responses, quad_points, discrimination, difficulty, guessing))]
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

    let result = py.detach(|| {
        cached_likelihoods(
            responses,
            quad_points.len(),
            &vec![2; responses.ncols()],
            true,
            |q, j, row| {
                let z = discrimination[j] * (quad_points[q] - difficulty[j]);
                log_3pl_row(z, guessing[j], row);
            },
        )
    });
    result.into_pyarray(py)
}

/// Compute log-likelihoods for multidimensional IRT
#[pyfunction]
#[pyo3(signature = (responses, quad_points, discrimination, difficulty))]
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

    let disc_sums: Vec<f64> = discrimination
        .rows()
        .into_iter()
        .map(|row| row.sum())
        .collect();
    let result = py.detach(|| {
        cached_likelihoods(
            responses,
            quad_points.nrows(),
            &vec![2; responses.ncols()],
            true,
            |q, j, row| {
                let mut z = 0.0;
                for f in 0..quad_points.ncols() {
                    z += discrimination[[j, f]] * quad_points[[q, f]];
                }
                z -= disc_sums[j] * difficulty[j];
                log_2pl_row(z, row);
            },
        )
    });
    result.into_pyarray(py)
}

/// Register likelihood functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_mirt, m)?)?;
    Ok(())
}
