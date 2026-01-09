//! Person scoring functions (EAP, WLE).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{
    EPSILON, compute_eap_with_se, compute_log_weights, fisher_info_2pl, log_likelihood_2pl_single,
    normalize_log_posterior,
};

/// Compute EAP (Expected A Posteriori) scores
#[pyfunction]
pub fn compute_eap_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
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
    let log_weights = compute_log_weights(&weight_vec);

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_posterior: Vec<f64> = (0..n_quad)
                .map(|q| {
                    log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec)
                        + log_weights[q]
                })
                .collect();

            let posterior = normalize_log_posterior(&log_posterior);
            compute_eap_with_se(&posterior, &quad_vec)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

/// Compute WLE (Weighted Likelihood Estimation) scores
#[pyfunction]
pub fn compute_wle_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    theta_min: f64,
    theta_max: f64,
    tol: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let valid_count = resp_row.iter().filter(|&&r| r >= 0).count();
            if valid_count == 0 {
                return (0.0, f64::INFINITY);
            }

            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let mut a = theta_min;
            let mut b = theta_max;

            while (b - a) > tol {
                let c = b - (b - a) / phi;
                let d = a + (b - a) / phi;

                let wl_c = wle_criterion(&resp_row, c, &disc_vec, &diff_vec, n_items);
                let wl_d = wle_criterion(&resp_row, d, &disc_vec, &diff_vec, n_items);

                if wl_c > wl_d {
                    b = d;
                } else {
                    a = c;
                }
            }

            let theta_wle = (a + b) / 2.0;

            let info = fisher_info_2pl(theta_wle, &disc_vec, &diff_vec);
            let se = if info > EPSILON {
                1.0 / info.sqrt()
            } else {
                f64::INFINITY
            };

            (theta_wle, se)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

/// WLE criterion function (log-likelihood + 0.5 * log(information))
#[inline]
fn wle_criterion(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
    _n_items: usize,
) -> f64 {
    let ll = log_likelihood_2pl_single(responses, theta, discrimination, difficulty);
    let info = fisher_info_2pl(theta, discrimination, difficulty);
    if info > EPSILON {
        ll + 0.5 * info.ln()
    } else {
        ll
    }
}

/// Register scoring functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_eap_scores, m)?)?;
    m.add_function(wrap_pyfunction!(compute_wle_scores, m)?)?;
    Ok(())
}
