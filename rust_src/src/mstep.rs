//! Parallel M-step optimization for various IRT models.
//!
//! This module provides parallelized M-step computation using Rayon,
//! enabling efficient parameter estimation for large-scale IRT analysis.

use ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Parallel M-step optimization for dichotomous items (2PL/3PL).
///
/// Uses Newton-Raphson optimization for each item in parallel.
/// Each item's parameters are optimized independently given the posterior weights.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights, quad_points, discrimination, difficulty, max_iter, tol, disc_bounds, diff_bounds, damping, regularization))]
#[allow(clippy::too_many_arguments)]
pub fn m_step_dichotomous_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    damping: f64,
    regularization: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc_init = discrimination.as_array();
    let diff_init = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let new_params: Vec<(f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut r_k = vec![0.0; n_quad];
            let mut n_k = vec![0.0; n_quad];

            for i in 0..n_persons {
                let resp = responses[[i, j]];
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

            let mut a = disc_init[j];
            let mut b = diff_init[j];

            for _ in 0..max_iter {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut hess_aa = 0.0;
                let mut hess_bb = 0.0;
                let mut hess_ab = 0.0;

                for q in 0..n_quad {
                    if n_k[q] < EPSILON {
                        continue;
                    }
                    let theta = quad_points[q];
                    let z = a * (theta - b);
                    let p = sigmoid(z);
                    let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);

                    let residual = r_k[q] - n_k[q] * p_clipped;

                    grad_a += residual * (theta - b);
                    grad_b += -residual * a;

                    let info = n_k[q] * p_clipped * (1.0 - p_clipped);
                    hess_aa += -info * (theta - b) * (theta - b);
                    hess_bb += -info * a * a;
                    hess_ab += info * a * (theta - b);
                }

                hess_aa -= regularization;
                hess_bb -= regularization;

                let det = hess_aa * hess_bb - hess_ab * hess_ab;
                if det.abs() < EPSILON {
                    break;
                }

                let delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det;
                let delta_b = (-hess_ab * grad_a + hess_aa * grad_b) / det;

                a = (a - delta_a * damping).clamp(disc_bounds.0, disc_bounds.1);
                b = (b - delta_b * damping).clamp(diff_bounds.0, diff_bounds.1);

                if delta_a.abs() < tol && delta_b.abs() < tol {
                    break;
                }
            }

            (a, b)
        })
        .collect();

    let disc_new: Array1<f64> = new_params
        .iter()
        .map(|(a, _)| *a)
        .collect::<Vec<_>>()
        .into();
    let diff_new: Array1<f64> = new_params
        .iter()
        .map(|(_, b)| *b)
        .collect::<Vec<_>>()
        .into();

    (disc_new.to_pyarray(py), diff_new.to_pyarray(py))
}

/// Parallel M-step for 3PL model including guessing parameter.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights, quad_points, discrimination, difficulty, guessing, max_iter, tol, disc_bounds, diff_bounds, guess_bounds, damping_ab, damping_c, regularization, regularization_c))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn m_step_3pl_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    guess_bounds: (f64, f64),
    damping_ab: f64,
    damping_c: f64,
    regularization: f64,
    regularization_c: f64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc_init = discrimination.as_array();
    let diff_init = difficulty.as_array();
    let guess_init = guessing.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let new_params: Vec<(f64, f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut r_k = vec![0.0; n_quad];
            let mut n_k = vec![0.0; n_quad];

            for i in 0..n_persons {
                let resp = responses[[i, j]];
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

            let mut a = disc_init[j];
            let mut b = diff_init[j];
            let mut c = guess_init[j];

            for _ in 0..max_iter {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut hess_aa = 0.0;
                let mut hess_bb = 0.0;

                for q in 0..n_quad {
                    if n_k[q] < EPSILON {
                        continue;
                    }
                    let theta = quad_points[q];
                    let z = a * (theta - b);
                    let p_star = sigmoid(z);
                    let p = c + (1.0 - c) * p_star;
                    let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);

                    let dp_da = (1.0 - c) * p_star * (1.0 - p_star) * (theta - b);
                    let dp_db = -(1.0 - c) * p_star * (1.0 - p_star) * a;

                    let residual = r_k[q] - n_k[q] * p_clipped;

                    grad_a += residual * dp_da / (p_clipped * (1.0 - p_clipped) + EPSILON);
                    grad_b += residual * dp_db / (p_clipped * (1.0 - p_clipped) + EPSILON);

                    let info = n_k[q] * p_clipped * (1.0 - p_clipped);
                    hess_aa -= info * dp_da * dp_da / (p_clipped * (1.0 - p_clipped) + EPSILON);
                    hess_bb -= info * dp_db * dp_db / (p_clipped * (1.0 - p_clipped) + EPSILON);
                }

                hess_aa -= regularization;
                hess_bb -= regularization;

                if hess_aa.abs() > EPSILON {
                    a = (a - grad_a / hess_aa * damping_ab).clamp(disc_bounds.0, disc_bounds.1);
                }
                if hess_bb.abs() > EPSILON {
                    b = (b - grad_b / hess_bb * damping_ab).clamp(diff_bounds.0, diff_bounds.1);
                }

                let mut grad_c = 0.0;
                let mut hess_cc = 0.0;

                for q in 0..n_quad {
                    if n_k[q] < EPSILON {
                        continue;
                    }
                    let theta = quad_points[q];
                    let z = a * (theta - b);
                    let p_star = sigmoid(z);
                    let p = c + (1.0 - c) * p_star;
                    let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);

                    let dp_dc = 1.0 - p_star;
                    let residual = r_k[q] - n_k[q] * p_clipped;

                    grad_c += residual * dp_dc / (p_clipped * (1.0 - p_clipped) + EPSILON);
                    hess_cc -= n_k[q] * dp_dc * dp_dc / (p_clipped * (1.0 - p_clipped) + EPSILON);
                }

                hess_cc -= regularization_c;

                if hess_cc.abs() > EPSILON {
                    c = (c - grad_c / hess_cc * damping_c).clamp(guess_bounds.0, guess_bounds.1);
                }

                if grad_a.abs() < tol && grad_b.abs() < tol && grad_c.abs() < tol {
                    break;
                }
            }

            (a, b, c)
        })
        .collect();

    let disc_new: Array1<f64> = new_params
        .iter()
        .map(|(a, _, _)| *a)
        .collect::<Vec<_>>()
        .into();
    let diff_new: Array1<f64> = new_params
        .iter()
        .map(|(_, b, _)| *b)
        .collect::<Vec<_>>()
        .into();
    let guess_new: Array1<f64> = new_params
        .iter()
        .map(|(_, _, c)| *c)
        .collect::<Vec<_>>()
        .into();

    (
        disc_new.to_pyarray(py),
        diff_new.to_pyarray(py),
        guess_new.to_pyarray(py),
    )
}

/// Compute expected counts for dichotomous items in parallel.
///
/// Returns r_k (correct responses) and n_k (total responses) per quadrature point.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights))]
pub fn compute_expected_counts_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = posterior_weights.ncols();

    let counts: Vec<(Vec<f64>, Vec<f64>)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut r_k = vec![0.0; n_quad];
            let mut n_k = vec![0.0; n_quad];

            for i in 0..n_persons {
                let resp = responses[[i, j]];
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

            (r_k, n_k)
        })
        .collect();

    let mut r_k_all = ndarray::Array2::zeros((n_items, n_quad));
    let mut n_k_all = ndarray::Array2::zeros((n_items, n_quad));

    for (j, (r_k, n_k)) in counts.into_iter().enumerate() {
        for q in 0..n_quad {
            r_k_all[[j, q]] = r_k[q];
            n_k_all[[j, q]] = n_k[q];
        }
    }

    (r_k_all.to_pyarray(py), n_k_all.to_pyarray(py))
}

/// Register M-step functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(m_step_dichotomous_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(m_step_3pl_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_parallel, m)?)?;
    Ok(())
}
