//! Regularized estimation for sparse MIRT models.
//!
//! This module provides coordinate descent optimization with LASSO, ridge,
//! and elastic net penalties for discovering simple structure in factor loadings.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Soft-thresholding operator for LASSO penalty.
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Coordinate descent M-step for regularized MIRT.
///
/// Optimizes factor loadings with LASSO/ridge/elastic net penalties.
///
/// # Arguments
/// * `r_k` - Expected correct responses (n_items, n_quad)
/// * `n_k` - Expected total responses (n_items, n_quad)
/// * `quad_points` - Quadrature points (n_quad, n_factors)
/// * `loadings` - Current factor loadings (n_items, n_factors)
/// * `intercepts` - Current intercepts (n_items,)
/// * `adaptive_weights` - Adaptive LASSO weights (n_items, n_factors)
/// * `lambda_val` - Regularization strength
/// * `alpha` - Elastic net mixing (1 = LASSO, 0 = ridge)
/// * `max_iter` - Maximum coordinate descent iterations
/// * `tol` - Convergence tolerance
#[pyfunction]
#[pyo3(signature = (r_k, n_k, quad_points, loadings, intercepts, adaptive_weights, lambda_val, alpha, max_iter, tol))]
#[allow(clippy::too_many_arguments)]
pub fn coordinate_descent_mstep_regularized<'py>(
    py: Python<'py>,
    r_k: PyReadonlyArray2<f64>,
    n_k: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray2<f64>,
    loadings: PyReadonlyArray2<f64>,
    intercepts: PyReadonlyArray1<f64>,
    adaptive_weights: PyReadonlyArray2<f64>,
    lambda_val: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let r_k = r_k.as_array();
    let n_k = n_k.as_array();
    let quad_points = quad_points.as_array();
    let loadings_init = loadings.as_array();
    let intercepts_init = intercepts.as_array();
    let adaptive_weights = adaptive_weights.as_array();

    let n_items = r_k.nrows();
    let n_quad = r_k.ncols();
    let n_factors = quad_points.ncols();

    let mut new_loadings = loadings_init.to_owned();
    let mut new_intercepts = intercepts_init.to_owned();

    for _iter in 0..max_iter {
        let loadings_old = new_loadings.clone();

        for j in 0..n_items {
            let r_j = r_k.row(j);
            let n_j = n_k.row(j);

            for k in 0..n_factors {
                let mut partial_z = vec![0.0; n_quad];
                for q in 0..n_quad {
                    let mut z = new_intercepts[j];
                    for f in 0..n_factors {
                        if f != k {
                            z += quad_points[[q, f]] * new_loadings[[j, f]];
                        }
                    }
                    partial_z[q] = z;
                }

                let mut gradient = 0.0;
                let mut hessian = 0.0;

                for q in 0..n_quad {
                    if n_j[q] < EPSILON {
                        continue;
                    }

                    let z = partial_z[q] + quad_points[[q, k]] * new_loadings[[j, k]];
                    let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);
                    let residual = r_j[q] - n_j[q] * p;
                    let x_k = quad_points[[q, k]];

                    gradient += residual * x_k;
                    hessian -= n_j[q] * p * (1.0 - p) * x_k * x_k;
                }

                if hessian.abs() < EPSILON {
                    continue;
                }

                let unpenalized = new_loadings[[j, k]] - gradient / hessian;
                let adaptive_w = adaptive_weights[[j, k]];
                let lam_eff = lambda_val * adaptive_w;

                let new_val = if alpha >= 1.0 - EPSILON {
                    soft_threshold(unpenalized, lam_eff / (-hessian))
                } else if alpha <= EPSILON {
                    unpenalized / (1.0 + 2.0 * lam_eff / (-hessian))
                } else {
                    let lasso_part = alpha * lam_eff;
                    let ridge_part = (1.0 - alpha) * lam_eff;
                    let shrunk = soft_threshold(unpenalized, lasso_part / (-hessian));
                    shrunk / (1.0 + 2.0 * ridge_part / (-hessian))
                };

                new_loadings[[j, k]] = new_val;
            }

            let mut gradient_d = 0.0;
            let mut hessian_d = 0.0;

            for q in 0..n_quad {
                if n_j[q] < EPSILON {
                    continue;
                }

                let mut z = new_intercepts[j];
                for f in 0..n_factors {
                    z += quad_points[[q, f]] * new_loadings[[j, f]];
                }
                let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

                gradient_d += r_j[q] - n_j[q] * p;
                hessian_d -= n_j[q] * p * (1.0 - p);
            }

            if hessian_d.abs() > EPSILON {
                new_intercepts[j] -= gradient_d / hessian_d;
            }
        }

        let mut max_change = 0.0_f64;
        for j in 0..n_items {
            for k in 0..n_factors {
                max_change = max_change.max((new_loadings[[j, k]] - loadings_old[[j, k]]).abs());
            }
        }

        if max_change < tol {
            break;
        }
    }

    (new_loadings.to_pyarray(py), new_intercepts.to_pyarray(py))
}

/// Register regularized estimation functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coordinate_descent_mstep_regularized, m)?)?;
    Ok(())
}
