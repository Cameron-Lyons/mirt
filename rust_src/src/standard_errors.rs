//! Parallel standard error computation for IRT models.
//!
//! Exploits the block diagonal structure of the Hessian matrix
//! since item parameters are independent given the data.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, log_sigmoid, sigmoid};

/// Compute per-item Fisher information (observed information) in parallel.
///
/// Exploits the block diagonal structure: items are independent,
/// so we only need to compute 2x2 blocks (for 2PL: discrimination, difficulty).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_item_se_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    h: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let se_results: Vec<(f64, f64)> = (0..n_items)
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

            let a = disc[j];
            let b = diff[j];

            let ll_center = compute_item_ll(&r_k, &n_k, &quad_points, a, b);

            let ll_a_plus = compute_item_ll(&r_k, &n_k, &quad_points, a + h, b);
            let ll_a_minus = compute_item_ll(&r_k, &n_k, &quad_points, a - h, b);
            let hess_aa = (ll_a_plus - 2.0 * ll_center + ll_a_minus) / (h * h);

            let ll_b_plus = compute_item_ll(&r_k, &n_k, &quad_points, a, b + h);
            let ll_b_minus = compute_item_ll(&r_k, &n_k, &quad_points, a, b - h);
            let hess_bb = (ll_b_plus - 2.0 * ll_center + ll_b_minus) / (h * h);

            let se_a = if hess_aa < -EPSILON {
                (-1.0 / hess_aa).sqrt()
            } else {
                f64::NAN
            };

            let se_b = if hess_bb < -EPSILON {
                (-1.0 / hess_bb).sqrt()
            } else {
                f64::NAN
            };

            (se_a, se_b)
        })
        .collect();

    let se_disc: Array1<f64> = se_results
        .iter()
        .map(|(a, _)| *a)
        .collect::<Vec<_>>()
        .into();
    let se_diff: Array1<f64> = se_results
        .iter()
        .map(|(_, b)| *b)
        .collect::<Vec<_>>()
        .into();

    (se_disc.to_pyarray(py), se_diff.to_pyarray(py))
}

/// Compute expected log-likelihood for a single item
fn compute_item_ll(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &ndarray::ArrayView1<f64>,
    a: f64,
    b: f64,
) -> f64 {
    let n_quad = quad_points.len();
    let mut ll = 0.0;

    for q in 0..n_quad {
        if n_k[q] < EPSILON {
            continue;
        }
        let theta = quad_points[q];
        let z = a * (theta - b);
        let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

        ll += r_k[q] * p.ln() + (n_k[q] - r_k[q]) * (1.0 - p).ln();
    }

    ll
}

/// Compute full observed information matrix with block diagonal structure.
///
/// For 2PL model, the Hessian is block diagonal with 2x2 blocks per item.
/// This function computes the full matrix but exploits the block structure.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_hessian_block_diagonal<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    h: f64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();
    let n_params = n_items * 2;

    let expected_counts: Vec<(Vec<f64>, Vec<f64>)> = (0..n_items)
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

    let blocks: Vec<[[f64; 2]; 2]> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let (ref r_k, ref n_k) = expected_counts[j];
            let a = disc[j];
            let b = diff[j];

            let ll_center = compute_item_ll(r_k, n_k, &quad_points, a, b);

            let ll_a_plus = compute_item_ll(r_k, n_k, &quad_points, a + h, b);
            let ll_a_minus = compute_item_ll(r_k, n_k, &quad_points, a - h, b);
            let hess_aa = (ll_a_plus - 2.0 * ll_center + ll_a_minus) / (h * h);

            let ll_b_plus = compute_item_ll(r_k, n_k, &quad_points, a, b + h);
            let ll_b_minus = compute_item_ll(r_k, n_k, &quad_points, a, b - h);
            let hess_bb = (ll_b_plus - 2.0 * ll_center + ll_b_minus) / (h * h);

            let ll_pp = compute_item_ll(r_k, n_k, &quad_points, a + h, b + h);
            let ll_pm = compute_item_ll(r_k, n_k, &quad_points, a + h, b - h);
            let ll_mp = compute_item_ll(r_k, n_k, &quad_points, a - h, b + h);
            let ll_mm = compute_item_ll(r_k, n_k, &quad_points, a - h, b - h);
            let hess_ab = (ll_pp - ll_pm - ll_mp + ll_mm) / (4.0 * h * h);

            [[hess_aa, hess_ab], [hess_ab, hess_bb]]
        })
        .collect();

    let mut hessian = Array2::zeros((n_params, n_params));

    for (j, block) in blocks.iter().enumerate() {
        let idx_a = j * 2;
        let idx_b = j * 2 + 1;

        hessian[[idx_a, idx_a]] = block[0][0];
        hessian[[idx_a, idx_b]] = block[0][1];
        hessian[[idx_b, idx_a]] = block[1][0];
        hessian[[idx_b, idx_b]] = block[1][1];
    }

    hessian.to_pyarray(py)
}

/// Compute standard errors from observed information matrix.
///
/// Takes the negative inverse of the Hessian and extracts diagonal elements.
#[pyfunction]
pub fn compute_se_from_hessian<'py>(
    py: Python<'py>,
    hessian: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let hessian = hessian.as_array();
    let n_params = hessian.nrows();

    let n_items = n_params / 2;

    let mut se = Array1::zeros(n_params);

    for j in 0..n_items {
        let idx_a = j * 2;
        let idx_b = j * 2 + 1;

        let h_aa = -hessian[[idx_a, idx_a]];
        let h_ab = -hessian[[idx_a, idx_b]];
        let h_bb = -hessian[[idx_b, idx_b]];

        let det = h_aa * h_bb - h_ab * h_ab;

        if det > EPSILON {
            let inv_aa = h_bb / det;
            let inv_bb = h_aa / det;

            se[idx_a] = if inv_aa > 0.0 {
                inv_aa.sqrt()
            } else {
                f64::NAN
            };
            se[idx_b] = if inv_bb > 0.0 {
                inv_bb.sqrt()
            } else {
                f64::NAN
            };
        } else {
            se[idx_a] = f64::NAN;
            se[idx_b] = f64::NAN;
        }
    }

    se.to_pyarray(py)
}

/// Compute complete data log-likelihood for all items.
///
/// Used for finite difference Hessian computation.
#[pyfunction]
pub fn compute_complete_data_ll<'py>(
    _py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> f64 {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let ll: f64 = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut person_ll = 0.0;

            for q in 0..n_quad {
                let w = posterior_weights[[i, q]];
                if w < EPSILON {
                    continue;
                }

                let theta = quad_points[q];
                let mut quad_ll = 0.0;

                for j in 0..n_items {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }

                    let z = disc[j] * (theta - diff[j]);
                    if resp == 1 {
                        quad_ll += log_sigmoid(z);
                    } else {
                        quad_ll += log_sigmoid(-z);
                    }
                }

                person_ll += w * quad_ll;
            }

            person_ll
        })
        .sum();

    ll
}

/// Register standard error functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_item_se_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hessian_block_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(compute_se_from_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(compute_complete_data_ll, m)?)?;
    Ok(())
}
