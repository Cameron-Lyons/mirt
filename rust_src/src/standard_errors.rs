//! Parallel standard error computation for IRT models.
//!
//! Exploits the block diagonal structure of the Hessian matrix
//! since item parameters are independent given the data.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, log_sigmoid, sigmoid};

/// Compute the exact complete-data Hessian block for one 2PL item.
fn compute_item_hessian_block(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &numpy::ndarray::ArrayView1<f64>,
    a: f64,
    b: f64,
) -> [[f64; 2]; 2] {
    let mut hess_aa = 0.0;
    let mut hess_ab = 0.0;
    let mut hess_bb = 0.0;

    for q in 0..quad_points.len() {
        if n_k[q] < EPSILON {
            continue;
        }
        let centered_theta = quad_points[q] - b;
        let probability = sigmoid(a * centered_theta);
        let curvature = n_k[q] * probability * (1.0 - probability);
        let score_residual = r_k[q] - n_k[q] * probability;

        hess_aa -= curvature * centered_theta * centered_theta;
        hess_ab += curvature * a * centered_theta - score_residual;
        hess_bb -= curvature * a * a;
    }

    [[hess_aa, hess_ab], [hess_ab, hess_bb]]
}

/// Invert a negative 2x2 Hessian block and return marginal standard errors.
fn standard_errors_from_block(block: [[f64; 2]; 2]) -> (f64, f64) {
    let information_aa = -block[0][0];
    let information_ab = -block[0][1];
    let information_bb = -block[1][1];
    let determinant = information_aa * information_bb - information_ab * information_ab;

    if information_aa > EPSILON && information_bb > EPSILON && determinant > EPSILON {
        (
            (information_bb / determinant).sqrt(),
            (information_aa / determinant).sqrt(),
        )
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Compute covariance-aware standard errors from exact 2PL Hessian blocks.
///
/// Exploits the block diagonal structure: items are independent,
/// so each discrimination/difficulty pair requires one 2x2 inversion.
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
    let _ = h;
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
            let block = compute_item_hessian_block(&r_k, &n_k, &quad_points, a, b);
            standard_errors_from_block(block)
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

/// Compute the exact complete-data Hessian with block diagonal structure.
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
    let _ = h;
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
            compute_item_hessian_block(r_k, n_k, &quad_points, a, b)
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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr1;

    #[test]
    fn analytic_hessian_block_matches_closed_form_components() {
        let r_k = [1.5];
        let n_k = [4.0];
        let quad_points = arr1(&[1.25]);
        let a = 1.4;
        let b = -0.25;
        let centered_theta = quad_points[0] - b;
        let probability = sigmoid(a * centered_theta);
        let curvature = n_k[0] * probability * (1.0 - probability);
        let score_residual = r_k[0] - n_k[0] * probability;

        let block = compute_item_hessian_block(&r_k, &n_k, &quad_points.view(), a, b);

        let expected_aa = -curvature * centered_theta * centered_theta;
        let expected_ab = curvature * a * centered_theta - score_residual;
        let expected_bb = -curvature * a * a;
        assert!((block[0][0] - expected_aa).abs() < 1e-14);
        assert!((block[0][1] - expected_ab).abs() < 1e-14);
        assert!((block[1][0] - expected_ab).abs() < 1e-14);
        assert!((block[1][1] - expected_bb).abs() < 1e-14);
    }

    #[test]
    fn standard_errors_invert_the_full_information_block() {
        let (se_a, se_b) = standard_errors_from_block([[-4.0, -1.0], [-1.0, -9.0]]);

        assert!((se_a - (9.0_f64 / 35.0).sqrt()).abs() < 1e-14);
        assert!((se_b - (4.0_f64 / 35.0).sqrt()).abs() < 1e-14);

        let singular = standard_errors_from_block([[-1.0, -1.0], [-1.0, -1.0]]);
        assert!(singular.0.is_nan());
        assert!(singular.1.is_nan());
    }
}
