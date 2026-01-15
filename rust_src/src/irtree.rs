//! IRTree model support for decomposing ordinal responses.
//!
//! This module provides functions for expanding ordinal responses to pseudo-items
//! and computing E-step posteriors for multidimensional IRTree models.

use ndarray::{Array1, Array2, Array3};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, logsumexp, sigmoid};

/// Compute log-likelihood for IRTree pseudo-items at all quadrature points.
///
/// # Arguments
/// * `pseudo_responses` - Binary pseudo-item responses (n_persons, n_items, max_nodes)
/// * `valid_mask` - Which pseudo-items are valid (n_persons, n_items, max_nodes)
/// * `trait_assignments` - Trait index for each node (n_items, max_nodes)
/// * `discrimination` - Item discriminations (n_items, max_nodes)
/// * `difficulty` - Item difficulties (n_items, max_nodes)
/// * `quad_points` - Quadrature points (n_quad, n_traits)
///
/// # Returns
/// Log-likelihoods (n_persons, n_quad)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn irtree_log_likelihoods<'py>(
    py: Python<'py>,
    pseudo_responses: PyReadonlyArray3<i32>,
    valid_mask: PyReadonlyArray3<i32>,
    trait_assignments: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let pseudo_responses = pseudo_responses.as_array();
    let valid_mask = valid_mask.as_array();
    let trait_assignments = trait_assignments.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let quad_points = quad_points.as_array();

    let n_persons = pseudo_responses.shape()[0];
    let n_items = pseudo_responses.shape()[1];
    let max_nodes = pseudo_responses.shape()[2];
    let n_quad = quad_points.nrows();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut ll_row = vec![0.0; n_quad];

            for q in 0..n_quad {
                let mut ll = 0.0;

                for j in 0..n_items {
                    for node_idx in 0..max_nodes {
                        if valid_mask[[i, j, node_idx]] == 0 {
                            continue;
                        }

                        let trait_idx = trait_assignments[[j, node_idx]] as usize;
                        let a = discrimination[[j, node_idx]];
                        let b = difficulty[[j, node_idx]];
                        let theta = quad_points[[q, trait_idx]];

                        let z = a * (theta - b);
                        let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

                        let resp = pseudo_responses[[i, j, node_idx]];
                        if resp == 1 {
                            ll += p.ln();
                        } else {
                            ll += (1.0 - p).ln();
                        }
                    }
                }

                ll_row[q] = ll;
            }

            ll_row
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.into_iter().enumerate() {
        for (q, val) in row.into_iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

/// E-step for IRTree models with multivariate normal prior.
///
/// # Arguments
/// * `log_likelihoods` - Log-likelihoods (n_persons, n_quad)
/// * `quad_weights` - Quadrature weights (n_quad,)
/// * `log_prior` - Log prior densities (n_quad,)
///
/// # Returns
/// Tuple of (posterior_weights, marginal_log_likelihood)
#[pyfunction]
pub fn irtree_e_step<'py>(
    py: Python<'py>,
    log_likelihoods: PyReadonlyArray2<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    log_prior: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let log_likelihoods = log_likelihoods.as_array();
    let quad_weights = quad_weights.as_array();
    let log_prior = log_prior.as_array();

    let n_persons = log_likelihoods.nrows();
    let n_quad = log_likelihoods.ncols();

    let log_weights: Vec<f64> = quad_weights.iter().map(|&w| w.ln()).collect();

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut log_joint = vec![0.0; n_quad];

            for q in 0..n_quad {
                log_joint[q] = log_likelihoods[[i, q]] + log_prior[q] + log_weights[q];
            }

            let log_marginal = logsumexp(&log_joint);

            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal)
        })
        .collect();

    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (posterior, log_marg)) in results.into_iter().enumerate() {
        for (q, &w) in posterior.iter().enumerate() {
            posterior_weights[[i, q]] = w;
        }
        marginal_ll[i] = log_marg.exp();
    }

    (posterior_weights.to_pyarray(py), marginal_ll.to_pyarray(py))
}

/// Compute expected counts for IRTree pseudo-items.
///
/// For each item and node, computes:
/// - r_k: Expected number of "1" responses at each quadrature point
/// - n_k: Expected total responses at each quadrature point
#[pyfunction]
pub fn irtree_expected_counts<'py>(
    py: Python<'py>,
    pseudo_responses: PyReadonlyArray3<i32>,
    valid_mask: PyReadonlyArray3<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>) {
    let pseudo_responses = pseudo_responses.as_array();
    let valid_mask = valid_mask.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = pseudo_responses.shape()[0];
    let n_items = pseudo_responses.shape()[1];
    let max_nodes = pseudo_responses.shape()[2];
    let n_quad = posterior_weights.ncols();

    #[allow(clippy::type_complexity)]
    let counts: Vec<((usize, usize), Vec<f64>, Vec<f64>)> = (0..n_items)
        .flat_map(|j| (0..max_nodes).map(move |node_idx| (j, node_idx)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(j, node_idx)| {
            let mut r_k = vec![0.0; n_quad];
            let mut n_k = vec![0.0; n_quad];

            for i in 0..n_persons {
                if valid_mask[[i, j, node_idx]] == 0 {
                    continue;
                }

                let resp = pseudo_responses[[i, j, node_idx]];
                for q in 0..n_quad {
                    let w = posterior_weights[[i, q]];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            ((j, node_idx), r_k, n_k)
        })
        .collect();

    let mut r_k_all = Array3::zeros((n_items, max_nodes, n_quad));
    let mut n_k_all = Array3::zeros((n_items, max_nodes, n_quad));

    for ((j, node_idx), r_k, n_k) in counts {
        for q in 0..n_quad {
            r_k_all[[j, node_idx, q]] = r_k[q];
            n_k_all[[j, node_idx, q]] = n_k[q];
        }
    }

    (r_k_all.to_pyarray(py), n_k_all.to_pyarray(py))
}

/// Compute EAP scores for IRTree traits.
#[pyfunction]
pub fn irtree_eap_scores<'py>(
    py: Python<'py>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();

    let n_persons = posterior_weights.nrows();
    let n_traits = quad_points.ncols();

    let scores: Vec<(Vec<f64>, Vec<f64>)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut theta = vec![0.0; n_traits];
            let mut theta_se = vec![0.0; n_traits];

            for k in 0..n_traits {
                let mean: f64 = (0..quad_points.nrows())
                    .map(|q| posterior_weights[[i, q]] * quad_points[[q, k]])
                    .sum();
                theta[k] = mean;

                let variance: f64 = (0..quad_points.nrows())
                    .map(|q| posterior_weights[[i, q]] * (quad_points[[q, k]] - mean).powi(2))
                    .sum();
                theta_se[k] = variance.sqrt();
            }

            (theta, theta_se)
        })
        .collect();

    let mut theta_out = Array2::zeros((n_persons, n_traits));
    let mut se_out = Array2::zeros((n_persons, n_traits));

    for (i, (theta, se)) in scores.into_iter().enumerate() {
        for k in 0..n_traits {
            theta_out[[i, k]] = theta[k];
            se_out[[i, k]] = se[k];
        }
    }

    (theta_out.to_pyarray(py), se_out.to_pyarray(py))
}

/// Log multivariate normal density.
#[pyfunction]
pub fn log_mvn_density<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    mean: PyReadonlyArray1<f64>,
    _cov_chol_diag: PyReadonlyArray1<f64>,
    cov_inv: PyReadonlyArray2<f64>,
    log_det: f64,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let mean = mean.as_array();
    let cov_inv = cov_inv.as_array();

    let n = x.nrows();
    let d = x.ncols();

    let log_densities: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut maha = 0.0;
            for j in 0..d {
                for k in 0..d {
                    let diff_j = x[[i, j]] - mean[j];
                    let diff_k = x[[i, k]] - mean[k];
                    maha += diff_j * cov_inv[[j, k]] * diff_k;
                }
            }

            let log_norm = -0.5 * ((d as f64) * (2.0 * std::f64::consts::PI).ln() + log_det);
            log_norm - 0.5 * maha
        })
        .collect();

    Array1::from(log_densities).to_pyarray(py)
}

/// Register IRTree functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(irtree_log_likelihoods, m)?)?;
    m.add_function(wrap_pyfunction!(irtree_e_step, m)?)?;
    m.add_function(wrap_pyfunction!(irtree_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(irtree_eap_scores, m)?)?;
    m.add_function(wrap_pyfunction!(log_mvn_density, m)?)?;
    Ok(())
}
