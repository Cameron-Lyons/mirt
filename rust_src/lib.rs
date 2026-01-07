//! High-performance Rust backend for MIRT (Multidimensional Item Response Theory)
//!
//! This module provides optimized implementations of computationally intensive
//! IRT algorithms using Rust with PyO3 bindings.

use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

// ============================================================================
// Constants
// ============================================================================

const LOG_2_PI: f64 = 1.8378770664093453; // ln(2*pi)
const EPSILON: f64 = 1e-10;

// ============================================================================
// Utility Functions
// ============================================================================

/// Numerically stable log-sum-exp
#[inline]
fn logsumexp(arr: &[f64]) -> f64 {
    if arr.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    let sum: f64 = arr.iter().map(|x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Logistic sigmoid function
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Log of logistic sigmoid (numerically stable)
#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Clip value to range
#[inline]
fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

// ============================================================================
// E-Step Computations (Critical Priority)
// ============================================================================

/// Compute log-likelihood for 2PL model at single theta point
#[inline]
fn log_likelihood_2pl_single(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
) -> f64 {
    let mut ll = 0.0;
    for (j, &resp) in responses.iter().enumerate() {
        if resp < 0 {
            continue; // Missing response
        }
        let z = discrimination[j] * (theta - difficulty[j]);
        if resp == 1 {
            ll += log_sigmoid(z);
        } else {
            ll += log_sigmoid(-z);
        }
    }
    ll
}

/// Compute log-likelihood for 3PL model at single theta point
#[inline]
fn log_likelihood_3pl_single(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
    guessing: &[f64],
) -> f64 {
    let mut ll = 0.0;
    for (j, &resp) in responses.iter().enumerate() {
        if resp < 0 {
            continue;
        }
        let p_star = sigmoid(discrimination[j] * (theta - difficulty[j]));
        let p = guessing[j] + (1.0 - guessing[j]) * p_star;
        let p_clipped = clip(p, EPSILON, 1.0 - EPSILON);
        if resp == 1 {
            ll += p_clipped.ln();
        } else {
            ll += (1.0 - p_clipped).ln();
        }
    }
    ll
}

/// Compute log-likelihoods for all persons at all quadrature points (2PL)
/// This is the critical E-step bottleneck
#[pyfunction]
fn compute_log_likelihoods_2pl<'py>(
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
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    // Parallel computation over persons
    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    log_likelihood_2pl_single(
                        &resp_row,
                        quad_points[q],
                        &disc_vec,
                        &diff_vec,
                    )
                })
                .collect()
        })
        .collect();

    // Convert to ndarray
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
fn compute_log_likelihoods_3pl<'py>(
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

/// Compute log-likelihoods for multidimensional 2PL (MIRT)
#[pyfunction]
fn compute_log_likelihoods_mirt<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray2<f64>,  // (n_quad, n_factors)
    discrimination: PyReadonlyArray2<f64>,  // (n_items, n_factors)
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

    // Pre-compute discrimination sums for efficiency
    let disc_sums: Vec<f64> = (0..n_items)
        .map(|j| discrimination.row(j).sum())
        .collect();

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
                        // z = a_j . theta - sum(a_j) * d_j
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

/// Complete E-step computation with posterior weights
#[pyfunction]
fn e_step_complete<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
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

    // Pre-compute log prior for each quadrature point
    let log_prior: Vec<f64> = quad_vec
        .iter()
        .map(|&theta| {
            let z = (theta - prior_mean) / prior_var.sqrt();
            -0.5 * (LOG_2_PI + prior_var.ln() + z * z)
        })
        .collect();

    let log_weights: Vec<f64> = weight_vec.iter().map(|w| (w + EPSILON).ln()).collect();

    // Parallel computation
    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            // Compute log joint for each quadrature point
            let log_joint: Vec<f64> = (0..n_quad)
                .map(|q| {
                    let ll = log_likelihood_2pl_single(
                        &resp_row,
                        quad_vec[q],
                        &disc_vec,
                        &diff_vec,
                    );
                    ll + log_prior[q] + log_weights[q]
                })
                .collect();

            // Log marginal likelihood
            let log_marginal = logsumexp(&log_joint);

            // Posterior weights
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal.exp())
        })
        .collect();

    // Convert to arrays
    let mut posterior_weights = Array2::zeros((n_persons, n_quad));
    let mut marginal_ll = Array1::zeros(n_persons);

    for (i, (post, marg)) in results.iter().enumerate() {
        for (q, &p) in post.iter().enumerate() {
            posterior_weights[[i, q]] = p;
        }
        marginal_ll[i] = *marg;
    }

    (posterior_weights.to_pyarray(py), marginal_ll.to_pyarray(py))
}

// ============================================================================
// M-Step Helper Computations
// ============================================================================

/// Compute r_k (expected counts) for dichotomous items
#[pyfunction]
fn compute_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
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

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

/// Compute r_kc (expected counts per category) for polytomous items
#[pyfunction]
fn compute_expected_counts_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    n_categories: usize,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_kc = Array2::zeros((n_quad, n_categories));

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 || resp as usize >= n_categories {
            continue;
        }
        for q in 0..n_quad {
            r_kc[[q, resp as usize]] += posterior_weights[[i, q]];
        }
    }

    r_kc.to_pyarray(py)
}

// ============================================================================
// SIBTEST Computations (High Priority)
// ============================================================================

/// Compute SIBTEST beta statistic efficiently
#[pyfunction]
fn sibtest_compute_beta<'py>(
    py: Python<'py>,
    ref_data: PyReadonlyArray2<i32>,
    focal_data: PyReadonlyArray2<i32>,
    ref_scores: PyReadonlyArray1<i32>,
    focal_scores: PyReadonlyArray1<i32>,
    suspect_items: PyReadonlyArray1<i32>,
) -> (f64, f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let ref_data = ref_data.as_array();
    let focal_data = focal_data.as_array();
    let ref_scores = ref_scores.as_array();
    let focal_scores = focal_scores.as_array();
    let suspect_items = suspect_items.as_array();

    // Get unique scores
    let all_scores: Vec<i32> = ref_scores.iter().chain(focal_scores.iter()).cloned().collect();
    let mut unique_scores: Vec<i32> = all_scores.clone();
    unique_scores.sort();
    unique_scores.dedup();

    let suspect_vec: Vec<usize> = suspect_items.iter().map(|&x| x as usize).collect();

    // Compute beta_k and n_k for each score level
    let results: Vec<(f64, f64)> = unique_scores
        .par_iter()
        .map(|&k| {
            // Find persons at this score level
            let ref_at_k: Vec<usize> = ref_scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == k)
                .map(|(i, _)| i)
                .collect();

            let focal_at_k: Vec<usize> = focal_scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s == k)
                .map(|(i, _)| i)
                .collect();

            let n_ref_k = ref_at_k.len();
            let n_focal_k = focal_at_k.len();

            if n_ref_k == 0 || n_focal_k == 0 {
                return (f64::NAN, 0.0);
            }

            // Compute mean suspect scores for each group at this level
            let mean_ref_k: f64 = ref_at_k
                .iter()
                .map(|&i| {
                    suspect_vec.iter().map(|&j| ref_data[[i, j]] as f64).sum::<f64>()
                })
                .sum::<f64>() / n_ref_k as f64;

            let mean_focal_k: f64 = focal_at_k
                .iter()
                .map(|&i| {
                    suspect_vec.iter().map(|&j| focal_data[[i, j]] as f64).sum::<f64>()
                })
                .sum::<f64>() / n_focal_k as f64;

            let beta_k = mean_ref_k - mean_focal_k;
            let weight = 2.0 * n_ref_k as f64 * n_focal_k as f64 / (n_ref_k + n_focal_k) as f64;

            (beta_k, weight)
        })
        .collect();

    // Filter valid results
    let valid: Vec<(f64, f64)> = results.into_iter().filter(|(b, _)| !b.is_nan()).collect();

    if valid.is_empty() {
        return (f64::NAN, f64::NAN, Array1::zeros(0).to_pyarray(py), Array1::zeros(0).to_pyarray(py));
    }

    let beta_k_arr: Array1<f64> = Array1::from(valid.iter().map(|(b, _)| *b).collect::<Vec<_>>());
    let n_k_arr: Array1<f64> = Array1::from(valid.iter().map(|(_, n)| *n).collect::<Vec<_>>());

    let total_weight: f64 = n_k_arr.sum();
    let beta: f64 = beta_k_arr.iter().zip(n_k_arr.iter()).map(|(&b, &n)| b * n).sum::<f64>() / total_weight;

    // Compute standard error
    let weighted_mean = beta;
    let weighted_var: f64 = beta_k_arr
        .iter()
        .zip(n_k_arr.iter())
        .map(|(&b, &n)| n * (b - weighted_mean).powi(2))
        .sum::<f64>() / total_weight;

    let n_total = (ref_scores.len() + focal_scores.len()) as f64;
    let se = (weighted_var / n_total).sqrt();

    (beta, se, beta_k_arr.to_pyarray(py), n_k_arr.to_pyarray(py))
}

/// Run SIBTEST for all items in parallel
#[pyfunction]
fn sibtest_all_items<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<i32>,
    groups: PyReadonlyArray1<i32>,
    anchor_items: Option<PyReadonlyArray1<i32>>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let data = data.as_array();
    let groups = groups.as_array();

    let n_items = data.ncols();

    // Determine reference and focal groups
    let mut unique_groups: Vec<i32> = groups.iter().cloned().collect();
    unique_groups.sort();
    unique_groups.dedup();
    let ref_group = unique_groups[0];
    let focal_group = unique_groups[1];

    let ref_mask: Vec<bool> = groups.iter().map(|&g| g == ref_group).collect();
    let focal_mask: Vec<bool> = groups.iter().map(|&g| g == focal_group).collect();

    let anchor_set: Option<Vec<usize>> = anchor_items.map(|a| {
        a.as_array().iter().map(|&x| x as usize).collect()
    });

    // Parallel computation for each item
    let results: Vec<(f64, f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|item_idx| {
            // Determine matching items for this item
            let matching: Vec<usize> = match &anchor_set {
                Some(anchors) => anchors.iter().filter(|&&j| j != item_idx).cloned().collect(),
                None => (0..n_items).filter(|&j| j != item_idx).collect(),
            };

            if matching.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }

            // Compute matching scores
            let ref_scores: Vec<i32> = ref_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_ref)| is_ref)
                .map(|(i, _)| matching.iter().map(|&j| data[[i, j]]).sum())
                .collect();

            let focal_scores: Vec<i32> = focal_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_focal)| is_focal)
                .map(|(i, _)| matching.iter().map(|&j| data[[i, j]]).sum())
                .collect();

            // Get unique score levels
            let all_scores: Vec<i32> = ref_scores.iter().chain(focal_scores.iter()).cloned().collect();
            let mut unique_scores = all_scores.clone();
            unique_scores.sort();
            unique_scores.dedup();

            // Extract reference and focal data
            let ref_data: Vec<Vec<i32>> = ref_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_ref)| is_ref)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            let focal_data: Vec<Vec<i32>> = focal_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_focal)| is_focal)
                .map(|(i, _)| data.row(i).to_vec())
                .collect();

            // Compute beta
            let mut beta_k_vec = Vec::new();
            let mut n_k_vec = Vec::new();

            for &k in &unique_scores {
                let ref_at_k: Vec<usize> = ref_scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == k)
                    .map(|(i, _)| i)
                    .collect();

                let focal_at_k: Vec<usize> = focal_scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == k)
                    .map(|(i, _)| i)
                    .collect();

                let n_ref_k = ref_at_k.len();
                let n_focal_k = focal_at_k.len();

                if n_ref_k > 0 && n_focal_k > 0 {
                    let mean_ref: f64 = ref_at_k.iter().map(|&i| ref_data[i][item_idx] as f64).sum::<f64>() / n_ref_k as f64;
                    let mean_focal: f64 = focal_at_k.iter().map(|&i| focal_data[i][item_idx] as f64).sum::<f64>() / n_focal_k as f64;

                    beta_k_vec.push(mean_ref - mean_focal);
                    n_k_vec.push(2.0 * n_ref_k as f64 * n_focal_k as f64 / (n_ref_k + n_focal_k) as f64);
                }
            }

            if beta_k_vec.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }

            let total_weight: f64 = n_k_vec.iter().sum();
            let beta: f64 = beta_k_vec.iter().zip(n_k_vec.iter()).map(|(&b, &n)| b * n).sum::<f64>() / total_weight;

            // Standard error
            let weighted_var: f64 = beta_k_vec
                .iter()
                .zip(n_k_vec.iter())
                .map(|(&b, &n)| n * (b - beta).powi(2))
                .sum::<f64>() / total_weight;

            let n_total = (ref_scores.len() + focal_scores.len()) as f64;
            let se = (weighted_var / n_total).sqrt();

            // Z-statistic and p-value
            let z = if se > EPSILON { beta / se } else { f64::NAN };
            let p_value = if z.is_nan() {
                f64::NAN
            } else {
                2.0 * (1.0 - normal_cdf(z.abs()))
            };

            (beta, z, p_value)
        })
        .collect();

    let betas: Array1<f64> = Array1::from(results.iter().map(|(b, _, _)| *b).collect::<Vec<_>>());
    let zs: Array1<f64> = Array1::from(results.iter().map(|(_, z, _)| *z).collect::<Vec<_>>());
    let p_values: Array1<f64> = Array1::from(results.iter().map(|(_, _, p)| *p).collect::<Vec<_>>());

    (betas.to_pyarray(py), zs.to_pyarray(py), p_values.to_pyarray(py))
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 { 1.0 - p } else { p }
}

// ============================================================================
// GRM/GPCM Simulation (High Priority)
// ============================================================================

/// Simulate responses from Graded Response Model
#[pyfunction]
fn simulate_grm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items)
        .map(|i| thresholds.row(i).to_vec())
        .collect();

    // Parallel simulation
    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    // Compute cumulative probabilities
                    let mut cum_probs = vec![1.0; n_categories];
                    for k in 0..(n_categories - 1) {
                        let z = disc_vec[j] * (theta_i - thresh_vec[j][k]);
                        cum_probs[k + 1] = sigmoid(z);
                    }

                    // Convert to category probabilities
                    let mut cat_probs = vec![0.0; n_categories];
                    for k in 0..n_categories {
                        let next = if k < n_categories - 1 { cum_probs[k + 1] } else { 0.0 };
                        cat_probs[k] = (cum_probs[k] - next).max(0.0);
                    }

                    // Normalize
                    let sum: f64 = cat_probs.iter().sum();
                    if sum > EPSILON {
                        for p in &mut cat_probs {
                            *p /= sum;
                        }
                    }

                    // Sample category
                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Simulate responses from GPCM
#[pyfunction]
fn simulate_gpcm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items)
        .map(|i| thresholds.row(i).to_vec())
        .collect();

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    // Compute numerators for each category
                    let mut numerators = vec![0.0; n_categories];
                    for k in 0..n_categories {
                        let mut cumsum = 0.0;
                        for v in 0..k {
                            cumsum += disc_vec[j] * (theta_i - thresh_vec[j][v]);
                        }
                        numerators[k] = cumsum.exp();
                    }

                    // Normalize to get probabilities
                    let sum: f64 = numerators.iter().sum();
                    let cat_probs: Vec<f64> = numerators.iter().map(|&n| n / sum).collect();

                    // Sample category
                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Simulate dichotomous responses (2PL/3PL)
#[pyfunction]
fn simulate_dichotomous<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: Option<PyReadonlyArray1<f64>>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let guess_vec: Vec<f64> = guessing
        .map(|g| g.as_array().to_vec())
        .unwrap_or_else(|| vec![0.0; n_items]);

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let z = disc_vec[j] * (theta_i - diff_vec[j]);
                    let p_star = sigmoid(z);
                    let p = guess_vec[j] + (1.0 - guess_vec[j]) * p_star;

                    let u: f64 = rng.random();
                    if u < p { 1 } else { 0 }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// Plausible Values Generation (Medium Priority)
// ============================================================================

/// Generate plausible values using posterior sampling
#[pyfunction]
fn generate_plausible_values_posterior<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    jitter_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
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

    let log_weights: Vec<f64> = weight_vec.iter().map(|&w| (w + EPSILON).ln()).collect();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let normal = Normal::new(0.0, jitter_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            // Compute log posterior
            let log_likes: Vec<f64> = (0..n_quad)
                .map(|q| {
                    log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec)
                })
                .collect();

            let log_posterior: Vec<f64> = log_likes
                .iter()
                .zip(log_weights.iter())
                .map(|(&ll, &lw)| ll + lw)
                .collect();

            // Normalize
            let max_lp = log_posterior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let posterior: Vec<f64> = log_posterior.iter().map(|&lp| (lp - max_lp).exp()).collect();
            let sum: f64 = posterior.iter().sum();
            let posterior: Vec<f64> = posterior.iter().map(|&p| p / sum).collect();

            // Generate plausible values
            (0..n_plausible)
                .map(|_| {
                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    let mut idx = n_quad - 1;
                    for (q, &p) in posterior.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            idx = q;
                            break;
                        }
                    }
                    quad_vec[idx] + rng.sample(normal)
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Generate plausible values using MCMC
#[pyfunction]
fn generate_plausible_values_mcmc<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    n_plausible: usize,
    n_iter: usize,
    proposal_sd: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let pvs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let proposal_dist = Normal::new(0.0, proposal_sd).unwrap();
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let mut theta = 0.0;
            let mut results = Vec::with_capacity(n_plausible);

            for p in 0..n_plausible {
                for _ in 0..n_iter {
                    let proposal = theta + rng.sample(proposal_dist);

                    let ll_current = log_likelihood_2pl_single(&resp_row, theta, &disc_vec, &diff_vec);
                    let ll_proposal = log_likelihood_2pl_single(&resp_row, proposal, &disc_vec, &diff_vec);

                    // Standard normal prior
                    let prior_current = -0.5 * theta * theta;
                    let prior_proposal = -0.5 * proposal * proposal;

                    let log_alpha = (ll_proposal + prior_proposal) - (ll_current + prior_current);

                    let u: f64 = rng.random();
                    if u.ln() < log_alpha {
                        theta = proposal;
                    }
                }
                results.push(theta);
            }
            results
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_plausible));
    for (i, row) in pvs.iter().enumerate() {
        for (p, &val) in row.iter().enumerate() {
            result[[i, p]] = val;
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// Model Fit - Bivariate Margins (Medium Priority)
// ============================================================================

/// Compute observed univariate and bivariate margins
#[pyfunction]
fn compute_observed_margins<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    // Univariate margins
    let obs_uni: Array1<f64> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let valid: Vec<f64> = responses
                .column(j)
                .iter()
                .filter(|&&r| r >= 0)
                .map(|&r| r as f64)
                .collect();
            if valid.is_empty() {
                0.0
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        })
        .collect::<Vec<f64>>()
        .into();

    // Bivariate margins (parallel over item pairs)
    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum = 0.0;
            let mut count = 0;
            for p in 0..n_persons {
                let ri = responses[[p, i]];
                let rj = responses[[p, j]];
                if ri >= 0 && rj >= 0 {
                    sum += (ri * rj) as f64;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            ((i, j), mean)
        })
        .collect();

    let mut obs_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        obs_bi[[i, j]] = val;
        obs_bi[[j, i]] = val;
    }

    (obs_uni.to_pyarray(py), obs_bi.to_pyarray(py))
}

/// Compute expected univariate and bivariate margins under model
#[pyfunction]
fn compute_expected_margins<'py>(
    py: Python<'py>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_quad = quad_points.len();
    let n_items = discrimination.len();

    // Pre-compute probabilities at each quadrature point
    let probs: Vec<Vec<f64>> = (0..n_items)
        .map(|j| {
            (0..n_quad)
                .map(|q| sigmoid(discrimination[j] * (quad_points[q] - difficulty[j])))
                .collect()
        })
        .collect();

    // Univariate expected margins
    let exp_uni: Array1<f64> = (0..n_items)
        .map(|j| {
            probs[j]
                .iter()
                .zip(quad_weights.iter())
                .map(|(&p, &w)| p * w)
                .sum()
        })
        .collect::<Vec<f64>>()
        .into();

    // Bivariate expected margins
    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let exp: f64 = (0..n_quad)
                .map(|q| probs[i][q] * probs[j][q] * quad_weights[q])
                .sum();
            ((i, j), exp)
        })
        .collect();

    let mut exp_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        exp_bi[[i, j]] = val;
        exp_bi[[j, i]] = val;
    }

    (exp_uni.to_pyarray(py), exp_bi.to_pyarray(py))
}

// ============================================================================
// Bootstrap Helpers (High Priority)
// ============================================================================

/// Generate bootstrap sample indices
#[pyfunction]
fn generate_bootstrap_indices<'py>(
    py: Python<'py>,
    n_persons: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let indices: Vec<Vec<i64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(seed + b as u64);
            (0..n_persons).map(|_| rng.random_range(0..n_persons as i64)).collect()
        })
        .collect();

    let mut result = Array2::zeros((n_bootstrap, n_persons));
    for (b, row) in indices.iter().enumerate() {
        for (i, &val) in row.iter().enumerate() {
            result[[b, i]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Resample responses matrix
#[pyfunction]
fn resample_responses<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    indices: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let indices = indices.as_array();

    let n_sample = indices.len();
    let n_items = responses.ncols();

    let mut result = Array2::zeros((n_sample, n_items));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_items {
            result[[i, j]] = responses[[idx as usize, j]];
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// Imputation Helpers (Medium Priority)
// ============================================================================

/// Impute missing responses using model probabilities
#[pyfunction]
fn impute_from_probabilities<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputed: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let orig = responses[[i, j]];
                    if orig != missing_code {
                        orig
                    } else {
                        let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                        let u: f64 = rng.random();
                        if u < p { 1 } else { 0 }
                    }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in imputed.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Multiple imputation in parallel
#[pyfunction]
fn multiple_imputation<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_mean: PyReadonlyArray1<f64>,
    theta_se: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    n_imputations: usize,
    seed: u64,
) -> Bound<'py, PyArray3<i32>> {
    let responses = responses.as_array();
    let theta_mean = theta_mean.as_array();
    let theta_se = theta_se.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputations: Vec<Vec<Vec<i32>>> = (0..n_imputations)
        .into_par_iter()
        .map(|m| {
            let base_seed = seed + (m * n_persons) as u64;

            (0..n_persons)
                .map(|i| {
                    let mut rng = Pcg64::seed_from_u64(base_seed + i as u64);

                    // Draw theta from posterior
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let theta_i = theta_mean[i] + rng.sample(normal) * theta_se[i];

                    (0..n_items)
                        .map(|j| {
                            let orig = responses[[i, j]];
                            if orig != missing_code {
                                orig
                            } else {
                                let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                                let u: f64 = rng.random();
                                if u < p { 1 } else { 0 }
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut result = Array3::zeros((n_imputations, n_persons, n_items));
    for (m, imp) in imputations.iter().enumerate() {
        for (i, row) in imp.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[m, i, j]] = val;
            }
        }
    }

    result.to_pyarray(py)
}

// ============================================================================
// Scoring Functions
// ============================================================================

/// Compute EAP (Expected A Posteriori) scores
#[pyfunction]
fn compute_eap_scores<'py>(
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

    let log_weights: Vec<f64> = weight_vec.iter().map(|&w| (w + EPSILON).ln()).collect();

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            // Compute log posterior
            let log_likes: Vec<f64> = (0..n_quad)
                .map(|q| {
                    log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec)
                })
                .collect();

            let log_posterior: Vec<f64> = log_likes
                .iter()
                .zip(log_weights.iter())
                .map(|(&ll, &lw)| ll + lw)
                .collect();

            // Normalize
            let max_lp = log_posterior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let posterior: Vec<f64> = log_posterior.iter().map(|&lp| (lp - max_lp).exp()).collect();
            let sum: f64 = posterior.iter().sum();
            let posterior: Vec<f64> = posterior.iter().map(|&p| p / sum).collect();

            // EAP estimate (posterior mean)
            let eap: f64 = posterior
                .iter()
                .zip(quad_vec.iter())
                .map(|(&p, &theta)| p * theta)
                .sum();

            // Posterior SD
            let psd: f64 = posterior
                .iter()
                .zip(quad_vec.iter())
                .map(|(&p, &theta)| p * (theta - eap).powi(2))
                .sum::<f64>()
                .sqrt();

            (eap, psd)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

// ============================================================================
// Python Module Definition
// ============================================================================

#[pymodule]
fn mirt_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // E-step functions
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_log_likelihoods_mirt, m)?)?;
    m.add_function(wrap_pyfunction!(e_step_complete, m)?)?;

    // M-step helpers
    m.add_function(wrap_pyfunction!(compute_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_polytomous, m)?)?;

    // SIBTEST
    m.add_function(wrap_pyfunction!(sibtest_compute_beta, m)?)?;
    m.add_function(wrap_pyfunction!(sibtest_all_items, m)?)?;

    // Simulation
    m.add_function(wrap_pyfunction!(simulate_grm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_dichotomous, m)?)?;

    // Plausible values
    m.add_function(wrap_pyfunction!(generate_plausible_values_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(generate_plausible_values_mcmc, m)?)?;

    // Model fit
    m.add_function(wrap_pyfunction!(compute_observed_margins, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_margins, m)?)?;

    // Bootstrap
    m.add_function(wrap_pyfunction!(generate_bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(resample_responses, m)?)?;

    // Imputation
    m.add_function(wrap_pyfunction!(impute_from_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_imputation, m)?)?;

    // Scoring
    m.add_function(wrap_pyfunction!(compute_eap_scores, m)?)?;

    Ok(())
}
