//! Multigroup IRT E-step computations.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

use crate::utils::{grm_category_probability, logsumexp};

fn compute_log_prior(quad_points: &[f64], prior_mean: f64, prior_var: f64) -> Vec<f64> {
    quad_points
        .iter()
        .map(|&theta| {
            let diff = theta - prior_mean;
            -0.5 * (2.0 * std::f64::consts::PI * prior_var).ln() - 0.5 * diff * diff / prior_var
        })
        .collect()
}

fn compute_log_quad_weights(quad_weights: &[f64]) -> Vec<f64> {
    quad_weights.iter().map(|&w| (w + 1e-300).ln()).collect()
}

fn compute_posterior_from_log_joint(log_joint: &[f64]) -> (Vec<f64>, f64) {
    let log_marginal = logsumexp(log_joint);
    let posterior: Vec<f64> = log_joint
        .iter()
        .map(|&lj| (lj - log_marginal).exp())
        .collect();
    (posterior, log_marginal)
}

fn compute_log_likelihoods_2pl_single(
    responses: &[i32],
    n_items: usize,
    quad_points: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    log_likes: &mut [f64],
) {
    for (q, &theta) in quad_points.iter().enumerate() {
        let mut ll = 0.0;
        for j in 0..n_items {
            let resp = responses[j];
            if resp >= 0 {
                let z = discrimination[j] * (theta - difficulty[j]);
                let p = 1.0 / (1.0 + (-z).exp());
                let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                if resp == 1 {
                    ll += p_clamped.ln();
                } else {
                    ll += (1.0 - p_clamped).ln();
                }
            }
        }
        log_likes[q] = ll;
    }
}

/// Compute multigroup E-step for 2PL models
///
/// Processes all groups in parallel using Rayon.
///
/// Parameters:
/// - responses_list: List of response matrices, one per group (n_persons_g, n_items)
/// - quad_points: Quadrature points (n_quad,)
/// - quad_weights: Quadrature weights (n_quad,)
/// - disc_list: List of discrimination arrays, one per group (n_items,)
/// - diff_list: List of difficulty arrays, one per group (n_items,)
/// - prior_means: Prior means per group (n_groups,)
/// - prior_vars: Prior variances per group (n_groups,)
///
/// Returns:
/// - posterior_weights: List of (n_persons_g, n_quad) arrays
/// - group_log_likelihoods: (n_groups,) array
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, diff_list, prior_means, prior_vars))]
pub fn multigroup_e_step_2pl<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    diff_list: Vec<PyReadonlyArray1<f64>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>) {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array().to_vec();
    let prior_vars = prior_vars.as_array().to_vec();
    let n_quad = quad_points.len();
    let _n_groups = responses_list.len();

    let log_quad_weights = compute_log_quad_weights(&quad_weights);

    let log_quad_weights_arc = Arc::new(log_quad_weights);
    let quad_points_arc = Arc::new(quad_points.clone());

    let group_n_persons: Vec<usize> = responses_list
        .iter()
        .map(|r| r.as_array().nrows())
        .collect();
    let n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(disc_list.iter())
        .zip(diff_list.iter())
        .enumerate()
        .map(|(g, ((resp, disc), diff))| {
            let resp_arr = resp.as_array();
            let disc_arr = disc.as_array();
            let diff_arr = diff.as_array();

            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();

            let resp_vec: Vec<Vec<i32>> =
                (0..n_persons).map(|i| resp_arr.row(i).to_vec()).collect();

            let disc_vec = Arc::new(disc_arr.to_vec());
            let diff_vec = Arc::new(diff_arr.to_vec());
            let log_prior = Arc::new(compute_log_prior(
                &quad_points,
                prior_means[g],
                prior_vars[g],
            ));

            (g, n_items, resp_vec, disc_vec, diff_vec, log_prior)
        })
        .collect();

    let all_persons: Vec<_> = group_data
        .into_iter()
        .flat_map(|(g, n_items, resp_vec, disc_vec, diff_vec, log_prior)| {
            resp_vec.into_iter().enumerate().map(move |(i, resp)| {
                (
                    g,
                    i,
                    resp,
                    n_items,
                    Arc::clone(&disc_vec),
                    Arc::clone(&diff_vec),
                    Arc::clone(&log_prior),
                )
            })
        })
        .collect();

    let log_qw = Arc::clone(&log_quad_weights_arc);
    let qp = Arc::clone(&quad_points_arc);

    let person_results: Vec<_> = all_persons
        .into_par_iter()
        .map(
            |(g, person_idx, person_resp, n_items, disc_vec, diff_vec, log_prior)| {
                let mut log_likes = vec![0.0; n_quad];
                compute_log_likelihoods_2pl_single(
                    &person_resp,
                    n_items,
                    &qp,
                    &disc_vec,
                    &diff_vec,
                    &mut log_likes,
                );

                let log_joint: Vec<f64> = (0..n_quad)
                    .map(|q| log_likes[q] + log_prior[q] + log_qw[q])
                    .collect();

                let (posterior, log_marginal) = compute_posterior_from_log_joint(&log_joint);
                (g, person_idx, posterior, log_marginal)
            },
        )
        .collect();

    let mut group_posteriors: Vec<Array2<f64>> = group_n_persons
        .iter()
        .map(|&n| Array2::zeros((n, n_quad)))
        .collect();
    let mut group_lls: Vec<f64> = vec![0.0; n_groups];

    for (g, person_idx, posterior, log_marginal) in person_results {
        for (q, &p) in posterior.iter().enumerate() {
            group_posteriors[g][[person_idx, q]] = p;
        }
        group_lls[g] += log_marginal;
    }

    let sorted_results: Vec<_> = (0..n_groups)
        .map(|g| (g, group_posteriors[g].clone(), group_lls[g]))
        .collect();

    let posterior_weights_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, pw, _)| pw.clone().to_pyarray(py))
        .collect();

    let group_lls: Vec<f64> = sorted_results.iter().map(|(_, _, ll)| *ll).collect();
    let group_lls_py = Array1::from_vec(group_lls).to_pyarray(py);

    (posterior_weights_py, group_lls_py)
}

/// Compute multigroup E-step for 3PL models
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, diff_list, guess_list, prior_means, prior_vars))]
pub fn multigroup_e_step_3pl<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    diff_list: Vec<PyReadonlyArray1<f64>>,
    guess_list: Vec<PyReadonlyArray1<f64>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>) {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array().to_vec();
    let prior_vars = prior_vars.as_array().to_vec();
    let n_quad = quad_points.len();
    let _n_groups = responses_list.len();

    let log_quad_weights = compute_log_quad_weights(&quad_weights);

    let log_quad_weights_arc = Arc::new(log_quad_weights);
    let quad_points_arc = Arc::new(quad_points.clone());

    let group_n_persons: Vec<usize> = responses_list
        .iter()
        .map(|r| r.as_array().nrows())
        .collect();
    let n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(disc_list.iter())
        .zip(diff_list.iter())
        .zip(guess_list.iter())
        .enumerate()
        .map(|(g, (((resp, disc), diff), guess))| {
            let resp_arr = resp.as_array();
            let disc_arr = disc.as_array();
            let diff_arr = diff.as_array();
            let guess_arr = guess.as_array();

            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();

            let resp_vec: Vec<Vec<i32>> =
                (0..n_persons).map(|i| resp_arr.row(i).to_vec()).collect();

            let disc_vec = Arc::new(disc_arr.to_vec());
            let diff_vec = Arc::new(diff_arr.to_vec());
            let guess_vec = Arc::new(guess_arr.to_vec());
            let log_prior = Arc::new(compute_log_prior(
                &quad_points,
                prior_means[g],
                prior_vars[g],
            ));

            (
                g, n_items, resp_vec, disc_vec, diff_vec, guess_vec, log_prior,
            )
        })
        .collect();

    let all_persons: Vec<_> = group_data
        .into_iter()
        .flat_map(
            |(g, n_items, resp_vec, disc_vec, diff_vec, guess_vec, log_prior)| {
                resp_vec.into_iter().enumerate().map(move |(i, resp)| {
                    (
                        g,
                        i,
                        resp,
                        n_items,
                        Arc::clone(&disc_vec),
                        Arc::clone(&diff_vec),
                        Arc::clone(&guess_vec),
                        Arc::clone(&log_prior),
                    )
                })
            },
        )
        .collect();

    let log_qw = Arc::clone(&log_quad_weights_arc);
    let qp = Arc::clone(&quad_points_arc);

    let person_results: Vec<_> = all_persons
        .into_par_iter()
        .map(
            |(g, person_idx, person_resp, n_items, disc_vec, diff_vec, guess_vec, log_prior)| {
                let mut log_likes = vec![0.0; n_quad];

                for (q, &theta) in qp.iter().enumerate() {
                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let resp = person_resp[j];
                        if resp >= 0 {
                            let z = disc_vec[j] * (theta - diff_vec[j]);
                            let p_star = 1.0 / (1.0 + (-z).exp());
                            let p = guess_vec[j] + (1.0 - guess_vec[j]) * p_star;
                            let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                            if resp == 1 {
                                ll += p_clamped.ln();
                            } else {
                                ll += (1.0 - p_clamped).ln();
                            }
                        }
                    }
                    log_likes[q] = ll;
                }

                let log_joint: Vec<f64> = (0..n_quad)
                    .map(|q| log_likes[q] + log_prior[q] + log_qw[q])
                    .collect();

                let (posterior, log_marginal) = compute_posterior_from_log_joint(&log_joint);
                (g, person_idx, posterior, log_marginal)
            },
        )
        .collect();

    let mut group_posteriors: Vec<Array2<f64>> = group_n_persons
        .iter()
        .map(|&n| Array2::zeros((n, n_quad)))
        .collect();
    let mut group_lls: Vec<f64> = vec![0.0; n_groups];

    for (g, person_idx, posterior, log_marginal) in person_results {
        for (q, &p) in posterior.iter().enumerate() {
            group_posteriors[g][[person_idx, q]] = p;
        }
        group_lls[g] += log_marginal;
    }

    let sorted_results: Vec<_> = (0..n_groups)
        .map(|g| (g, group_posteriors[g].clone(), group_lls[g]))
        .collect();

    let posterior_weights_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, pw, _)| pw.clone().to_pyarray(py))
        .collect();

    let group_lls: Vec<f64> = sorted_results.iter().map(|(_, _, ll)| *ll).collect();
    let group_lls_py = Array1::from_vec(group_lls).to_pyarray(py);

    (posterior_weights_py, group_lls_py)
}

/// Compute expected counts for all groups in parallel (for M-step)
///
/// Returns r_k (expected correct) and n_k (expected total) per group per item
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (responses_list, posterior_weights_list))]
pub fn multigroup_expected_counts<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    posterior_weights_list: Vec<PyReadonlyArray2<f64>>,
) -> (
    Vec<Bound<'py, PyArray2<f64>>>,
    Vec<Bound<'py, PyArray2<f64>>>,
) {
    let _n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(posterior_weights_list.iter())
        .enumerate()
        .map(|(g, (resp, weights))| {
            let resp_arr = resp.as_array();
            let weights_arr = weights.as_array();
            (g, resp_arr.to_owned(), weights_arr.to_owned())
        })
        .collect();

    let results: Vec<_> = group_data
        .into_par_iter()
        .map(|(g, resp_arr, weights_arr)| {
            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();
            let n_quad = weights_arr.ncols();

            let mut r_k_all = Array2::zeros((n_items, n_quad));
            let mut n_k_all = Array2::zeros((n_items, n_quad));

            for j in 0..n_items {
                for i in 0..n_persons {
                    let resp = resp_arr[[i, j]];
                    if resp >= 0 {
                        for q in 0..n_quad {
                            let w = weights_arr[[i, q]];
                            n_k_all[[j, q]] += w;
                            if resp == 1 {
                                r_k_all[[j, q]] += w;
                            }
                        }
                    }
                }
            }

            (g, r_k_all, n_k_all)
        })
        .collect();

    let mut sorted_results = results;
    sorted_results.sort_by_key(|(g, _, _)| *g);

    let r_k_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, r_k, _)| r_k.clone().to_pyarray(py))
        .collect();

    let n_k_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, _, n_k)| n_k.clone().to_pyarray(py))
        .collect();

    (r_k_py, n_k_py)
}

fn compute_grm_log_likelihood_single(
    responses: &[i32],
    n_items: usize,
    theta: f64,
    discrimination: &[f64],
    thresholds: &[Vec<f64>],
    n_categories: &[usize],
) -> f64 {
    let mut ll = 0.0;
    for j in 0..n_items {
        let resp = responses[j];
        if resp >= 0 {
            let prob = grm_category_probability(
                theta,
                discrimination[j],
                &thresholds[j],
                resp as usize,
                n_categories[j],
            );
            ll += prob.ln();
        }
    }
    ll
}

/// Compute multigroup E-step for GRM models
///
/// Processes all groups in parallel using Rayon.
///
/// Parameters:
/// - responses_list: List of response matrices, one per group (n_persons_g, n_items)
/// - quad_points: Quadrature points (n_quad,)
/// - quad_weights: Quadrature weights (n_quad,)
/// - disc_list: List of discrimination arrays, one per group (n_items,)
/// - thresh_list: List of threshold matrices, one per group (n_items, max_categories-1)
/// - n_categories_list: List of n_categories arrays, one per group (n_items,)
/// - prior_means: Prior means per group (n_groups,)
/// - prior_vars: Prior variances per group (n_groups,)
///
/// Returns:
/// - posterior_weights: List of (n_persons_g, n_quad) arrays
/// - group_log_likelihoods: (n_groups,) array
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, thresh_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_grm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    thresh_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>) {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array().to_vec();
    let prior_vars = prior_vars.as_array().to_vec();
    let n_quad = quad_points.len();
    let _n_groups = responses_list.len();

    let log_quad_weights = compute_log_quad_weights(&quad_weights);

    let log_quad_weights_arc = Arc::new(log_quad_weights);
    let quad_points_arc = Arc::new(quad_points.clone());

    let group_n_persons: Vec<usize> = responses_list
        .iter()
        .map(|r| r.as_array().nrows())
        .collect();
    let n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(disc_list.iter())
        .zip(thresh_list.iter())
        .zip(n_categories_list.iter())
        .enumerate()
        .map(|(g, (((resp, disc), thresh), n_cats))| {
            let resp_arr = resp.as_array();
            let disc_arr = disc.as_array();
            let thresh_arr = thresh.as_array();
            let n_cats_arr = n_cats.as_array();

            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();

            let resp_vec: Vec<Vec<i32>> =
                (0..n_persons).map(|i| resp_arr.row(i).to_vec()).collect();

            let disc_vec = Arc::new(disc_arr.to_vec());
            let n_cats_vec: Vec<usize> = n_cats_arr.iter().map(|&x| x as usize).collect();

            let thresh_vecs: Vec<Vec<f64>> = (0..n_items)
                .map(|j| {
                    let n_thresh = n_cats_vec[j] - 1;
                    (0..n_thresh).map(|k| thresh_arr[[j, k]]).collect()
                })
                .collect();

            let thresh_vecs_arc = Arc::new(thresh_vecs);
            let n_cats_vec_arc = Arc::new(n_cats_vec);
            let log_prior = Arc::new(compute_log_prior(
                &quad_points,
                prior_means[g],
                prior_vars[g],
            ));

            (
                g,
                n_items,
                resp_vec,
                disc_vec,
                thresh_vecs_arc,
                n_cats_vec_arc,
                log_prior,
            )
        })
        .collect();

    let all_persons: Vec<_> = group_data
        .into_iter()
        .flat_map(
            |(g, n_items, resp_vec, disc_vec, thresh_vecs, n_cats_vec, log_prior)| {
                resp_vec.into_iter().enumerate().map(move |(i, resp)| {
                    (
                        g,
                        i,
                        resp,
                        n_items,
                        Arc::clone(&disc_vec),
                        Arc::clone(&thresh_vecs),
                        Arc::clone(&n_cats_vec),
                        Arc::clone(&log_prior),
                    )
                })
            },
        )
        .collect();

    let log_qw = Arc::clone(&log_quad_weights_arc);
    let qp = Arc::clone(&quad_points_arc);

    let person_results: Vec<_> = all_persons
        .into_par_iter()
        .map(
            |(
                g,
                person_idx,
                person_resp,
                n_items,
                disc_vec,
                thresh_vecs,
                n_cats_vec,
                log_prior,
            )| {
                let log_likes: Vec<f64> = qp
                    .iter()
                    .map(|&theta| {
                        compute_grm_log_likelihood_single(
                            &person_resp,
                            n_items,
                            theta,
                            &disc_vec,
                            &thresh_vecs,
                            &n_cats_vec,
                        )
                    })
                    .collect();

                let log_joint: Vec<f64> = (0..n_quad)
                    .map(|q| log_likes[q] + log_prior[q] + log_qw[q])
                    .collect();

                let (posterior, log_marginal) = compute_posterior_from_log_joint(&log_joint);
                (g, person_idx, posterior, log_marginal)
            },
        )
        .collect();

    let mut group_posteriors: Vec<Array2<f64>> = group_n_persons
        .iter()
        .map(|&n| Array2::zeros((n, n_quad)))
        .collect();
    let mut group_lls: Vec<f64> = vec![0.0; n_groups];

    for (g, person_idx, posterior, log_marginal) in person_results {
        for (q, &p) in posterior.iter().enumerate() {
            group_posteriors[g][[person_idx, q]] = p;
        }
        group_lls[g] += log_marginal;
    }

    let sorted_results: Vec<_> = (0..n_groups)
        .map(|g| (g, group_posteriors[g].clone(), group_lls[g]))
        .collect();

    let posterior_weights_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, pw, _)| pw.clone().to_pyarray(py))
        .collect();

    let group_lls: Vec<f64> = sorted_results.iter().map(|(_, _, ll)| *ll).collect();
    let group_lls_py = Array1::from_vec(group_lls).to_pyarray(py);

    (posterior_weights_py, group_lls_py)
}

fn compute_gpcm_log_likelihood_single(
    responses: &[i32],
    n_items: usize,
    theta: f64,
    discrimination: &[f64],
    steps: &[Vec<f64>],
    n_categories: &[usize],
) -> f64 {
    let mut ll = 0.0;
    for j in 0..n_items {
        let resp = responses[j];
        if resp < 0 {
            continue;
        }

        let a = discrimination[j];
        let n_cat = n_categories[j];

        let mut numerators = vec![0.0; n_cat];
        for k in 1..n_cat {
            numerators[k] = numerators[k - 1] + a * (theta - steps[j][k]);
        }

        let max_num = numerators.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = numerators.iter().map(|&x| (x - max_num).exp()).sum();
        let log_denom = max_num + sum_exp.ln();

        let prob = (numerators[resp as usize] - log_denom).exp().max(1e-10);
        ll += prob.ln();
    }
    ll
}

fn compute_nrm_log_likelihood_single(
    responses: &[i32],
    n_items: usize,
    theta: f64,
    slopes: &[Vec<f64>],
    intercepts: &[Vec<f64>],
    n_categories: &[usize],
) -> f64 {
    let mut ll = 0.0;
    for j in 0..n_items {
        let resp = responses[j];
        if resp < 0 {
            continue;
        }

        let n_cat = n_categories[j];

        let mut logits = vec![0.0; n_cat];
        for k in 0..n_cat {
            logits[k] = slopes[j][k] * theta + intercepts[j][k];
        }

        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let log_denom = max_logit + sum_exp.ln();

        let prob = (logits[resp as usize] - log_denom).exp().max(1e-10);
        ll += prob.ln();
    }
    ll
}

/// Compute multigroup E-step for GPCM models
///
/// Processes all groups in parallel using Rayon.
///
/// Parameters:
/// - responses_list: List of response matrices, one per group (n_persons_g, n_items)
/// - quad_points: Quadrature points (n_quad,)
/// - quad_weights: Quadrature weights (n_quad,)
/// - disc_list: List of discrimination arrays, one per group (n_items,)
/// - steps_list: List of step matrices, one per group (n_items, max_categories)
/// - n_categories_list: List of n_categories arrays, one per group (n_items,)
/// - prior_means: Prior means per group (n_groups,)
/// - prior_vars: Prior variances per group (n_groups,)
///
/// Returns:
/// - posterior_weights: List of (n_persons_g, n_quad) arrays
/// - group_log_likelihoods: (n_groups,) array
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, disc_list, steps_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_gpcm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    disc_list: Vec<PyReadonlyArray1<f64>>,
    steps_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>) {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array().to_vec();
    let prior_vars = prior_vars.as_array().to_vec();
    let n_quad = quad_points.len();
    let _n_groups = responses_list.len();

    let log_quad_weights = compute_log_quad_weights(&quad_weights);

    let log_quad_weights_arc = Arc::new(log_quad_weights);
    let quad_points_arc = Arc::new(quad_points.clone());

    let group_n_persons: Vec<usize> = responses_list
        .iter()
        .map(|r| r.as_array().nrows())
        .collect();
    let n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(disc_list.iter())
        .zip(steps_list.iter())
        .zip(n_categories_list.iter())
        .enumerate()
        .map(|(g, (((resp, disc), steps), n_cats))| {
            let resp_arr = resp.as_array();
            let disc_arr = disc.as_array();
            let steps_arr = steps.as_array();
            let n_cats_arr = n_cats.as_array();

            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();

            let resp_vec: Vec<Vec<i32>> =
                (0..n_persons).map(|i| resp_arr.row(i).to_vec()).collect();

            let disc_vec = Arc::new(disc_arr.to_vec());
            let n_cats_vec: Vec<usize> = n_cats_arr.iter().map(|&x| x as usize).collect();

            let steps_vecs: Vec<Vec<f64>> = (0..n_items)
                .map(|j| {
                    let n_cat = n_cats_vec[j];
                    (0..n_cat).map(|k| steps_arr[[j, k]]).collect()
                })
                .collect();

            let steps_vecs_arc = Arc::new(steps_vecs);
            let n_cats_vec_arc = Arc::new(n_cats_vec);
            let log_prior = Arc::new(compute_log_prior(
                &quad_points,
                prior_means[g],
                prior_vars[g],
            ));

            (
                g,
                n_items,
                resp_vec,
                disc_vec,
                steps_vecs_arc,
                n_cats_vec_arc,
                log_prior,
            )
        })
        .collect();

    let all_persons: Vec<_> = group_data
        .into_iter()
        .flat_map(
            |(g, n_items, resp_vec, disc_vec, steps_vecs, n_cats_vec, log_prior)| {
                resp_vec.into_iter().enumerate().map(move |(i, resp)| {
                    (
                        g,
                        i,
                        resp,
                        n_items,
                        Arc::clone(&disc_vec),
                        Arc::clone(&steps_vecs),
                        Arc::clone(&n_cats_vec),
                        Arc::clone(&log_prior),
                    )
                })
            },
        )
        .collect();

    let log_qw = Arc::clone(&log_quad_weights_arc);
    let qp = Arc::clone(&quad_points_arc);

    let person_results: Vec<_> = all_persons
        .into_par_iter()
        .map(
            |(g, person_idx, person_resp, n_items, disc_vec, steps_vecs, n_cats_vec, log_prior)| {
                let log_likes: Vec<f64> = qp
                    .iter()
                    .map(|&theta| {
                        compute_gpcm_log_likelihood_single(
                            &person_resp,
                            n_items,
                            theta,
                            &disc_vec,
                            &steps_vecs,
                            &n_cats_vec,
                        )
                    })
                    .collect();

                let log_joint: Vec<f64> = (0..n_quad)
                    .map(|q| log_likes[q] + log_prior[q] + log_qw[q])
                    .collect();

                let (posterior, log_marginal) = compute_posterior_from_log_joint(&log_joint);
                (g, person_idx, posterior, log_marginal)
            },
        )
        .collect();

    let mut group_posteriors: Vec<Array2<f64>> = group_n_persons
        .iter()
        .map(|&n| Array2::zeros((n, n_quad)))
        .collect();
    let mut group_lls: Vec<f64> = vec![0.0; n_groups];

    for (g, person_idx, posterior, log_marginal) in person_results {
        for (q, &p) in posterior.iter().enumerate() {
            group_posteriors[g][[person_idx, q]] = p;
        }
        group_lls[g] += log_marginal;
    }

    let sorted_results: Vec<_> = (0..n_groups)
        .map(|g| (g, group_posteriors[g].clone(), group_lls[g]))
        .collect();

    let posterior_weights_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, pw, _)| pw.clone().to_pyarray(py))
        .collect();

    let group_lls: Vec<f64> = sorted_results.iter().map(|(_, _, ll)| *ll).collect();
    let group_lls_py = Array1::from_vec(group_lls).to_pyarray(py);

    (posterior_weights_py, group_lls_py)
}

/// Compute multigroup E-step for NRM models
///
/// Processes all groups in parallel using Rayon.
///
/// Parameters:
/// - responses_list: List of response matrices, one per group (n_persons_g, n_items)
/// - quad_points: Quadrature points (n_quad,)
/// - quad_weights: Quadrature weights (n_quad,)
/// - slopes_list: List of slope matrices, one per group (n_items, max_categories)
/// - intercepts_list: List of intercept matrices, one per group (n_items, max_categories)
/// - n_categories_list: List of n_categories arrays, one per group (n_items,)
/// - prior_means: Prior means per group (n_groups,)
/// - prior_vars: Prior variances per group (n_groups,)
///
/// Returns:
/// - posterior_weights: List of (n_persons_g, n_quad) arrays
/// - group_log_likelihoods: (n_groups,) array
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (responses_list, quad_points, quad_weights, slopes_list, intercepts_list, n_categories_list, prior_means, prior_vars))]
pub fn multigroup_e_step_nrm<'py>(
    py: Python<'py>,
    responses_list: Vec<PyReadonlyArray2<i32>>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    slopes_list: Vec<PyReadonlyArray2<f64>>,
    intercepts_list: Vec<PyReadonlyArray2<f64>>,
    n_categories_list: Vec<PyReadonlyArray1<i32>>,
    prior_means: PyReadonlyArray1<f64>,
    prior_vars: PyReadonlyArray1<f64>,
) -> (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray1<f64>>) {
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let prior_means = prior_means.as_array().to_vec();
    let prior_vars = prior_vars.as_array().to_vec();
    let n_quad = quad_points.len();
    let _n_groups = responses_list.len();

    let log_quad_weights = compute_log_quad_weights(&quad_weights);

    let log_quad_weights_arc = Arc::new(log_quad_weights);
    let quad_points_arc = Arc::new(quad_points.clone());

    let group_n_persons: Vec<usize> = responses_list
        .iter()
        .map(|r| r.as_array().nrows())
        .collect();
    let n_groups = responses_list.len();

    let group_data: Vec<_> = responses_list
        .iter()
        .zip(slopes_list.iter())
        .zip(intercepts_list.iter())
        .zip(n_categories_list.iter())
        .enumerate()
        .map(|(g, (((resp, slopes), intercepts), n_cats))| {
            let resp_arr = resp.as_array();
            let slopes_arr = slopes.as_array();
            let intercepts_arr = intercepts.as_array();
            let n_cats_arr = n_cats.as_array();

            let n_persons = resp_arr.nrows();
            let n_items = resp_arr.ncols();

            let resp_vec: Vec<Vec<i32>> =
                (0..n_persons).map(|i| resp_arr.row(i).to_vec()).collect();

            let n_cats_vec: Vec<usize> = n_cats_arr.iter().map(|&x| x as usize).collect();

            let slopes_vecs: Vec<Vec<f64>> = (0..n_items)
                .map(|j| {
                    let n_cat = n_cats_vec[j];
                    (0..n_cat).map(|k| slopes_arr[[j, k]]).collect()
                })
                .collect();

            let intercepts_vecs: Vec<Vec<f64>> = (0..n_items)
                .map(|j| {
                    let n_cat = n_cats_vec[j];
                    (0..n_cat).map(|k| intercepts_arr[[j, k]]).collect()
                })
                .collect();

            let slopes_vecs_arc = Arc::new(slopes_vecs);
            let intercepts_vecs_arc = Arc::new(intercepts_vecs);
            let n_cats_vec_arc = Arc::new(n_cats_vec);
            let log_prior = Arc::new(compute_log_prior(
                &quad_points,
                prior_means[g],
                prior_vars[g],
            ));

            (
                g,
                n_items,
                resp_vec,
                slopes_vecs_arc,
                intercepts_vecs_arc,
                n_cats_vec_arc,
                log_prior,
            )
        })
        .collect();

    let all_persons: Vec<_> = group_data
        .into_iter()
        .flat_map(
            |(g, n_items, resp_vec, slopes_vecs, intercepts_vecs, n_cats_vec, log_prior)| {
                resp_vec.into_iter().enumerate().map(move |(i, resp)| {
                    (
                        g,
                        i,
                        resp,
                        n_items,
                        Arc::clone(&slopes_vecs),
                        Arc::clone(&intercepts_vecs),
                        Arc::clone(&n_cats_vec),
                        Arc::clone(&log_prior),
                    )
                })
            },
        )
        .collect();

    let log_qw = Arc::clone(&log_quad_weights_arc);
    let qp = Arc::clone(&quad_points_arc);

    let person_results: Vec<_> = all_persons
        .into_par_iter()
        .map(
            |(
                g,
                person_idx,
                person_resp,
                n_items,
                slopes_vecs,
                intercepts_vecs,
                n_cats_vec,
                log_prior,
            )| {
                let log_likes: Vec<f64> = qp
                    .iter()
                    .map(|&theta| {
                        compute_nrm_log_likelihood_single(
                            &person_resp,
                            n_items,
                            theta,
                            &slopes_vecs,
                            &intercepts_vecs,
                            &n_cats_vec,
                        )
                    })
                    .collect();

                let log_joint: Vec<f64> = (0..n_quad)
                    .map(|q| log_likes[q] + log_prior[q] + log_qw[q])
                    .collect();

                let (posterior, log_marginal) = compute_posterior_from_log_joint(&log_joint);
                (g, person_idx, posterior, log_marginal)
            },
        )
        .collect();

    let mut group_posteriors: Vec<Array2<f64>> = group_n_persons
        .iter()
        .map(|&n| Array2::zeros((n, n_quad)))
        .collect();
    let mut group_lls: Vec<f64> = vec![0.0; n_groups];

    for (g, person_idx, posterior, log_marginal) in person_results {
        for (q, &p) in posterior.iter().enumerate() {
            group_posteriors[g][[person_idx, q]] = p;
        }
        group_lls[g] += log_marginal;
    }

    let sorted_results: Vec<_> = (0..n_groups)
        .map(|g| (g, group_posteriors[g].clone(), group_lls[g]))
        .collect();

    let posterior_weights_py: Vec<_> = sorted_results
        .iter()
        .map(|(_, pw, _)| pw.clone().to_pyarray(py))
        .collect();

    let group_lls: Vec<f64> = sorted_results.iter().map(|(_, _, ll)| *ll).collect();
    let group_lls_py = Array1::from_vec(group_lls).to_pyarray(py);

    (posterior_weights_py, group_lls_py)
}

/// Register multigroup functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multigroup_e_step_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_grm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_e_step_nrm, m)?)?;
    m.add_function(wrap_pyfunction!(multigroup_expected_counts, m)?)?;
    Ok(())
}
