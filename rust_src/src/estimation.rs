//! Parameter estimation functions (EM, Gibbs, MHRM, Bootstrap).

use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::utils::{
    EPSILON, compute_log_weights, gauss_hermite_quadrature, log_sigmoid, logsumexp, sigmoid,
};

/// EM algorithm for 2PL model fitting
#[pyfunction]
pub fn em_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    bool,
) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];

    for j in 0..n_items {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n_persons {
            let r = responses[[i, j]];
            if r >= 0 {
                sum += r as f64;
                count += 1;
            }
        }
        if count > 0 {
            let p = (sum / count as f64).clamp(0.01, 0.99);
            difficulty[j] = -p.ln() / (1.0 - p).ln().abs().max(0.01);
        }
    }

    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iteration = 0;

    for iter in 0..max_iter {
        iteration = iter + 1;

        let (posterior_weights, marginal_ll) = e_step_2pl_internal(
            &responses,
            &quad_points,
            &quad_weights,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            n_quadpts,
        );

        let current_ll: f64 = marginal_ll.iter().map(|&x| (x + EPSILON).ln()).sum();

        if (current_ll - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = current_ll;

        m_step_2pl_internal(
            &responses,
            &posterior_weights,
            &quad_points,
            &mut discrimination,
            &mut difficulty,
            n_persons,
            n_items,
            n_quadpts,
        );
    }

    let disc_arr: Array1<f64> = discrimination.into();
    let diff_arr: Array1<f64> = difficulty.into();

    (
        disc_arr.to_pyarray(py),
        diff_arr.to_pyarray(py),
        prev_ll,
        iteration,
        converged,
    )
}

#[allow(clippy::too_many_arguments)]
fn e_step_2pl_internal(
    responses: &ndarray::ArrayView2<i32>,
    quad_points: &[f64],
    quad_weights: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
    n_quad: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let log_weights = compute_log_weights(quad_weights);

    let results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut log_joint = vec![0.0; n_quad];

            for q in 0..n_quad {
                let theta = quad_points[q];
                let mut ll = 0.0;

                for j in 0..n_items {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    let z = discrimination[j] * (theta - difficulty[j]);
                    if resp == 1 {
                        ll += log_sigmoid(z);
                    } else {
                        ll += log_sigmoid(-z);
                    }
                }

                log_joint[q] = ll + log_weights[q];
            }

            let log_marginal = logsumexp(&log_joint);
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal.exp())
        })
        .collect();

    let posterior_weights: Vec<Vec<f64>> = results.iter().map(|(p, _)| p.clone()).collect();
    let marginal_ll: Vec<f64> = results.iter().map(|(_, m)| *m).collect();

    (posterior_weights, marginal_ll)
}

#[allow(clippy::too_many_arguments)]
fn m_step_2pl_internal(
    responses: &ndarray::ArrayView2<i32>,
    posterior_weights: &[Vec<f64>],
    quad_points: &[f64],
    discrimination: &mut [f64],
    difficulty: &mut [f64],
    n_persons: usize,
    n_items: usize,
    n_quad: usize,
) {
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
                    let w = posterior_weights[i][q];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            let mut a = discrimination[j];
            let mut b = difficulty[j];

            for _ in 0..10 {
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

                hess_aa -= 0.01;
                hess_bb -= 0.01;

                let det = hess_aa * hess_bb - hess_ab * hess_ab;
                if det.abs() < EPSILON {
                    break;
                }

                let delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det;
                let delta_b = (-hess_ab * grad_a + hess_aa * grad_b) / det;

                a = (a - delta_a * 0.5).clamp(0.1, 5.0);
                b = (b - delta_b * 0.5).clamp(-6.0, 6.0);

                if delta_a.abs() < 1e-4 && delta_b.abs() < 1e-4 {
                    break;
                }
            }

            (a, b)
        })
        .collect();

    for (j, (a, b)) in new_params.into_iter().enumerate() {
        discrimination[j] = a;
        difficulty[j] = b;
    }
}

/// Gibbs sampling for 2PL model
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn gibbs_sample_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    seed: u64,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];
    let mut theta: Vec<f64> = vec![0.0; n_persons];

    let n_samples = (n_iter - burnin) / thin;
    let mut disc_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut diff_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut theta_chain: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut ll_chain: Vec<f64> = Vec::with_capacity(n_samples);

    let mut rng = Pcg64::seed_from_u64(seed);
    let proposal_theta = Normal::new(0.0, 0.5).unwrap();
    let proposal_param = Normal::new(0.0, 0.1).unwrap();

    for iter in 0..n_iter {
        theta = sample_theta_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            &mut rng,
            &proposal_theta,
        );

        discrimination = sample_discrimination_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_items,
            &mut rng,
            &proposal_param,
        );

        difficulty = sample_difficulty_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_items,
            &mut rng,
            &proposal_param,
        );

        if iter >= burnin && (iter - burnin).is_multiple_of(thin) {
            disc_chain.push(discrimination.clone());
            diff_chain.push(difficulty.clone());
            theta_chain.push(theta.clone());

            let ll = compute_total_ll(
                &responses,
                &theta,
                &discrimination,
                &difficulty,
                n_persons,
                n_items,
            );
            ll_chain.push(ll);
        }
    }

    let disc_arr = Array2::from_shape_vec(
        (n_samples, n_items),
        disc_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let diff_arr = Array2::from_shape_vec(
        (n_samples, n_items),
        diff_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let theta_arr = Array3::from_shape_vec(
        (n_samples, n_persons, 1),
        theta_chain.into_iter().flatten().collect(),
    )
    .unwrap();

    let ll_arr: Array1<f64> = ll_chain.into();

    (
        disc_arr.to_pyarray(py),
        diff_arr.to_pyarray(py),
        theta_arr.to_pyarray(py),
        ll_arr.to_pyarray(py),
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_theta_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let seeds: Vec<u64> = (0..n_persons).map(|_| rng.random()).collect();

    let new_theta: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut local_rng = Pcg64::seed_from_u64(seeds[i]);

            let current = theta[i];
            let proposed = current + local_rng.sample(proposal);

            let mut ll_current = 0.0;
            let mut ll_proposed = 0.0;

            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp < 0 {
                    continue;
                }
                let z_curr = discrimination[j] * (current - difficulty[j]);
                let z_prop = discrimination[j] * (proposed - difficulty[j]);

                if resp == 1 {
                    ll_current += log_sigmoid(z_curr);
                    ll_proposed += log_sigmoid(z_prop);
                } else {
                    ll_current += log_sigmoid(-z_curr);
                    ll_proposed += log_sigmoid(-z_prop);
                }
            }

            let prior_current = -0.5 * current * current;
            let prior_proposed = -0.5 * proposed * proposed;

            let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

            if local_rng.random::<f64>().ln() < log_alpha {
                proposed
            } else {
                current
            }
        })
        .collect();

    new_theta
}

fn sample_discrimination_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_disc = discrimination.to_vec();

    for j in 0..n_items {
        let current = discrimination[j];
        let proposed = (current + rng.sample(proposal)).clamp(0.1, 5.0);

        let mut ll_current = 0.0;
        let mut ll_proposed = 0.0;

        for i in 0..n_persons {
            let resp = responses[[i, j]];
            if resp < 0 {
                continue;
            }
            let z_curr = current * (theta[i] - difficulty[j]);
            let z_prop = proposed * (theta[i] - difficulty[j]);

            if resp == 1 {
                ll_current += log_sigmoid(z_curr);
                ll_proposed += log_sigmoid(z_prop);
            } else {
                ll_current += log_sigmoid(-z_curr);
                ll_proposed += log_sigmoid(-z_prop);
            }
        }

        let prior_current = -0.5 * current.ln().powi(2);
        let prior_proposed = -0.5 * proposed.ln().powi(2);

        let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

        if rng.random::<f64>().ln() < log_alpha {
            new_disc[j] = proposed;
        }
    }

    new_disc
}

fn sample_difficulty_mh(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut Pcg64,
    proposal: &Normal<f64>,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_diff = difficulty.to_vec();

    for j in 0..n_items {
        let current = difficulty[j];
        let proposed = (current + rng.sample(proposal)).clamp(-6.0, 6.0);

        let mut ll_current = 0.0;
        let mut ll_proposed = 0.0;

        for i in 0..n_persons {
            let resp = responses[[i, j]];
            if resp < 0 {
                continue;
            }
            let z_curr = discrimination[j] * (theta[i] - current);
            let z_prop = discrimination[j] * (theta[i] - proposed);

            if resp == 1 {
                ll_current += log_sigmoid(z_curr);
                ll_proposed += log_sigmoid(z_prop);
            } else {
                ll_current += log_sigmoid(-z_curr);
                ll_proposed += log_sigmoid(-z_prop);
            }
        }

        let prior_current = -0.5 * current * current;
        let prior_proposed = -0.5 * proposed * proposed;

        let log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current);

        if rng.random::<f64>().ln() < log_alpha {
            new_diff[j] = proposed;
        }
    }

    new_diff
}

fn compute_total_ll(
    responses: &ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
) -> f64 {
    (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut ll = 0.0;
            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp < 0 {
                    continue;
                }
                let z = discrimination[j] * (theta[i] - difficulty[j]);
                if resp == 1 {
                    ll += log_sigmoid(z);
                } else {
                    ll += log_sigmoid(-z);
                }
            }
            ll
        })
        .sum()
}

/// Metropolis-Hastings Robbins-Monro for 2PL
#[pyfunction]
pub fn mhrm_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_cycles: usize,
    burnin: usize,
    proposal_sd: f64,
    seed: u64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let mut discrimination: Vec<f64> = vec![1.0; n_items];
    let mut difficulty: Vec<f64> = vec![0.0; n_items];
    let mut theta: Vec<f64> = vec![0.0; n_persons];

    let mut rng = Pcg64::seed_from_u64(seed);
    let proposal = Normal::new(0.0, proposal_sd).unwrap();

    for cycle in 0..n_cycles {
        theta = sample_theta_mh(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
            &mut rng,
            &proposal,
        );

        let gain = 1.0 / (cycle as f64 + 1.0);

        if cycle >= burnin {
            for j in 0..n_items {
                let mut grad_a = 0.0;
                let mut grad_b = 0.0;
                let mut count = 0;

                for i in 0..n_persons {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    count += 1;
                    let z = discrimination[j] * (theta[i] - difficulty[j]);
                    let p = sigmoid(z);
                    let residual = resp as f64 - p;

                    grad_a += residual * (theta[i] - difficulty[j]);
                    grad_b += -residual * discrimination[j];
                }

                if count > 0 {
                    grad_a /= count as f64;
                    grad_b /= count as f64;

                    discrimination[j] = (discrimination[j] + gain * grad_a).clamp(0.1, 5.0);
                    difficulty[j] = (difficulty[j] + gain * grad_b).clamp(-6.0, 6.0);
                }
            }
        }
    }

    let ll = compute_total_ll(
        &responses,
        &theta,
        &discrimination,
        &difficulty,
        n_persons,
        n_items,
    );

    let disc_arr: Array1<f64> = discrimination.into();
    let diff_arr: Array1<f64> = difficulty.into();

    (disc_arr.to_pyarray(py), diff_arr.to_pyarray(py), ll)
}

/// Bootstrap parameter estimation for 2PL
#[pyfunction]
pub fn bootstrap_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_bootstrap: usize,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let results: Vec<(Vec<f64>, Vec<f64>)> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(seed + b as u64);

            let indices: Vec<usize> = (0..n_persons)
                .map(|_| rng.random_range(0..n_persons))
                .collect();

            let mut boot_responses = Array2::zeros((n_persons, n_items));
            for (new_i, &orig_i) in indices.iter().enumerate() {
                for j in 0..n_items {
                    boot_responses[[new_i, j]] = responses[[orig_i, j]];
                }
            }

            let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);
            let mut discrimination: Vec<f64> = vec![1.0; n_items];
            let mut difficulty: Vec<f64> = vec![0.0; n_items];

            for j in 0..n_items {
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..n_persons {
                    let r = boot_responses[[i, j]];
                    if r >= 0 {
                        sum += r as f64;
                        count += 1;
                    }
                }
                if count > 0 {
                    let p = (sum / count as f64).clamp(0.01, 0.99);
                    difficulty[j] = -p.ln() / (1.0 - p).ln().abs().max(0.01);
                }
            }

            let mut prev_ll = f64::NEG_INFINITY;

            for _ in 0..max_iter {
                let (posterior_weights, marginal_ll) = e_step_2pl_internal(
                    &boot_responses.view(),
                    &quad_points,
                    &quad_weights,
                    &discrimination,
                    &difficulty,
                    n_persons,
                    n_items,
                    n_quadpts,
                );

                let current_ll: f64 = marginal_ll.iter().map(|&x| (x + EPSILON).ln()).sum();

                if (current_ll - prev_ll).abs() < tol {
                    break;
                }
                prev_ll = current_ll;

                m_step_2pl_internal(
                    &boot_responses.view(),
                    &posterior_weights,
                    &quad_points,
                    &mut discrimination,
                    &mut difficulty,
                    n_persons,
                    n_items,
                    n_quadpts,
                );
            }

            (discrimination, difficulty)
        })
        .collect();

    let mut disc_samples = Array2::zeros((n_bootstrap, n_items));
    let mut diff_samples = Array2::zeros((n_bootstrap, n_items));

    for (b, (disc, diff)) in results.into_iter().enumerate() {
        for j in 0..n_items {
            disc_samples[[b, j]] = disc[j];
            diff_samples[[b, j]] = diff[j];
        }
    }

    (disc_samples.to_pyarray(py), diff_samples.to_pyarray(py))
}

/// Single EM iteration for 2PL model (combined E+M step to reduce FFI overhead)
///
/// Returns new parameters, posterior weights, and log-likelihood in a single call.
#[pyfunction]
#[pyo3(signature = (responses, quad_points, quad_weights, discrimination, difficulty, prior_mean, prior_var, max_m_iter, m_tol, disc_bounds, diff_bounds, damping, regularization))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn em_iteration_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: numpy::PyReadonlyArray1<f64>,
    quad_weights: numpy::PyReadonlyArray1<f64>,
    discrimination: numpy::PyReadonlyArray1<f64>,
    difficulty: numpy::PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
    max_m_iter: usize,
    m_tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    damping: f64,
    regularization: f64,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let disc_init = discrimination.as_array().to_vec();
    let diff_init = difficulty.as_array().to_vec();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let log_weights = compute_log_weights(&quad_weights);

    let log_prior: Vec<f64> = quad_points
        .iter()
        .map(|&theta| {
            let z = (theta - prior_mean) / prior_var.sqrt();
            -0.5 * (std::f64::consts::TAU * prior_var).ln() - 0.5 * z * z
        })
        .collect();

    let e_step_results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut log_joint = vec![0.0; n_quad];

            for q in 0..n_quad {
                let theta = quad_points[q];
                let mut ll = 0.0;

                for j in 0..n_items {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    let z = disc_init[j] * (theta - diff_init[j]);
                    if resp == 1 {
                        ll += log_sigmoid(z);
                    } else {
                        ll += log_sigmoid(-z);
                    }
                }

                log_joint[q] = ll + log_prior[q] + log_weights[q];
            }

            let log_marginal = logsumexp(&log_joint);
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal)
        })
        .collect();

    let posterior_weights: Vec<Vec<f64>> = e_step_results.iter().map(|(p, _)| p.clone()).collect();
    let log_likelihood: f64 = e_step_results.iter().map(|(_, lm)| lm).sum();

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
                    let w = posterior_weights[i][q];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            let mut a = disc_init[j];
            let mut b = diff_init[j];

            for _ in 0..max_m_iter {
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

                if delta_a.abs() < m_tol && delta_b.abs() < m_tol {
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

    let mut posterior_arr = ndarray::Array2::zeros((n_persons, n_quad));
    for (i, pw) in posterior_weights.iter().enumerate() {
        for (q, &w) in pw.iter().enumerate() {
            posterior_arr[[i, q]] = w;
        }
    }

    (
        disc_new.to_pyarray(py),
        diff_new.to_pyarray(py),
        posterior_arr.to_pyarray(py),
        log_likelihood,
    )
}

/// Single EM iteration for 3PL model (combined E+M step)
#[pyfunction]
#[pyo3(signature = (responses, quad_points, quad_weights, discrimination, difficulty, guessing, prior_mean, prior_var, max_m_iter, m_tol, disc_bounds, diff_bounds, guess_bounds, damping_ab, damping_c, regularization, regularization_c))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn em_iteration_3pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: numpy::PyReadonlyArray1<f64>,
    quad_weights: numpy::PyReadonlyArray1<f64>,
    discrimination: numpy::PyReadonlyArray1<f64>,
    difficulty: numpy::PyReadonlyArray1<f64>,
    guessing: numpy::PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
    max_m_iter: usize,
    m_tol: f64,
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
    Bound<'py, PyArray2<f64>>,
    f64,
) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let disc_init = discrimination.as_array().to_vec();
    let diff_init = difficulty.as_array().to_vec();
    let guess_init = guessing.as_array().to_vec();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let log_weights = compute_log_weights(&quad_weights);

    let log_prior: Vec<f64> = quad_points
        .iter()
        .map(|&theta| {
            let z = (theta - prior_mean) / prior_var.sqrt();
            -0.5 * (std::f64::consts::TAU * prior_var).ln() - 0.5 * z * z
        })
        .collect();

    let e_step_results: Vec<(Vec<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut log_joint = vec![0.0; n_quad];

            for q in 0..n_quad {
                let theta = quad_points[q];
                let mut ll = 0.0;

                for j in 0..n_items {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        continue;
                    }
                    let z = disc_init[j] * (theta - diff_init[j]);
                    let p_star = sigmoid(z);
                    let p = guess_init[j] + (1.0 - guess_init[j]) * p_star;
                    let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);
                    if resp == 1 {
                        ll += p_clipped.ln();
                    } else {
                        ll += (1.0 - p_clipped).ln();
                    }
                }

                log_joint[q] = ll + log_prior[q] + log_weights[q];
            }

            let log_marginal = logsumexp(&log_joint);
            let posterior: Vec<f64> = log_joint
                .iter()
                .map(|&lj| (lj - log_marginal).exp())
                .collect();

            (posterior, log_marginal)
        })
        .collect();

    let posterior_weights: Vec<Vec<f64>> = e_step_results.iter().map(|(p, _)| p.clone()).collect();
    let log_likelihood: f64 = e_step_results.iter().map(|(_, lm)| lm).sum();

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
                    let w = posterior_weights[i][q];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            let mut a = disc_init[j];
            let mut b = diff_init[j];
            let mut c = guess_init[j];

            for _ in 0..max_m_iter {
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

                if grad_a.abs() < m_tol && grad_b.abs() < m_tol && grad_c.abs() < m_tol {
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

    let mut posterior_arr = ndarray::Array2::zeros((n_persons, n_quad));
    for (i, pw) in posterior_weights.iter().enumerate() {
        for (q, &w) in pw.iter().enumerate() {
            posterior_arr[[i, q]] = w;
        }
    }

    (
        disc_new.to_pyarray(py),
        diff_new.to_pyarray(py),
        guess_new.to_pyarray(py),
        posterior_arr.to_pyarray(py),
        log_likelihood,
    )
}

/// Register estimation functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(em_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(gibbs_sample_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(mhrm_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(em_iteration_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(em_iteration_3pl, m)?)?;
    Ok(())
}
