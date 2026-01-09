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

/// Register estimation functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(em_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(gibbs_sample_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(mhrm_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_fit_2pl, m)?)?;
    Ok(())
}
