//! Parameter estimation functions (EM, Gibbs, MHRM, Bootstrap).

use numpy::ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{prelude::*, rngs::StdRng};
use rayon::prelude::*;

use crate::counts::binary_item_counts;
use crate::likelihood::{log_2pl_row, log_3pl_row};
use crate::likelihood_cache::{cached_likelihoods, normalize_log_posterior_rows};
use crate::utils::{
    EPSILON, NormalSampler, compute_log_weights, gauss_hermite_quadrature, log_sigmoid, sigmoid,
};

/// EM algorithm for 2PL model fitting
#[pyfunction]
#[pyo3(signature = (responses, n_quadpts, max_iter, tol, frequencies=None))]
#[allow(clippy::type_complexity)]
pub fn em_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
    frequencies: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    usize,
    bool,
)> {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let frequencies = frequencies
        .map(|v| v.as_array().to_vec())
        .unwrap_or_else(|| vec![1.0; n_persons]);
    if frequencies.len() != n_persons || frequencies.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(PyValueError::new_err(
            "frequencies must be positive finite row weights",
        ));
    }
    let (fit, final_log_likelihood) = py.detach(|| {
        let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);
        let log_weights = compute_log_weights(&quad_weights);
        let fit = fit_2pl_em(
            responses,
            &quad_points,
            &log_weights,
            Some(&frequencies),
            vec![1.0; n_items],
            starting_difficulty(responses, Some(&frequencies)),
            max_iter,
            tol,
        );
        let final_log_likelihood = fit.converged_log_likelihood.unwrap_or_else(|| {
            let (_, log_marginal) = e_step_2pl(
                responses,
                &quad_points,
                None,
                &log_weights,
                &fit.discrimination,
                &fit.difficulty,
            );
            weighted_total(&log_marginal, Some(&frequencies))
        });
        (fit, final_log_likelihood)
    });
    let converged = fit.converged_log_likelihood.is_some();
    Ok((
        fit.discrimination.into_pyarray(py),
        fit.difficulty.into_pyarray(py),
        final_log_likelihood,
        fit.iterations,
        converged,
    ))
}

/// Newton-Raphson controls for the per-item 2PL M-step.
struct NewtonControls {
    max_iter: usize,
    tol: f64,
    damping: f64,
    regularization: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
}

/// Fixed M-step controls of the self-contained EM fits (em_fit_2pl, bootstrap).
const EM_FIT_NEWTON: NewtonControls = NewtonControls {
    max_iter: 10,
    tol: 1e-4,
    damping: 0.5,
    regularization: 0.01,
    disc_bounds: (0.1, 5.0),
    diff_bounds: (-6.0, 6.0),
};

struct Em2plFit {
    discrimination: Vec<f64>,
    difficulty: Vec<f64>,
    iterations: usize,
    /// Log-likelihood at the returned parameters when the run converged.
    converged_log_likelihood: Option<f64>,
}

/// Logit-scale starting difficulties from (weighted) proportions correct.
///
/// An item answered correctly with proportion `p` starts at `-logit(p)`, the
/// difficulty of a unit-slope item with that success probability at
/// `theta = 0`. Clamping `p` to `[0.01, 0.99]` keeps starts within
/// `+-ln(99)`, inside the difficulty bounds.
fn starting_difficulty(responses: ArrayView2<'_, i32>, weights: Option<&[f64]>) -> Vec<f64> {
    responses
        .columns()
        .into_iter()
        .map(|column| {
            let mut sum = 0.0;
            let mut count = 0.0;
            for (i, &response) in column.iter().enumerate() {
                if response >= 0 {
                    let weight = weights.map_or(1.0, |weights| weights[i]);
                    sum += response as f64 * weight;
                    count += weight;
                }
            }
            if count > 0.0 {
                let p = (sum / count).clamp(0.01, 0.99);
                ((1.0 - p) / p).ln()
            } else {
                0.0
            }
        })
        .collect()
}

fn weighted_total(values: &[f64], weights: Option<&[f64]>) -> f64 {
    match weights {
        Some(weights) => values.iter().zip(weights).map(|(&v, &w)| w * v).sum(),
        None => values.iter().sum(),
    }
}

/// EM iterations shared by em_fit_2pl and every bootstrap replicate.
#[allow(clippy::too_many_arguments)]
fn fit_2pl_em(
    responses: ArrayView2<'_, i32>,
    quad_points: &[f64],
    log_weights: &[f64],
    weights: Option<&[f64]>,
    mut discrimination: Vec<f64>,
    mut difficulty: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> Em2plFit {
    let mut prev_ll = f64::NEG_INFINITY;
    for iter in 0..max_iter {
        let (posterior, log_marginal) = e_step_2pl(
            responses,
            quad_points,
            None,
            log_weights,
            &discrimination,
            &difficulty,
        );
        let current_ll = weighted_total(&log_marginal, weights);
        if (current_ll - prev_ll).abs() < tol {
            return Em2plFit {
                discrimination,
                difficulty,
                iterations: iter + 1,
                converged_log_likelihood: Some(current_ll),
            };
        }
        prev_ll = current_ll;
        (discrimination, difficulty) = m_step_2pl(
            responses,
            posterior.as_slice().expect("contiguous posterior"),
            quad_points,
            weights,
            &discrimination,
            &difficulty,
            &EM_FIT_NEWTON,
        );
    }
    Em2plFit {
        discrimination,
        difficulty,
        iterations: max_iter,
        converged_log_likelihood: None,
    }
}

/// Posterior weights over the grid and per-person log marginal likelihoods.
fn e_step_2pl(
    responses: ArrayView2<'_, i32>,
    quad_points: &[f64],
    log_prior: Option<&[f64]>,
    log_weights: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
) -> (Array2<f64>, Vec<f64>) {
    let mut posterior = cached_likelihoods(
        responses,
        quad_points.len(),
        &vec![2; responses.ncols()],
        true,
        |q, j, row| log_2pl_row(discrimination[j] * (quad_points[q] - difficulty[j]), row),
    );
    let marginal = normalize_log_posterior_rows(&mut posterior, log_prior, log_weights);
    (posterior, marginal)
}

/// Item-parallel 2PL M-step from expected counts; returns the new (a, b).
fn m_step_2pl(
    responses: ArrayView2<'_, i32>,
    posterior: &[f64],
    quad_points: &[f64],
    weights: Option<&[f64]>,
    discrimination: &[f64],
    difficulty: &[f64],
    controls: &NewtonControls,
) -> (Vec<f64>, Vec<f64>) {
    (0..responses.ncols())
        .into_par_iter()
        .map(|j| {
            let (r_k, n_k) =
                binary_item_counts(responses, posterior, quad_points.len(), j, weights);
            newton_2pl_item(
                &r_k,
                &n_k,
                quad_points,
                discrimination[j],
                difficulty[j],
                controls,
            )
        })
        .unzip()
}

/// Damped, ridge-regularized Newton-Raphson for one item's expected 2PL likelihood.
fn newton_2pl_item(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    mut a: f64,
    mut b: f64,
    controls: &NewtonControls,
) -> (f64, f64) {
    for _ in 0..controls.max_iter {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        let mut hess_aa = 0.0;
        let mut hess_bb = 0.0;
        let mut hess_ab = 0.0;

        for ((&theta, &r), &n) in quad_points.iter().zip(r_k).zip(n_k) {
            if n < EPSILON {
                continue;
            }
            let z = a * (theta - b);
            let p_clipped = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

            let residual = r - n * p_clipped;

            grad_a += residual * (theta - b);
            grad_b += -residual * a;

            let info = n * p_clipped * (1.0 - p_clipped);
            hess_aa += -info * (theta - b) * (theta - b);
            hess_bb += -info * a * a;
            hess_ab += info * a * (theta - b);
        }

        hess_aa -= controls.regularization;
        hess_bb -= controls.regularization;

        let det = hess_aa * hess_bb - hess_ab * hess_ab;
        if det.abs() < EPSILON {
            break;
        }

        let delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det;
        let delta_b = (-hess_ab * grad_a + hess_aa * grad_b) / det;

        a = (a - delta_a * controls.damping).clamp(controls.disc_bounds.0, controls.disc_bounds.1);
        b = (b - delta_b * controls.damping).clamp(controls.diff_bounds.0, controls.diff_bounds.1);

        if delta_a.abs() < controls.tol && delta_b.abs() < controls.tol {
            break;
        }
    }
    (a, b)
}

/// Gibbs sampling for 2PL model
///
/// Keeps every `thin`-th draw from iteration `burnin` onward, which yields
/// `ceil((n_iter - burnin) / thin)` draws.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn gibbs_sample_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be at least 1"));
    }
    if burnin >= n_iter {
        return Err(PyValueError::new_err("burnin must be less than n_iter"));
    }
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_samples = (n_iter - burnin).div_ceil(thin);

    let (disc_arr, diff_arr, theta_arr, ll_arr) = py.detach(|| {
        let mut discrimination: Vec<f64> = vec![1.0; n_items];
        let mut difficulty: Vec<f64> = vec![0.0; n_items];
        let mut theta: Vec<f64> = vec![0.0; n_persons];

        let mut disc_chain = Vec::with_capacity(n_samples * n_items);
        let mut diff_chain = Vec::with_capacity(n_samples * n_items);
        let mut theta_chain = Vec::with_capacity(n_samples * n_persons);
        let mut ll_chain = Vec::with_capacity(n_samples);

        let mut rng = StdRng::seed_from_u64(seed);
        let proposal_theta_sd = 0.5;
        let mut proposal_param = NormalSampler::new(0.0, 0.1);

        for iter in 0..n_iter {
            theta = sample_theta_mh(
                &responses,
                &theta,
                &discrimination,
                &difficulty,
                n_persons,
                n_items,
                &mut rng,
                proposal_theta_sd,
            );

            discrimination = sample_discrimination_mh(
                &responses,
                &theta,
                &discrimination,
                &difficulty,
                n_items,
                &mut rng,
                &mut proposal_param,
            );

            difficulty = sample_difficulty_mh(
                &responses,
                &theta,
                &discrimination,
                &difficulty,
                n_items,
                &mut rng,
                &mut proposal_param,
            );

            if iter >= burnin && (iter - burnin).is_multiple_of(thin) {
                disc_chain.extend_from_slice(&discrimination);
                diff_chain.extend_from_slice(&difficulty);
                theta_chain.extend_from_slice(&theta);
                ll_chain.push(compute_total_ll(
                    &responses,
                    &theta,
                    &discrimination,
                    &difficulty,
                    n_persons,
                    n_items,
                ));
            }
        }

        (
            Array2::from_shape_vec((n_samples, n_items), disc_chain).expect("one row per draw"),
            Array2::from_shape_vec((n_samples, n_items), diff_chain).expect("one row per draw"),
            Array3::from_shape_vec((n_samples, n_persons, 1), theta_chain)
                .expect("one row per draw"),
            Array1::from(ll_chain),
        )
    });

    Ok((
        disc_arr.into_pyarray(py),
        diff_arr.into_pyarray(py),
        theta_arr.into_pyarray(py),
        ll_arr.into_pyarray(py),
    ))
}

#[allow(clippy::too_many_arguments)]
fn sample_theta_mh(
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
    rng: &mut StdRng,
    proposal_sd: f64,
) -> Vec<f64> {
    let seeds: Vec<u64> = (0..n_persons).map(|_| rng.random()).collect();

    let new_theta: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut local_rng = StdRng::seed_from_u64(seeds[i]);
            let mut proposal = NormalSampler::new(0.0, proposal_sd);

            let current = theta[i];
            let proposed = current + proposal.sample(&mut local_rng);

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
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut StdRng,
    proposal: &mut NormalSampler,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_disc = discrimination.to_vec();

    for j in 0..n_items {
        let current = discrimination[j];
        let proposed = (current + proposal.sample(rng)).clamp(0.1, 5.0);

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
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_items: usize,
    rng: &mut StdRng,
    proposal: &mut NormalSampler,
) -> Vec<f64> {
    let n_persons = theta.len();
    let mut new_diff = difficulty.to_vec();

    for j in 0..n_items {
        let current = difficulty[j];
        let proposed = (current + proposal.sample(rng)).clamp(-6.0, 6.0);

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
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    n_persons: usize,
    n_items: usize,
) -> f64 {
    // Sum sequentially so seeded chains reproduce exactly for any thread count.
    let person_ll: Vec<f64> = (0..n_persons)
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
        .collect();
    person_ll.iter().sum()
}

/// Ability sweeps before the first MHRM parameter step.
const MHRM_WARMUP_SWEEPS: usize = 30;
/// Step halvings allowed before an MHRM item keeps its parameters.
const MHRM_MAX_HALVINGS: usize = 10;
/// Eigenvalue floor relative to the largest information eigenvalue.
const MHRM_CURVATURE_FLOOR: f64 = 1e-8;
/// Optimizer box `[a, b]` and per-cycle step limits for MHRM items.
const MHRM_LOWER: [f64; 2] = [0.1, -6.0];
const MHRM_UPPER: [f64; 2] = [5.0, 6.0];
const MHRM_STEP_LIMIT: [f64; 2] = [1.0, 1.0];

/// Metropolis-Hastings Robbins-Monro (Cai, 2010) for 2PL
///
/// After 30 warm-up ability sweeps, every cycle draws abilities with a
/// random-walk Metropolis step and moves each item along its complete-data
/// score, preconditioned by the running complete-data information
/// `G_k = G_{k-1} + g_k (H_k - G_{k-1})`. Burn-in cycles use `g_k = 1`; later
/// gains are `1 / (t + 1)` (`"standard"`) or `min(1, 10 / (t + 10))`
/// (`"adaptive"`) for the `t`-th post-burn-in cycle. Steps hold coordinates that
/// a bound would cut, move at most one unit, and halve until the item's
/// complete-data likelihood on the draw does not decrease. Returns the mean of
/// the post-burn-in iterates (the final iterate when `burnin >= n_cycles`) and
/// the log-likelihood of the final ability draw under those parameters.
#[pyfunction]
#[pyo3(signature = (responses, n_cycles, burnin, proposal_sd, seed, gain_sequence="standard"))]
#[allow(clippy::type_complexity)]
pub fn mhrm_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_cycles: usize,
    burnin: usize,
    proposal_sd: f64,
    seed: u64,
    gain_sequence: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    if n_cycles == 0 {
        return Err(PyValueError::new_err("n_cycles must be at least 1"));
    }
    if !proposal_sd.is_finite() || proposal_sd <= 0.0 {
        return Err(PyValueError::new_err(
            "proposal_sd must be finite and positive",
        ));
    }
    let adaptive_gain = match gain_sequence {
        "standard" => false,
        "adaptive" => true,
        _ => {
            return Err(PyValueError::new_err(
                "gain_sequence must be 'standard' or 'adaptive'",
            ));
        }
    };
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let (discrimination, difficulty, ll) = py.detach(|| {
        let mut discrimination: Vec<f64> = vec![1.0; n_items];
        let mut difficulty: Vec<f64> = vec![0.0; n_items];
        let mut information: Vec<Option<[f64; 3]>> = vec![None; n_items];
        let mut theta: Vec<f64> = vec![0.0; n_persons];
        let mut disc_sum = vec![0.0; n_items];
        let mut diff_sum = vec![0.0; n_items];

        let mut rng = StdRng::seed_from_u64(seed);
        let mut sample = |theta: &[f64], discrimination: &[f64], difficulty: &[f64]| {
            sample_theta_mh(
                &responses,
                theta,
                discrimination,
                difficulty,
                n_persons,
                n_items,
                &mut rng,
                proposal_sd,
            )
        };
        for _ in 0..MHRM_WARMUP_SWEEPS {
            theta = sample(&theta, &discrimination, &difficulty);
        }

        for cycle in 0..n_cycles {
            theta = sample(&theta, &discrimination, &difficulty);

            let gain = if cycle < burnin {
                1.0
            } else if adaptive_gain {
                (10.0 / ((cycle - burnin) as f64 + 10.0)).min(1.0)
            } else {
                1.0 / ((cycle - burnin) as f64 + 1.0)
            };
            let step = |j: usize| {
                mhrm_2pl_item(
                    &responses,
                    &theta,
                    j,
                    [discrimination[j], difficulty[j]],
                    information[j],
                    gain,
                )
            };
            // Small cycles finish faster than a parallel region wakes its workers.
            let updates: Vec<([f64; 2], Option<[f64; 3]>)> = if responses.len() < 32_768 {
                (0..n_items).map(step).collect()
            } else {
                (0..n_items).into_par_iter().map(step).collect()
            };
            for (j, ([a, b], item_information)) in updates.into_iter().enumerate() {
                discrimination[j] = a;
                difficulty[j] = b;
                information[j] = item_information;
            }

            if cycle >= burnin {
                for j in 0..n_items {
                    disc_sum[j] += discrimination[j];
                    diff_sum[j] += difficulty[j];
                }
            }
        }

        if n_cycles > burnin {
            let kept = (n_cycles - burnin) as f64;
            discrimination = disc_sum.iter().map(|sum| sum / kept).collect();
            difficulty = diff_sum.iter().map(|sum| sum / kept).collect();
        }

        let ll = compute_total_ll(
            &responses,
            &theta,
            &discrimination,
            &difficulty,
            n_persons,
            n_items,
        );
        (discrimination, difficulty, ll)
    });

    Ok((
        discrimination.into_pyarray(py),
        difficulty.into_pyarray(py),
        ll,
    ))
}

/// Clipped complete-data loss of item `j` at `[a, b]` with its gradient and
/// symmetric Hessian `[h_aa, h_ab, h_bb]`; clipped probabilities contribute
/// only to the loss, as in the NumPy kernel.
fn complete_data_2pl_item(
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    j: usize,
    [a, b]: [f64; 2],
) -> (f64, [f64; 2], [f64; 3], usize) {
    let mut loss = 0.0;
    let mut gradient = [0.0; 2];
    let mut hessian = [0.0; 3];
    let mut count = 0usize;
    for (&response, &ability) in responses.column(j).iter().zip(theta) {
        if response < 0 {
            continue;
        }
        count += 1;
        let centered = ability - b;
        let raw = sigmoid(a * centered);
        let p = raw.clamp(EPSILON, 1.0 - EPSILON);
        loss -= if response == 1 {
            p.ln()
        } else {
            (1.0 - p).ln()
        };
        if p != raw {
            continue;
        }
        let residual = p - response as f64;
        let weight = p * (1.0 - p);
        gradient[0] += residual * centered;
        gradient[1] -= residual * a;
        hessian[0] += weight * centered * centered;
        hessian[1] -= weight * a * centered + residual;
        hessian[2] += weight * a * a;
    }
    (loss, gradient, hessian, count)
}

/// Solve the symmetric 2x2 system `information @ step = rhs` with eigenvalues
/// replaced by their magnitudes and floored relative to the largest.
fn precondition_2x2([h_aa, h_ab, h_bb]: [f64; 3], rhs: [f64; 2]) -> [f64; 2] {
    let mean = 0.5 * (h_aa + h_bb);
    let radius = (0.25 * (h_aa - h_bb).powi(2) + h_ab * h_ab).sqrt();
    let values = [mean - radius, mean + radius];
    // Eigenvector of the larger eigenvalue; the other is orthogonal.
    let (x, y) = if h_ab != 0.0 {
        (values[1] - h_bb, h_ab)
    } else if h_aa >= h_bb {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let norm = x.hypot(y);
    let first = [x / norm, y / norm];
    let second = [-first[1], first[0]];
    let scale = values[0].abs().max(values[1].abs());
    let floor = (MHRM_CURVATURE_FLOOR * scale).max(f64::MIN_POSITIVE);
    let mut step = [0.0; 2];
    for (vector, value) in [(second, values[0]), (first, values[1])] {
        let coefficient = (vector[0] * rhs[0] + vector[1] * rhs[1]) / value.abs().max(floor);
        step[0] += coefficient * vector[0];
        step[1] += coefficient * vector[1];
    }
    step
}

/// One preconditioned Robbins-Monro update of item `j`, returning the new
/// `[a, b]` and running information. Mirrors `MHRMEstimator._update_parameters`.
fn mhrm_2pl_item(
    responses: &numpy::ndarray::ArrayView2<i32>,
    theta: &[f64],
    j: usize,
    params: [f64; 2],
    information: Option<[f64; 3]>,
    gain: f64,
) -> ([f64; 2], Option<[f64; 3]>) {
    let (loss, gradient, hessian, count) = complete_data_2pl_item(responses, theta, j, params);
    if count == 0 {
        return (params, information);
    }
    let current = match information {
        None => hessian,
        Some(previous) => [0, 1, 2].map(|k| previous[k] + gain * (hessian[k] - previous[k])),
    };

    let held = [0, 1].map(|k| {
        (params[k] <= MHRM_LOWER[k] && gradient[k] > 0.0)
            || (params[k] >= MHRM_UPPER[k] && gradient[k] < 0.0)
    });
    let mut step = match held {
        [false, false] => precondition_2x2(current, [-gradient[0], -gradient[1]]),
        [true, true] => [0.0, 0.0],
        [false, true] => [precondition_1x1(current[0], -gradient[0]), 0.0],
        [true, false] => [0.0, precondition_1x1(current[2], -gradient[1])],
    };
    step = step.map(|value| value * gain);
    if !step.iter().all(|value| value.is_finite()) {
        return (params, Some(current));
    }
    let ratio = (step[0].abs() / MHRM_STEP_LIMIT[0]).max(step[1].abs() / MHRM_STEP_LIMIT[1]);
    if ratio > 1.0 {
        step = step.map(|value| value / ratio);
    }
    for _ in 0..MHRM_MAX_HALVINGS {
        let trial = [0, 1].map(|k| (params[k] + step[k]).clamp(MHRM_LOWER[k], MHRM_UPPER[k]));
        if complete_data_2pl_item(responses, theta, j, trial).0 <= loss {
            return (trial, Some(current));
        }
        step = step.map(|value| value * 0.5);
    }
    (params, Some(current))
}

/// One-coordinate counterpart of `precondition_2x2`.
fn precondition_1x1(information: f64, rhs: f64) -> f64 {
    rhs / information
        .abs()
        .max((MHRM_CURVATURE_FLOOR * information.abs()).max(f64::MIN_POSITIVE))
}

/// Bootstrap parameter estimation for 2PL
#[pyfunction]
#[pyo3(signature = (responses, n_bootstrap, n_quadpts, max_iter, tol, seed, initial_discrimination=None, initial_difficulty=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn bootstrap_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_bootstrap: usize,
    n_quadpts: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    initial_discrimination: Option<PyReadonlyArray1<f64>>,
    initial_difficulty: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let initial_parameters = match (initial_discrimination, initial_difficulty) {
        (None, None) => None,
        (Some(discrimination), Some(difficulty)) => {
            let discrimination = discrimination.as_array().to_vec();
            let difficulty = difficulty.as_array().to_vec();
            if discrimination.len() != n_items || difficulty.len() != n_items {
                return Err(PyValueError::new_err(
                    "initial parameters must contain one value per item",
                ));
            }
            if !discrimination.iter().all(|value| value.is_finite())
                || !difficulty.iter().all(|value| value.is_finite())
            {
                return Err(PyValueError::new_err("initial parameters must be finite"));
            }
            Some((discrimination, difficulty))
        }
        _ => {
            return Err(PyValueError::new_err(
                "initial_discrimination and initial_difficulty must be provided together",
            ));
        }
    };

    let (disc_samples, diff_samples) = py.detach(|| {
        let (quad_points, quad_weights) = gauss_hermite_quadrature(n_quadpts);
        let log_weights = compute_log_weights(&quad_weights);
        let results: Vec<(Vec<f64>, Vec<f64>)> = (0..n_bootstrap)
            .into_par_iter()
            .map(|b| {
                let mut rng = StdRng::seed_from_u64(seed + b as u64);
                let indices: Vec<usize> = (0..n_persons)
                    .map(|_| rng.random_range(0..n_persons))
                    .collect();
                let boot_responses = responses.select(Axis(0), &indices);
                let (discrimination, difficulty) = match &initial_parameters {
                    Some((discrimination, difficulty)) => {
                        (discrimination.clone(), difficulty.clone())
                    }
                    None => (
                        vec![1.0; n_items],
                        starting_difficulty(boot_responses.view(), None),
                    ),
                };
                let fit = fit_2pl_em(
                    boot_responses.view(),
                    &quad_points,
                    &log_weights,
                    None,
                    discrimination,
                    difficulty,
                    max_iter,
                    tol,
                );
                (fit.discrimination, fit.difficulty)
            })
            .collect();

        let mut disc_samples = Array2::zeros((n_bootstrap, n_items));
        let mut diff_samples = Array2::zeros((n_bootstrap, n_items));
        for (b, (disc, diff)) in results.into_iter().enumerate() {
            disc_samples.row_mut(b).assign(&Array1::from(disc));
            diff_samples.row_mut(b).assign(&Array1::from(diff));
        }
        (disc_samples, diff_samples)
    });

    Ok((disc_samples.into_pyarray(py), diff_samples.into_pyarray(py)))
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

    let (disc_new, diff_new, posterior_arr, log_likelihood) = py.detach(|| {
        let log_weights = compute_log_weights(&quad_weights);

        let log_prior_adjustment = crate::utils::normalized_log_gaussian_adjustment(
            &quad_points,
            &quad_weights,
            prior_mean,
            prior_var,
        );
        let (posterior_weights, marginal) = e_step_2pl(
            responses,
            &quad_points,
            Some(&log_prior_adjustment),
            &log_weights,
            &disc_init,
            &diff_init,
        );
        let log_likelihood: f64 = marginal.iter().sum();
        let controls = NewtonControls {
            max_iter: max_m_iter,
            tol: m_tol,
            damping,
            regularization,
            disc_bounds,
            diff_bounds,
        };
        let (disc_new, diff_new) = m_step_2pl(
            responses,
            posterior_weights.as_slice().expect("contiguous posterior"),
            &quad_points,
            None,
            &disc_init,
            &diff_init,
            &controls,
        );

        (disc_new, diff_new, posterior_weights, log_likelihood)
    });

    (
        disc_new.into_pyarray(py),
        diff_new.into_pyarray(py),
        posterior_arr.into_pyarray(py),
        log_likelihood,
    )
}

/// Controls for the projected Newton M-step of one 3PL item.
struct Newton3plControls {
    max_iter: usize,
    tol: f64,
    /// Ridge added to the Fisher information diagonal of `(a, b, c)`.
    ridge: [f64; 3],
    lower: [f64; 3],
    upper: [f64; 3],
}

/// Expected complete-data log-likelihood of one 3PL item on the grid.
fn expected_log_likelihood_3pl(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    [a, b, c]: [f64; 3],
) -> f64 {
    quad_points
        .iter()
        .zip(r_k)
        .zip(n_k)
        .filter(|(_, n)| **n >= EPSILON)
        .map(|((&theta, &r), &n)| {
            let p = (c + (1.0 - c) * sigmoid(a * (theta - b))).clamp(EPSILON, 1.0 - EPSILON);
            r * p.ln() + (n - r) * (1.0 - p).ln()
        })
        .sum()
}

/// Score and ridge-stabilized Fisher information of [`expected_log_likelihood_3pl`].
fn score_and_information_3pl(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    [a, b, c]: [f64; 3],
    ridge: [f64; 3],
) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut score = [0.0; 3];
    let mut information = [[0.0; 3]; 3];
    for i in 0..3 {
        information[i][i] = ridge[i];
    }
    for ((&theta, &r), &n) in quad_points.iter().zip(r_k).zip(n_k) {
        if n < EPSILON {
            continue;
        }
        let p_star = sigmoid(a * (theta - b));
        let p = (c + (1.0 - c) * p_star).clamp(EPSILON, 1.0 - EPSILON);
        let slope = (1.0 - c) * p_star * (1.0 - p_star);
        let gradient = [slope * (theta - b), -slope * a, 1.0 - p_star];
        let variance = (p * (1.0 - p)).max(EPSILON);
        let score_scale = (r - n * p) / variance;
        let info_scale = n / variance;
        for i in 0..3 {
            score[i] += score_scale * gradient[i];
            for k in 0..3 {
                information[i][k] += info_scale * gradient[i] * gradient[k];
            }
        }
    }
    (score, information)
}

/// Solve `information[F, F] step[F] = score[F]` for the free coordinates `F` by
/// Gaussian elimination with partial pivoting; fixed coordinates get a zero step.
fn solve_free_coordinates(
    information: &[[f64; 3]; 3],
    score: &[f64; 3],
    free: &[bool; 3],
) -> Option<[f64; 3]> {
    let index: Vec<usize> = (0..3).filter(|&i| free[i]).collect();
    let m = index.len();
    let mut matrix = [[0.0; 4]; 3];
    for (row, &i) in index.iter().enumerate() {
        for (col, &k) in index.iter().enumerate() {
            matrix[row][col] = information[i][k];
        }
        matrix[row][3] = score[i];
    }
    for col in 0..m {
        let pivot =
            (col..m).max_by(|&x, &y| matrix[x][col].abs().total_cmp(&matrix[y][col].abs()))?;
        matrix.swap(col, pivot);
        if !matrix[col][col].is_finite() || matrix[col][col].abs() < EPSILON {
            return None;
        }
        let pivot_row = matrix[col];
        for row in &mut matrix[col + 1..m] {
            let factor = row[col] / pivot_row[col];
            for (value, &pivot_value) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    let mut solution = [0.0; 3];
    for row in (0..m).rev() {
        let tail: f64 = (row + 1..m).map(|k| matrix[row][k] * solution[k]).sum();
        solution[row] = (matrix[row][3] - tail) / matrix[row][row];
    }
    let mut step = [0.0; 3];
    for (row, &i) in index.iter().enumerate() {
        step[i] = solution[row];
    }
    step.iter().all(|value| value.is_finite()).then_some(step)
}

/// Largest distance from a bound at which [`newton_3pl_item`] treats a
/// coordinate as on that bound.
const ACTIVE_BOUND_WIDTH: f64 = 1e-3;

/// Bounded Fisher-scoring ascent on the expected log-likelihood of one 3PL item.
///
/// This is a projected Newton method with an epsilon-active set (Bertsekas,
/// 1982). A coordinate within `min(ACTIVE_BOUND_WIDTH, |x - P(x + score)|)` of
/// a bound is active when its score points out of the box, and steps onto that
/// bound. Free coordinates on such a band whose reduced Newton step points out
/// of the box are held fixed. The remaining coordinates solve the reduced
/// system, so a guessing parameter at (or just inside) zero leaves the `(a, b)`
/// update to the two-parameter problem instead of the infeasible joint step.
/// Without the band, iterates that stop just inside a bound jam there. Steps
/// are halved until the objective does not drop.
fn newton_3pl_item(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    start: [f64; 3],
    controls: &Newton3plControls,
) -> [f64; 3] {
    let Newton3plControls {
        lower,
        upper,
        ridge,
        ..
    } = *controls;
    let objective = |params: [f64; 3]| expected_log_likelihood_3pl(r_k, n_k, quad_points, params);
    let mut x = start;
    for _ in 0..controls.max_iter {
        let (score, information) = score_and_information_3pl(r_k, n_k, quad_points, x, ridge);
        let projected_score = (0..3)
            .map(|i| (x[i] - (x[i] + score[i]).clamp(lower[i], upper[i])).abs())
            .fold(0.0, f64::max);
        let width = ACTIVE_BOUND_WIDTH.min(projected_score);
        let at_lower = |i: usize, direction: f64| x[i] <= lower[i] + width && direction <= 0.0;
        let at_upper = |i: usize, direction: f64| x[i] >= upper[i] - width && direction >= 0.0;
        let blocked = |i: usize, direction: f64| at_lower(i, direction) || at_upper(i, direction);
        let active: [bool; 3] = std::array::from_fn(|i| blocked(i, score[i]));
        let mut free = active.map(|is_active| !is_active);
        let step = loop {
            let Some(step) = solve_free_coordinates(&information, &score, &free) else {
                break None;
            };
            let outward: Vec<usize> = (0..3)
                .filter(|&i| free[i] && step[i] != 0.0 && blocked(i, step[i]))
                .collect();
            if outward.is_empty() {
                break Some(step);
            }
            for i in outward {
                free[i] = false;
            }
        };
        let Some(mut step) = step else {
            break;
        };
        for i in (0..3).filter(|&i| active[i]) {
            let bound = if at_lower(i, score[i]) {
                lower[i]
            } else {
                upper[i]
            };
            step[i] = bound - x[i];
        }

        let current = objective(x);
        let mut scale = 1.0;
        let mut accepted = None;
        for _ in 0..12 {
            let candidate: [f64; 3] =
                std::array::from_fn(|i| (x[i] + scale * step[i]).clamp(lower[i], upper[i]));
            let value = objective(candidate);
            if value.is_finite() && value + EPSILON >= current {
                accepted = Some(candidate);
                break;
            }
            scale *= 0.5;
        }
        let Some(next) = accepted else {
            break;
        };
        let max_change = (0..3).map(|i| (next[i] - x[i]).abs()).fold(0.0, f64::max);
        x = next;
        if max_change < controls.tol {
            break;
        }
    }
    x
}

/// Single EM iteration for 3PL model (combined E+M step)
///
/// The M-step runs [`newton_3pl_item`] on each item's expected counts. The
/// ridge terms `regularization` (a, b) and `regularization_c` (c) only stabilize
/// the Newton steps; the bounded optimum is that of the unpenalized objective.
#[pyfunction]
#[pyo3(signature = (responses, quad_points, quad_weights, discrimination, difficulty, guessing, prior_mean, prior_var, max_m_iter, m_tol, disc_bounds, diff_bounds, guess_bounds, regularization, regularization_c, frequencies=None))]
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
    regularization: f64,
    regularization_c: f64,
    frequencies: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
)> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let disc_init = discrimination.as_array().to_vec();
    let diff_init = difficulty.as_array().to_vec();
    let guess_init = guessing.as_array().to_vec();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();
    let frequencies = frequencies
        .map(|v| v.as_array().to_vec())
        .unwrap_or_else(|| vec![1.0; n_persons]);
    if frequencies.len() != n_persons || frequencies.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(PyValueError::new_err(
            "frequencies must be positive finite row weights",
        ));
    }

    let (disc_new, diff_new, guess_new, posterior_arr, log_likelihood) = py.detach(|| {
        let log_weights = compute_log_weights(&quad_weights);

        let log_prior_adjustment = crate::utils::normalized_log_gaussian_adjustment(
            &quad_points,
            &quad_weights,
            prior_mean,
            prior_var,
        );

        let mut posterior_weights =
            cached_likelihoods(responses, n_quad, &vec![2; n_items], true, |q, j, row| {
                log_3pl_row(
                    disc_init[j] * (quad_points[q] - diff_init[j]),
                    guess_init[j],
                    row,
                );
            });
        let marginal = normalize_log_posterior_rows(
            &mut posterior_weights,
            Some(&log_prior_adjustment),
            &log_weights,
        );
        let log_likelihood: f64 = marginal
            .iter()
            .zip(&frequencies)
            .map(|(lm, f)| lm * f)
            .sum();

        let posterior_rows = posterior_weights.as_slice().expect("contiguous posterior");
        let controls = Newton3plControls {
            max_iter: max_m_iter,
            tol: m_tol,
            ridge: [regularization, regularization, regularization_c],
            lower: [disc_bounds.0, diff_bounds.0, guess_bounds.0],
            upper: [disc_bounds.1, diff_bounds.1, guess_bounds.1],
        };
        let new_params: Vec<[f64; 3]> = (0..n_items)
            .into_par_iter()
            .map(|j| {
                let (r_k, n_k) =
                    binary_item_counts(responses, posterior_rows, n_quad, j, Some(&frequencies));

                newton_3pl_item(
                    &r_k,
                    &n_k,
                    &quad_points,
                    [disc_init[j], diff_init[j], guess_init[j]],
                    &controls,
                )
            })
            .collect();

        let column =
            |index: usize| -> Vec<f64> { new_params.iter().map(|params| params[index]).collect() };
        let (disc_new, diff_new, guess_new) = (column(0), column(1), column(2));

        (
            disc_new,
            diff_new,
            guess_new,
            posterior_weights,
            log_likelihood,
        )
    });
    Ok((
        disc_new.into_pyarray(py),
        diff_new.into_pyarray(py),
        guess_new.into_pyarray(py),
        posterior_arr.into_pyarray(py),
        log_likelihood,
    ))
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

#[cfg(test)]
mod tests {
    use super::{
        Newton3plControls, NewtonControls, complete_data_2pl_item, expected_log_likelihood_3pl,
        mhrm_2pl_item, newton_2pl_item, newton_3pl_item, precondition_2x2,
        score_and_information_3pl, starting_difficulty,
    };
    use crate::utils::{gauss_hermite_quadrature, sigmoid};
    use numpy::ndarray::Array2;

    const CONTROLS: Newton3plControls = Newton3plControls {
        max_iter: 200,
        tol: 1e-12,
        ridge: [0.01, 0.01, 0.1],
        lower: [0.1, -6.0, 0.0],
        upper: [5.0, 6.0, 0.35],
    };

    /// Expected counts on a 21-point grid for success probabilities `p(theta)`.
    fn counts(p: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (points, weights) = gauss_hermite_quadrature(21);
        let n_k: Vec<f64> = weights.iter().map(|w| 2000.0 * w).collect();
        let r_k = points.iter().zip(&n_k).map(|(&t, &n)| n * p(t)).collect();
        (points, r_k, n_k)
    }

    #[test]
    fn starting_difficulties_are_negative_logits_of_proportions_correct() {
        // Columns: p = 0.5, 0.8, 0.2 after skipping missing responses, then
        // an always-correct item clamped to p = 0.99 and an unanswered item.
        let responses = Array2::from_shape_vec(
            (5, 5),
            vec![
                1, 1, 0, 1, -1, //
                0, 1, 0, 1, -1, //
                1, 1, 1, 1, -1, //
                0, 1, 0, 1, -1, //
                -1, 0, 0, 1, -1,
            ],
        )
        .unwrap();
        let start = starting_difficulty(responses.view(), None);
        let expected = [0.0, -4.0f64.ln(), 4.0f64.ln(), -99.0f64.ln(), 0.0];
        for (value, target) in start.iter().zip(expected) {
            assert!((value - target).abs() < 1e-12, "{start:?}");
        }
        // Weights scale each person's contribution: p = 3 / 4 for item 0.
        let weighted = starting_difficulty(responses.view(), Some(&[3.0, 1.0, 0.0, 0.0, 1.0]));
        assert!((weighted[0] + 3.0f64.ln()).abs() < 1e-12, "{weighted:?}");
    }

    #[test]
    fn interior_optimum_recovers_generating_parameters() {
        let (points, r_k, n_k) = counts(|t| 0.2 + 0.8 * sigmoid(1.3 * (t - 0.4)));
        let [a, b, c] = newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.1], &CONTROLS);

        assert!((a - 1.3).abs() < 1e-6, "a = {a}");
        assert!((b - 0.4).abs() < 1e-6, "b = {b}");
        assert!((c - 0.2).abs() < 1e-6, "c = {c}");
    }

    #[test]
    fn guessing_on_its_bound_leaves_the_two_parameter_optimum() {
        // A lower asymptote below zero puts the unconstrained guessing optimum
        // outside the box, so the bounded optimum is the 2PL fit with c = 0.
        let (points, r_k, n_k) =
            counts(|t| (-0.04 + 1.04 * sigmoid(1.1 * (t + 0.3))).clamp(1e-6, 1.0));
        let [a, b, c] = newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.2], &CONTROLS);

        let two_parameter = NewtonControls {
            max_iter: 200,
            tol: 1e-13,
            damping: 1.0,
            regularization: 0.0,
            disc_bounds: (0.1, 5.0),
            diff_bounds: (-6.0, 6.0),
        };
        let (a_2pl, b_2pl) = newton_2pl_item(&r_k, &n_k, &points, 1.0, 0.0, &two_parameter);

        assert_eq!(c, 0.0);
        assert!((a - a_2pl).abs() < 1e-8, "a = {a}, 2PL a = {a_2pl}");
        assert!((b - b_2pl).abs() < 1e-8, "b = {b}, 2PL b = {b_2pl}");
        let (score, _) = score_and_information_3pl(&r_k, &n_k, &points, [a, b, c], [0.0; 3]);
        assert!(score[0].abs() < 1e-6 && score[1].abs() < 1e-6, "{score:?}");
        assert!(score[2] < 0.0, "guessing score must point below its bound");
    }

    #[test]
    fn guessing_just_inside_its_upper_bound_moves_onto_it() {
        // The joint step from c just below 0.35 leaves the box, and its clipped
        // (a, b) part lowers the objective at every step length, so without the
        // active band the iterate never moves.
        let (points, r_k, n_k) = counts(|t| 0.5 + 0.5 * sigmoid(4.755 * (t - 1.288)));
        let [a, b, c] = newton_3pl_item(&r_k, &n_k, &points, [0.3, 3.3, 0.3495], &CONTROLS);

        assert_eq!(c, 0.35);
        let (score, _) = score_and_information_3pl(&r_k, &n_k, &points, [a, b, c], [0.0; 3]);
        assert!(score[0].abs() < 1e-6 && score[1].abs() < 1e-6, "{score:?}");
        assert!(score[2] > 0.0, "guessing score must point above its bound");
    }

    #[test]
    fn guessing_just_above_zero_reaches_the_two_parameter_generator() {
        // Without the active band this start jams at b = -6 with c = 3e-4,
        // 130 log-likelihood units below the optimum.
        let (points, r_k, n_k) = counts(|t| sigmoid(1.56 * (t + 1.5)));
        let [a, b, c] = newton_3pl_item(&r_k, &n_k, &points, [1.5, 2.5, 3e-4], &CONTROLS);

        assert!((a - 1.56).abs() < 1e-6, "a = {a}");
        assert!((b + 1.5).abs() < 1e-6, "b = {b}");
        assert!(c.abs() < 1e-6, "c = {c}");
    }

    #[test]
    fn bounded_steps_never_decrease_the_objective() {
        let (points, r_k, n_k) = counts(|t| 0.3 + 0.7 * sigmoid(2.5 * (t - 1.0)));
        let one_step = Newton3plControls {
            max_iter: 1,
            ..CONTROLS
        };
        let mut params = [0.5, -1.0, 0.0];
        let mut previous = expected_log_likelihood_3pl(&r_k, &n_k, &points, params);
        for _ in 0..40 {
            params = newton_3pl_item(&r_k, &n_k, &points, params, &one_step);
            let value = expected_log_likelihood_3pl(&r_k, &n_k, &points, params);
            assert!(value + 1e-9 >= previous);
            assert!(
                (0..3).all(|i| CONTROLS.lower[i] <= params[i] && params[i] <= CONTROLS.upper[i])
            );
            previous = value;
        }
        assert!((params[2] - 0.3).abs() < 1e-4, "{params:?}");
    }

    #[test]
    fn precondition_solves_with_curvature_magnitudes() {
        // [[3, 1], [1, 2]] is positive definite: an ordinary solve.
        let step = precondition_2x2([3.0, 1.0, 2.0], [1.0, -2.0]);
        assert!((step[0] - 0.8).abs() < 1e-12 && (step[1] + 1.4).abs() < 1e-12);
        // A negative eigenvalue is replaced by its magnitude.
        let step = precondition_2x2([2.0, 0.0, -4.0], [1.0, -2.0]);
        assert!((step[0] - 0.5).abs() < 1e-12 && (step[1] + 0.5).abs() < 1e-12);
        let step = precondition_2x2([0.0, 0.0, 0.0], [1.0, -2.0]);
        assert!(step.iter().all(|value| value.is_finite()));
    }

    /// Abilities on a grid and deterministic 2PL responses at `[a, b]`.
    fn grid_responses(a: f64, b: f64) -> (Vec<f64>, Array2<i32>) {
        let theta: Vec<f64> = (0..400).map(|i| -3.0 + 6.0 * i as f64 / 399.0).collect();
        let responses = Array2::from_shape_fn((theta.len(), 1), |(i, _)| {
            let p = sigmoid(a * (theta[i] - b));
            i32::from((i as f64 * 0.618_034).fract() < p)
        });
        (theta, responses)
    }

    #[test]
    fn complete_data_hessian_matches_gradient_differences() {
        let (theta, responses) = grid_responses(1.4, 0.3);
        let view = responses.view();
        let params = [1.1, -0.2];
        let (_, _, hessian, count) = complete_data_2pl_item(&view, &theta, 0, params);
        assert_eq!(count, theta.len());
        let step = 1e-6;
        for (k, rows) in [(0, [0, 1]), (1, [1, 2])] {
            let mut forward = params;
            forward[k] += step;
            let mut backward = params;
            backward[k] -= step;
            let up = complete_data_2pl_item(&view, &theta, 0, forward).1;
            let down = complete_data_2pl_item(&view, &theta, 0, backward).1;
            for (row, index) in rows.into_iter().enumerate() {
                let expected = (up[row] - down[row]) / (2.0 * step);
                assert!((hessian[index] - expected).abs() < 1e-5 * expected.abs().max(1.0));
            }
        }
    }

    #[test]
    fn unit_gain_steps_climb_to_the_complete_data_maximum() {
        let (theta, responses) = grid_responses(1.4, 0.3);
        let view = responses.view();
        let mut params = [1.0, 0.0];
        let mut information = None;
        let mut previous = f64::INFINITY;
        for _ in 0..30 {
            (params, information) = mhrm_2pl_item(&view, &theta, 0, params, information, 1.0);
            let (loss, gradient, _, _) = complete_data_2pl_item(&view, &theta, 0, params);
            assert!(loss <= previous + 1e-9);
            previous = loss;
            if gradient.iter().all(|value| value.abs() < 1e-8) {
                break;
            }
        }
        let gradient = complete_data_2pl_item(&view, &theta, 0, params).1;
        assert!(
            gradient.iter().all(|value| value.abs() < 1e-6),
            "{gradient:?}"
        );
        assert!((params[0] - 1.4).abs() < 0.3 && (params[1] - 0.3).abs() < 0.2);
    }
}
