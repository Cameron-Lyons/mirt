//! Parameter estimation functions (MHRM, bootstrap EM, 3PL EM iteration).

use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{prelude::*, rngs::StdRng};
use rayon::prelude::*;

use crate::box_newton::{BoxNewtonControls, projected_newton};
use crate::counts::binary_item_counts;
use crate::item_prior::CoordinatePrior;
use crate::likelihood::{log_2pl_row, log_3pl_row};
use crate::likelihood_cache::{cached_likelihoods, normalize_log_posterior_rows};
use crate::polytomous_mstep::EMThreadPool;
use crate::utils::{
    EPSILON, NormalSampler, check_binary_responses, check_bounds, check_quadrature,
    compute_log_weights, log_sigmoid, sigmoid, stream_seed,
};

/// Newton iterations and tolerance of each bootstrap M-step, with the ridge
/// that stabilizes its `(a, b)` steps.
const BOOTSTRAP_M_ITER: usize = 10;
const BOOTSTRAP_M_TOL: f64 = 1e-4;
const BOOTSTRAP_RIDGE: [f64; 2] = [0.01, 0.01];

/// Logit-scale starting difficulties from proportions correct.
///
/// An item answered correctly with proportion `p` starts at `-logit(p)`, the
/// difficulty of a unit-slope item with that success probability at
/// `theta = 0`. Clamping `p` to `[0.01, 0.99]` keeps starts within
/// `+-ln(99)`.
fn starting_difficulty(responses: ArrayView2<'_, i32>) -> Vec<f64> {
    responses
        .columns()
        .into_iter()
        .map(|column| {
            let mut sum = 0.0;
            let mut count = 0.0;
            for &response in column {
                if response >= 0 {
                    sum += response as f64;
                    count += 1.0;
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

/// EM iterations of one bootstrap replicate, stopped when the marginal
/// log-likelihood changes by less than `tol`; returns the final `(a, b)`.
fn fit_2pl_em(
    responses: ArrayView2<'_, i32>,
    quad_points: &[f64],
    log_weights: &[f64],
    mut parameters: (Vec<f64>, Vec<f64>),
    controls: &BoxNewtonControls<2>,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut prev_ll = f64::NEG_INFINITY;
    for _ in 0..max_iter {
        let (discrimination, difficulty) = &parameters;
        let (posterior, log_marginal) = e_step_2pl(
            responses,
            quad_points,
            log_weights,
            discrimination,
            difficulty,
        );
        let current_ll: f64 = log_marginal.iter().sum();
        if (current_ll - prev_ll).abs() < tol {
            break;
        }
        prev_ll = current_ll;
        parameters = m_step_2pl(
            responses,
            posterior.as_slice().expect("contiguous posterior"),
            quad_points,
            discrimination,
            difficulty,
            controls,
        );
    }
    parameters
}

/// Posterior weights over the grid and per-person log marginal likelihoods.
fn e_step_2pl(
    responses: ArrayView2<'_, i32>,
    quad_points: &[f64],
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
    let marginal = normalize_log_posterior_rows(&mut posterior, None, log_weights);
    (posterior, marginal)
}

/// Item-parallel 2PL M-step from expected counts; returns the new (a, b).
fn m_step_2pl(
    responses: ArrayView2<'_, i32>,
    posterior: &[f64],
    quad_points: &[f64],
    discrimination: &[f64],
    difficulty: &[f64],
    controls: &BoxNewtonControls<2>,
) -> (Vec<f64>, Vec<f64>) {
    (0..responses.ncols())
        .into_par_iter()
        .map(|j| {
            let (r_k, n_k) = binary_item_counts(responses, posterior, quad_points.len(), j, None);
            let [a, b] = newton_2pl_item(
                &r_k,
                &n_k,
                quad_points,
                [discrimination[j], difficulty[j]],
                controls,
            );
            (a, b)
        })
        .unzip()
}

/// Expected complete-data log-likelihood of one 2PL item on the grid.
fn expected_log_likelihood_2pl(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    [a, b]: [f64; 2],
) -> f64 {
    quad_points
        .iter()
        .zip(r_k)
        .zip(n_k)
        .filter(|(_, n)| **n >= EPSILON)
        .map(|((&theta, &r), &n)| {
            let z = a * (theta - b);
            r * log_sigmoid(z) + (n - r) * log_sigmoid(-z)
        })
        .sum()
}

/// Score and ridge-stabilized Fisher information of [`expected_log_likelihood_2pl`].
fn score_and_information_2pl(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    [a, b]: [f64; 2],
    ridge: [f64; 2],
) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut score = [0.0; 2];
    let mut information = [[ridge[0], 0.0], [0.0, ridge[1]]];
    for ((&theta, &r), &n) in quad_points.iter().zip(r_k).zip(n_k) {
        if n < EPSILON {
            continue;
        }
        let centered = theta - b;
        let p = sigmoid(a * centered);
        let residual = r - n * p;
        let weight = n * p * (1.0 - p);
        score[0] += residual * centered;
        score[1] -= residual * a;
        information[0][0] += weight * centered * centered;
        information[0][1] -= weight * a * centered;
        information[1][1] += weight * a * a;
    }
    information[1][0] = information[0][1];
    (score, information)
}

/// Bounded Fisher-scoring ascent on one item's expected 2PL log-likelihood.
///
/// A difficulty held by its bound leaves the slope to the one-dimensional
/// problem (see [`projected_newton`]), so the step ends at a stationary point
/// of the box-constrained objective. Clamping each coordinate of the joint
/// step instead kept the coupling term of the held coordinate in the slope
/// update and stopped where the slope's score was not zero.
fn newton_2pl_item(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    start: [f64; 2],
    controls: &BoxNewtonControls<2>,
) -> [f64; 2] {
    projected_newton(
        start,
        controls,
        |params| expected_log_likelihood_2pl(r_k, n_k, quad_points, params),
        |params| score_and_information_2pl(r_k, n_k, quad_points, params, controls.ridge),
    )
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
/// Per-cycle step limits of MHRM items in `[a, b]`.
const MHRM_STEP_LIMIT: [f64; 2] = [1.0, 1.0];

/// Optimizer box of MHRM items: lower and upper `[a, b]`.
#[derive(Clone, Copy)]
struct ItemBox {
    lower: [f64; 2],
    upper: [f64; 2],
}

/// Metropolis-Hastings Robbins-Monro (Cai, 2010) for 2PL
///
/// After 30 warm-up ability sweeps, every cycle draws abilities with a
/// random-walk Metropolis step and moves each item along its complete-data
/// score, preconditioned by the running complete-data information
/// `G_k = G_{k-1} + g_k (H_k - G_{k-1})`. Burn-in cycles use `g_k = 1`; later
/// gains are `1 / (t + 1)` (`"standard"`) or `min(1, 10 / (t + 10))`
/// (`"adaptive"`) for the `t`-th post-burn-in cycle. Steps hold coordinates that
/// a bound would cut, move at most one unit, and halve until the item's
/// complete-data likelihood on the draw does not decrease. Slopes stay in
/// `disc_bounds` and difficulties in `diff_bounds`; the chain starts from unit
/// slopes and zero difficulties clamped into that box. Returns the mean of the
/// post-burn-in iterates (the final iterate when `burnin >= n_cycles`) and the
/// log-likelihood of the final ability draw under those parameters.
#[pyfunction]
#[pyo3(signature = (responses, n_cycles, burnin, proposal_sd, seed, disc_bounds, diff_bounds, gain_sequence="standard"))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn mhrm_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_cycles: usize,
    burnin: usize,
    proposal_sd: f64,
    seed: u64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    gain_sequence: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    check_bounds("disc_bounds", disc_bounds)?;
    check_bounds("diff_bounds", diff_bounds)?;
    let item_box = ItemBox {
        lower: [disc_bounds.0, diff_bounds.0],
        upper: [disc_bounds.1, diff_bounds.1],
    };
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
    check_binary_responses(responses)?;
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let (discrimination, difficulty, ll) = py.detach(|| {
        let mut discrimination: Vec<f64> =
            vec![1.0f64.clamp(disc_bounds.0, disc_bounds.1); n_items];
        let mut difficulty: Vec<f64> = vec![0.0f64.clamp(diff_bounds.0, diff_bounds.1); n_items];
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
                    item_box,
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
    item_box: ItemBox,
) -> ([f64; 2], Option<[f64; 3]>) {
    let ItemBox { lower, upper } = item_box;
    let (loss, gradient, hessian, count) = complete_data_2pl_item(responses, theta, j, params);
    if count == 0 {
        return (params, information);
    }
    let current = match information {
        None => hessian,
        Some(previous) => [0, 1, 2].map(|k| previous[k] + gain * (hessian[k] - previous[k])),
    };

    let held = [0, 1].map(|k| {
        (params[k] <= lower[k] && gradient[k] > 0.0) || (params[k] >= upper[k] && gradient[k] < 0.0)
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
        let trial = [0, 1].map(|k| (params[k] + step[k]).clamp(lower[k], upper[k]));
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
///
/// Each replicate resamples persons with replacement and fits the 2PL by
/// marginal maximum likelihood on the given grid, whose weights integrate
/// against the latent density. Its M-steps keep slopes in `disc_bounds` and
/// difficulties in `diff_bounds` by bounded Fisher scoring, and its EM loop
/// stops once the log-likelihood changes by less than `tol`. Replicate `b`
/// draws from its own stream, seeded by mixing `seed` and `b`, so replicates
/// of neighbouring seeds are unrelated and results do not depend on the
/// thread count. Starts are the given parameters, clamped into the box, or
/// unit slopes and logit difficulties.
#[pyfunction]
#[pyo3(signature = (
    responses,
    n_bootstrap,
    quad_points,
    quad_weights,
    max_iter,
    tol,
    seed,
    disc_bounds,
    diff_bounds,
    initial_discrimination=None,
    initial_difficulty=None,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn bootstrap_fit_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    n_bootstrap: usize,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
    seed: u64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    initial_discrimination: Option<PyReadonlyArray1<f64>>,
    initial_difficulty: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array().to_vec();
    let quad_weights = quad_weights.as_array().to_vec();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    check_binary_responses(responses)?;
    check_quadrature(&quad_points, &quad_weights)?;
    check_bounds("disc_bounds", disc_bounds)?;
    check_bounds("diff_bounds", diff_bounds)?;
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be at least 1"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    let controls = BoxNewtonControls {
        max_iter: BOOTSTRAP_M_ITER,
        tol: BOOTSTRAP_M_TOL,
        ridge: BOOTSTRAP_RIDGE,
        lower: [disc_bounds.0, diff_bounds.0],
        upper: [disc_bounds.1, diff_bounds.1],
    };
    let clamp_into_box = |discrimination: Vec<f64>, difficulty: Vec<f64>| {
        (
            discrimination
                .into_iter()
                .map(|a| a.clamp(disc_bounds.0, disc_bounds.1))
                .collect::<Vec<f64>>(),
            difficulty
                .into_iter()
                .map(|b| b.clamp(diff_bounds.0, diff_bounds.1))
                .collect::<Vec<f64>>(),
        )
    };
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
            Some(clamp_into_box(discrimination, difficulty))
        }
        _ => {
            return Err(PyValueError::new_err(
                "initial_discrimination and initial_difficulty must be provided together",
            ));
        }
    };

    let (disc_samples, diff_samples) = py.detach(|| {
        let log_weights = compute_log_weights(&quad_weights);
        let results: Vec<(Vec<f64>, Vec<f64>)> = (0..n_bootstrap)
            .into_par_iter()
            .map(|b| {
                let mut rng = StdRng::seed_from_u64(stream_seed(seed, &[b as u64]));
                let indices: Vec<usize> = (0..n_persons)
                    .map(|_| rng.random_range(0..n_persons))
                    .collect();
                let boot_responses = responses.select(Axis(0), &indices);
                let start = match &initial_parameters {
                    Some(parameters) => parameters.clone(),
                    None => clamp_into_box(
                        vec![1.0; n_items],
                        starting_difficulty(boot_responses.view()),
                    ),
                };
                fit_2pl_em(
                    boot_responses.view(),
                    &quad_points,
                    &log_weights,
                    start,
                    &controls,
                    max_iter,
                    tol,
                )
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

/// Controls of the projected Newton M-step of one 3PL item, with the ridge
/// of `(a, b, c)`.
type Newton3plControls = BoxNewtonControls<3>;

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

/// Bounded Fisher-scoring ascent on the expected log-posterior of one 3PL item.
///
/// The objective is the expected log-likelihood plus the log densities of the
/// `(a, b, c)` priors. A guessing parameter at (or just inside) zero leaves
/// the `(a, b)` update to the two-parameter problem instead of the infeasible
/// joint step (see [`projected_newton`]). A prior adds its gradient to the
/// score and its curvature, where negative, to the information, so the steps
/// stay ascent directions while their fixed point is the posterior mode.
fn newton_3pl_item(
    r_k: &[f64],
    n_k: &[f64],
    quad_points: &[f64],
    start: [f64; 3],
    controls: &Newton3plControls,
    priors: &[CoordinatePrior; 3],
) -> [f64; 3] {
    let has_prior = priors.iter().any(|prior| *prior != CoordinatePrior::None);
    projected_newton(
        start,
        controls,
        |params| {
            let value = expected_log_likelihood_3pl(r_k, n_k, quad_points, params);
            if has_prior {
                value
                    + (0..3)
                        .map(|i| priors[i].log_density(params[i]))
                        .sum::<f64>()
            } else {
                value
            }
        },
        |params| {
            let (mut score, mut information) =
                score_and_information_3pl(r_k, n_k, quad_points, params, controls.ridge);
            if has_prior {
                for i in 0..3 {
                    let (gradient, curvature) = priors[i].derivatives(params[i]);
                    score[i] += gradient;
                    information[i][i] += (-curvature).max(0.0);
                }
            }
            (score, information)
        },
    )
}

/// Single EM iteration for 3PL model (combined E+M step)
///
/// The M-step runs [`newton_3pl_item`] on each item's expected counts. The
/// ridge terms `regularization` (a, b) and `regularization_c` (c) only stabilize
/// the Newton steps; the bounded optimum is that of the unpenalized objective.
/// Items are optimized in parallel on `pool`, the fit's [`EMThreadPool`] of
/// `min(n_jobs, n_items)` workers, when one is given, and otherwise on the
/// global pool, which the E-step always uses. Each item's step is independent
/// of the others, so the results do not depend on the pool. `priors`, a
/// `(3, 3)` table of `(code, first, second)` rows for `a`, `b` and `c` (see
/// [`CoordinatePrior::decode`]), turns each M-step into a posterior-mode step.
#[pyfunction]
#[pyo3(signature = (responses, quad_points, quad_weights, discrimination, difficulty, guessing, prior_mean, prior_var, max_m_iter, m_tol, disc_bounds, diff_bounds, guess_bounds, regularization, regularization_c, frequencies=None, n_jobs=1, pool=None, priors=None))]
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
    n_jobs: usize,
    pool: Option<PyRef<'_, EMThreadPool>>,
    priors: Option<PyReadonlyArray2<f64>>,
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
    check_binary_responses(responses)?;
    check_quadrature(&quad_points, &quad_weights)?;
    for (name, values) in [
        ("discrimination", &disc_init),
        ("difficulty", &diff_init),
        ("guessing", &guess_init),
    ] {
        if values.len() != n_items {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({n_items},), got ({},)",
                values.len()
            )));
        }
    }
    if !prior_mean.is_finite() || !prior_var.is_finite() || prior_var <= 0.0 {
        return Err(PyValueError::new_err(
            "prior_mean must be finite and prior_var finite and positive",
        ));
    }
    if max_m_iter == 0 {
        return Err(PyValueError::new_err(
            "max_m_iter must be a positive integer",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs must be positive"));
    }
    check_bounds("disc_bounds", disc_bounds)?;
    check_bounds("diff_bounds", diff_bounds)?;
    check_bounds("guess_bounds", guess_bounds)?;
    let frequencies = frequencies
        .map(|v| v.as_array().to_vec())
        .unwrap_or_else(|| vec![1.0; n_persons]);
    if frequencies.len() != n_persons || frequencies.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(PyValueError::new_err(
            "frequencies must be positive finite row weights",
        ));
    }
    let item_priors = match priors {
        None => [CoordinatePrior::None; 3],
        Some(table) => {
            let table = table.as_array();
            let decoded: Option<Vec<CoordinatePrior>> = (table.dim() == (3, 3))
                .then(|| {
                    (0..3)
                        .map(|i| {
                            CoordinatePrior::decode(table[[i, 0]], table[[i, 1]], table[[i, 2]])
                        })
                        .collect()
                })
                .flatten();
            let decoded = decoded.ok_or_else(|| {
                PyValueError::new_err("priors must be a (3, 3) table of (code, first, second) rows")
            })?;
            [decoded[0], decoded[1], decoded[2]]
        }
    };
    let shared_pool = pool.as_ref().map(|value| value.pool());
    if shared_pool.is_some_and(|pool| pool.current_num_threads() != n_jobs.min(n_items)) {
        return Err(PyValueError::new_err(
            "pool size must match n_jobs and item count",
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
        let fit = |j: usize| {
            let (r_k, n_k) =
                binary_item_counts(responses, posterior_rows, n_quad, j, Some(&frequencies));
            newton_3pl_item(
                &r_k,
                &n_k,
                &quad_points,
                [disc_init[j], diff_init[j], guess_init[j]],
                &controls,
                &item_priors,
            )
        };
        let new_params: Vec<[f64; 3]> = if n_items < 2 {
            (0..n_items).map(fit).collect()
        } else if let Some(pool) = shared_pool {
            pool.install(|| (0..n_items).into_par_iter().map(fit).collect())
        } else {
            (0..n_items).into_par_iter().map(fit).collect()
        };

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
    m.add_function(wrap_pyfunction!(mhrm_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_fit_2pl, m)?)?;
    m.add_function(wrap_pyfunction!(em_iteration_3pl, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ItemBox, Newton3plControls, complete_data_2pl_item, expected_log_likelihood_2pl,
        expected_log_likelihood_3pl, mhrm_2pl_item, newton_2pl_item, newton_3pl_item,
        precondition_2x2, score_and_information_2pl, score_and_information_3pl,
        starting_difficulty,
    };
    use crate::box_newton::BoxNewtonControls;
    use crate::item_prior::CoordinatePrior;

    const NO_PRIORS: [CoordinatePrior; 3] = [CoordinatePrior::None; 3];
    use crate::utils::sigmoid;
    use numpy::ndarray::Array2;

    const CONTROLS: Newton3plControls = Newton3plControls {
        max_iter: 200,
        tol: 1e-12,
        ridge: [0.01, 0.01, 0.1],
        lower: [0.1, -6.0, 0.0],
        upper: [5.0, 6.0, 0.35],
    };

    /// Expected counts of 2000 standard normal abilities on a 41-point grid
    /// over `[-6, 6]` for success probabilities `p(theta)`.
    fn counts(p: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let points: Vec<f64> = (0..41).map(|k| -6.0 + 0.3 * k as f64).collect();
        let density: Vec<f64> = points.iter().map(|t| (-0.5 * t * t).exp()).collect();
        let total: f64 = density.iter().sum();
        let n_k: Vec<f64> = density.iter().map(|w| 2000.0 * w / total).collect();
        let r_k = points.iter().zip(&n_k).map(|(&t, &n)| n * p(t)).collect();
        (points, r_k, n_k)
    }

    const WIDE_BOX: ItemBox = ItemBox {
        lower: [0.1, -30.0],
        upper: [20.0, 30.0],
    };

    const TWO_PARAMETER: BoxNewtonControls<2> = BoxNewtonControls {
        max_iter: 200,
        tol: 1e-13,
        ridge: [0.0; 2],
        lower: [0.1, -6.0],
        upper: [5.0, 6.0],
    };

    #[test]
    fn two_parameter_interior_optimum_recovers_generating_parameters() {
        let (points, r_k, n_k) = counts(|t| sigmoid(1.7 * (t + 0.6)));
        let [a, b] = newton_2pl_item(&r_k, &n_k, &points, [1.0, 0.0], &TWO_PARAMETER);

        assert!((a - 1.7).abs() < 1e-8, "a = {a}");
        assert!((b + 0.6).abs() < 1e-8, "b = {b}");
    }

    #[test]
    fn difficulty_on_its_bound_leaves_a_stationary_slope() {
        // Regression (bootstrap M-step): an easy, weak item whose difficulty
        // optimum lies below -6. Clamping each coordinate of the joint Newton
        // step kept the coupling term of the held difficulty in the slope
        // update, so the slope stopped where its score was not zero.
        let (points, r_k, n_k) = counts(|t| sigmoid(0.45 * (t + 8.0)));
        let [a, b] = newton_2pl_item(&r_k, &n_k, &points, [1.0, 0.0], &TWO_PARAMETER);

        assert_eq!(b, -6.0);
        let (score, _) = score_and_information_2pl(&r_k, &n_k, &points, [a, b], [0.0; 2]);
        assert!(score[0].abs() < 1e-8, "{score:?}");
        assert!(
            score[1] < 0.0,
            "difficulty score must point below its bound"
        );
        // The slope maximizes the objective along the bound.
        let best = expected_log_likelihood_2pl(&r_k, &n_k, &points, [a, b]);
        for k in 1..2000 {
            let slope = 0.1 + 0.0025 * k as f64;
            let value = expected_log_likelihood_2pl(&r_k, &n_k, &points, [slope, -6.0]);
            assert!(value <= best + 1e-9, "slope {slope} beats {a}");
        }
    }

    #[test]
    fn two_parameter_steps_never_decrease_the_objective() {
        let (points, r_k, n_k) = counts(|t| sigmoid(4.0 * (t - 5.0)));
        let one_step = BoxNewtonControls {
            max_iter: 1,
            ..TWO_PARAMETER
        };
        let mut params = [0.3, -2.0];
        let mut previous = expected_log_likelihood_2pl(&r_k, &n_k, &points, params);
        for _ in 0..60 {
            params = newton_2pl_item(&r_k, &n_k, &points, params, &one_step);
            let value = expected_log_likelihood_2pl(&r_k, &n_k, &points, params);
            assert!(value + 1e-9 >= previous);
            assert!((0..2).all(|i| {
                TWO_PARAMETER.lower[i] <= params[i] && params[i] <= TWO_PARAMETER.upper[i]
            }));
            previous = value;
        }
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
        let start = starting_difficulty(responses.view());
        let expected = [0.0, -4.0f64.ln(), 4.0f64.ln(), -99.0f64.ln(), 0.0];
        for (value, target) in start.iter().zip(expected) {
            assert!((value - target).abs() < 1e-12, "{start:?}");
        }
    }

    #[test]
    fn interior_optimum_recovers_generating_parameters() {
        let (points, r_k, n_k) = counts(|t| 0.2 + 0.8 * sigmoid(1.3 * (t - 0.4)));
        let [a, b, c] =
            newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.1], &CONTROLS, &NO_PRIORS);

        assert!((a - 1.3).abs() < 1e-6, "a = {a}");
        assert!((b - 0.4).abs() < 1e-6, "b = {b}");
        assert!((c - 0.2).abs() < 1e-6, "c = {c}");
    }

    #[test]
    fn priors_move_the_optimum_to_the_posterior_mode() {
        let (points, r_k, n_k) = counts(|t| 0.2 + 0.8 * sigmoid(1.3 * (t - 0.4)));
        let priors = [
            CoordinatePrior::LogNormal {
                mu: 0.0,
                sigma: 0.5,
            },
            CoordinatePrior::Normal {
                mu: 0.0,
                sigma: 2.0,
            },
            CoordinatePrior::Beta {
                alpha: 5.0,
                beta: 17.0,
            },
        ];
        let mode = newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.1], &CONTROLS, &priors);
        let mle = newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.1], &CONTROLS, &NO_PRIORS);

        let (mut score, _) = score_and_information_3pl(&r_k, &n_k, &points, mode, [0.0; 3]);
        for i in 0..3 {
            score[i] += priors[i].derivatives(mode[i]).0;
        }
        assert!(score.iter().all(|value| value.abs() < 1e-6), "{score:?}");
        // The expected counts follow the generating item exactly, so the
        // likelihood alone recovers it; the priors pull the mode away.
        assert!(
            (0..3).any(|i| (mode[i] - mle[i]).abs() > 1e-3),
            "{mode:?} {mle:?}"
        );
    }

    #[test]
    fn guessing_on_its_bound_leaves_the_two_parameter_optimum() {
        // A lower asymptote below zero puts the unconstrained guessing optimum
        // outside the box, so the bounded optimum is the 2PL fit with c = 0.
        let (points, r_k, n_k) =
            counts(|t| (-0.04 + 1.04 * sigmoid(1.1 * (t + 0.3))).clamp(1e-6, 1.0));
        let [a, b, c] =
            newton_3pl_item(&r_k, &n_k, &points, [1.0, 0.0, 0.2], &CONTROLS, &NO_PRIORS);

        let [a_2pl, b_2pl] = newton_2pl_item(&r_k, &n_k, &points, [1.0, 0.0], &TWO_PARAMETER);

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
        let [a, b, c] = newton_3pl_item(
            &r_k,
            &n_k,
            &points,
            [0.3, 3.3, 0.3495],
            &CONTROLS,
            &NO_PRIORS,
        );

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
        let [a, b, c] =
            newton_3pl_item(&r_k, &n_k, &points, [1.5, 2.5, 3e-4], &CONTROLS, &NO_PRIORS);

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
            params = newton_3pl_item(&r_k, &n_k, &points, params, &one_step, &NO_PRIORS);
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
            (params, information) =
                mhrm_2pl_item(&view, &theta, 0, params, information, 1.0, WIDE_BOX);
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

    #[test]
    fn unit_gain_steps_stay_in_the_given_box() {
        let (theta, responses) = grid_responses(1.4, 0.3);
        let view = responses.view();
        let narrow = ItemBox {
            lower: [0.1, -0.2],
            upper: [0.9, 0.2],
        };
        let mut params = [0.5, 0.0];
        let mut information = None;
        for _ in 0..60 {
            (params, information) =
                mhrm_2pl_item(&view, &theta, 0, params, information, 1.0, narrow);
            assert!((0..2).all(|k| narrow.lower[k] <= params[k] && params[k] <= narrow.upper[k]));
        }
        // The complete-data optimum lies above both upper bounds.
        assert_eq!(params, [0.9, 0.2]);
    }
}
