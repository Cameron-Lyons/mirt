//! Dynamic and longitudinal IRT models.
//!
//! This module provides Rust implementations for:
//! - Bayesian Knowledge Tracing (BKT) forward-backward algorithm
//! - Forward-filtering backward-sampling (FFBS)
//! - Longitudinal IRT likelihood computations

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// BKT forward algorithm for a single person.
///
/// # Arguments
/// * `responses` - Response sequence (n_trials,)
/// * `skill_assignments` - Skill index for each trial (n_trials,)
/// * `p_init` - Initial knowledge probability per skill (n_skills,)
/// * `p_learn` - Learning probability per skill (n_skills,)
/// * `p_forget` - Forgetting probability per skill (n_skills,)
/// * `p_slip` - Slip probability per skill (n_skills,)
/// * `p_guess` - Guess probability per skill (n_skills,)
///
/// # Returns
/// (alpha, scaling) where alpha[t, s] = P(L_t = s | X_1:t)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_forward<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    p_init: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_trials = responses.len();
    let mut alpha = Array2::zeros((n_trials, 2));
    let mut scaling = Array1::zeros(n_trials);

    let skill_idx = skills[0] as usize;
    let p_0 = p_init[skill_idx];

    for s in 0..2 {
        let prior = if s == 1 { p_0 } else { 1.0 - p_0 };
        let emission = compute_emission(responses[0], s, skill_idx, &p_slip, &p_guess);
        alpha[[0, s]] = prior * emission;
    }

    scaling[0] = alpha[[0, 0]] + alpha[[0, 1]];
    if scaling[0] > EPSILON {
        alpha[[0, 0]] /= scaling[0];
        alpha[[0, 1]] /= scaling[0];
    }

    for t in 1..n_trials {
        let skill_idx = skills[t] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        for s in 0..2 {
            let mut sum = 0.0;
            for s_prev in 0..2 {
                let trans = transition_prob(s_prev, s, p_l, p_f);
                sum += alpha[[t - 1, s_prev]] * trans;
            }

            let emission = compute_emission(responses[t], s, skill_idx, &p_slip, &p_guess);
            alpha[[t, s]] = sum * emission;
        }

        scaling[t] = alpha[[t, 0]] + alpha[[t, 1]];
        if scaling[t] > EPSILON {
            alpha[[t, 0]] /= scaling[t];
            alpha[[t, 1]] /= scaling[t];
        }
    }

    (alpha.to_pyarray(py), scaling.to_pyarray(py))
}

/// BKT backward algorithm for a single person.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_backward<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    scaling: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let scaling = scaling.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_trials = responses.len();
    let mut beta = Array2::zeros((n_trials, 2));

    beta[[n_trials - 1, 0]] = 1.0;
    beta[[n_trials - 1, 1]] = 1.0;

    for t in (0..n_trials - 1).rev() {
        let skill_idx = skills[t + 1] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        for s in 0..2 {
            let mut sum = 0.0;
            for s_next in 0..2 {
                let trans = transition_prob(s, s_next, p_l, p_f);
                let emission =
                    compute_emission(responses[t + 1], s_next, skill_idx, &p_slip, &p_guess);
                sum += trans * emission * beta[[t + 1, s_next]];
            }
            beta[[t, s]] = sum;
        }

        if scaling[t + 1] > EPSILON {
            beta[[t, 0]] /= scaling[t + 1];
            beta[[t, 1]] /= scaling[t + 1];
        }
    }

    beta.to_pyarray(py)
}

/// BKT forward-backward for multiple persons in parallel.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_forward_backward_batch<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    p_init: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_persons = responses.nrows();
    let n_trials = responses.ncols();

    let results: Vec<(Array2<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let person_responses: Vec<i32> = (0..n_trials).map(|t| responses[[i, t]]).collect();

            let (alpha, scaling) = forward_single(
                &person_responses,
                skills.as_slice().unwrap(),
                p_init.as_slice().unwrap(),
                p_learn.as_slice().unwrap(),
                p_forget.as_slice().unwrap(),
                p_slip.as_slice().unwrap(),
                p_guess.as_slice().unwrap(),
            );

            let beta = backward_single(
                &person_responses,
                skills.as_slice().unwrap(),
                &scaling,
                p_learn.as_slice().unwrap(),
                p_forget.as_slice().unwrap(),
                p_slip.as_slice().unwrap(),
                p_guess.as_slice().unwrap(),
            );

            let mut gamma = Array2::zeros((n_trials, 2));
            for t in 0..n_trials {
                gamma[[t, 0]] = alpha[[t, 0]] * beta[[t, 0]];
                gamma[[t, 1]] = alpha[[t, 1]] * beta[[t, 1]];
                let sum = gamma[[t, 0]] + gamma[[t, 1]];
                if sum > EPSILON {
                    gamma[[t, 0]] /= sum;
                    gamma[[t, 1]] /= sum;
                }
            }

            let log_likelihood: f64 = scaling.iter().map(|s| (s + EPSILON).ln()).sum();

            (gamma, log_likelihood)
        })
        .collect();

    let mut gamma_out = Array2::zeros((n_persons, n_trials));
    let mut ll_out = Array1::zeros(n_persons);

    for (i, (gamma, ll)) in results.into_iter().enumerate() {
        for t in 0..n_trials {
            gamma_out[[i, t]] = gamma[[t, 1]];
        }
        ll_out[i] = ll;
    }

    (gamma_out.to_pyarray(py), ll_out.to_pyarray(py))
}

fn forward_single(
    responses: &[i32],
    skills: &[i32],
    p_init: &[f64],
    p_learn: &[f64],
    p_forget: &[f64],
    p_slip: &[f64],
    p_guess: &[f64],
) -> (Array2<f64>, Vec<f64>) {
    let n_trials = responses.len();
    let mut alpha = Array2::zeros((n_trials, 2));
    let mut scaling = vec![0.0; n_trials];

    let skill_idx = skills[0] as usize;
    let p_0 = p_init[skill_idx];

    for s in 0..2 {
        let prior = if s == 1 { p_0 } else { 1.0 - p_0 };
        let emission = compute_emission_slice(responses[0], s, skill_idx, p_slip, p_guess);
        alpha[[0, s]] = prior * emission;
    }

    scaling[0] = alpha[[0, 0]] + alpha[[0, 1]];
    if scaling[0] > EPSILON {
        alpha[[0, 0]] /= scaling[0];
        alpha[[0, 1]] /= scaling[0];
    }

    for t in 1..n_trials {
        let skill_idx = skills[t] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        for s in 0..2 {
            let mut sum = 0.0;
            for s_prev in 0..2 {
                let trans = transition_prob(s_prev, s, p_l, p_f);
                sum += alpha[[t - 1, s_prev]] * trans;
            }

            let emission = compute_emission_slice(responses[t], s, skill_idx, p_slip, p_guess);
            alpha[[t, s]] = sum * emission;
        }

        scaling[t] = alpha[[t, 0]] + alpha[[t, 1]];
        if scaling[t] > EPSILON {
            alpha[[t, 0]] /= scaling[t];
            alpha[[t, 1]] /= scaling[t];
        }
    }

    (alpha, scaling)
}

fn backward_single(
    responses: &[i32],
    skills: &[i32],
    scaling: &[f64],
    p_learn: &[f64],
    p_forget: &[f64],
    p_slip: &[f64],
    p_guess: &[f64],
) -> Array2<f64> {
    let n_trials = responses.len();
    let mut beta = Array2::zeros((n_trials, 2));

    beta[[n_trials - 1, 0]] = 1.0;
    beta[[n_trials - 1, 1]] = 1.0;

    for t in (0..n_trials - 1).rev() {
        let skill_idx = skills[t + 1] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        for s in 0..2 {
            let mut sum = 0.0;
            for s_next in 0..2 {
                let trans = transition_prob(s, s_next, p_l, p_f);
                let emission =
                    compute_emission_slice(responses[t + 1], s_next, skill_idx, p_slip, p_guess);
                sum += trans * emission * beta[[t + 1, s_next]];
            }
            beta[[t, s]] = sum;
        }

        if scaling[t + 1] > EPSILON {
            beta[[t, 0]] /= scaling[t + 1];
            beta[[t, 1]] /= scaling[t + 1];
        }
    }

    beta
}

#[inline]
fn transition_prob(from: usize, to: usize, p_learn: f64, p_forget: f64) -> f64 {
    match (from, to) {
        (0, 0) => 1.0 - p_learn,
        (0, 1) => p_learn,
        (1, 0) => p_forget,
        (1, 1) => 1.0 - p_forget,
        _ => 0.0,
    }
}

#[inline]
fn compute_emission(
    response: i32,
    learned: usize,
    skill_idx: usize,
    p_slip: &ndarray::ArrayView1<f64>,
    p_guess: &ndarray::ArrayView1<f64>,
) -> f64 {
    if response < 0 {
        return 1.0;
    }

    if learned == 1 {
        if response == 1 {
            1.0 - p_slip[skill_idx]
        } else {
            p_slip[skill_idx]
        }
    } else if response == 1 {
        p_guess[skill_idx]
    } else {
        1.0 - p_guess[skill_idx]
    }
}

#[inline]
fn compute_emission_slice(
    response: i32,
    learned: usize,
    skill_idx: usize,
    p_slip: &[f64],
    p_guess: &[f64],
) -> f64 {
    if response < 0 {
        return 1.0;
    }

    if learned == 1 {
        if response == 1 {
            1.0 - p_slip[skill_idx]
        } else {
            p_slip[skill_idx]
        }
    } else if response == 1 {
        p_guess[skill_idx]
    } else {
        1.0 - p_guess[skill_idx]
    }
}

/// Viterbi algorithm for finding most likely state sequence.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_viterbi<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    p_init: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<i32>> {
    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_trials = responses.len();
    let mut delta = Array2::zeros((n_trials, 2));
    let mut psi = Array2::<usize>::zeros((n_trials, 2));

    let skill_idx = skills[0] as usize;
    let p_0 = p_init[skill_idx];

    for s in 0..2 {
        let prior = if s == 1 {
            (p_0 + EPSILON).ln()
        } else {
            (1.0 - p_0 + EPSILON).ln()
        };
        let emission =
            (compute_emission(responses[0], s, skill_idx, &p_slip, &p_guess) + EPSILON).ln();
        delta[[0, s]] = prior + emission;
    }

    for t in 1..n_trials {
        let skill_idx = skills[t] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        for s in 0..2 {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_prev = 0;

            for s_prev in 0..2 {
                let trans = (transition_prob(s_prev, s, p_l, p_f) + EPSILON).ln();
                let val = delta[[t - 1, s_prev]] + trans;
                if val > best_val {
                    best_val = val;
                    best_prev = s_prev;
                }
            }

            psi[[t, s]] = best_prev;
            let emission =
                (compute_emission(responses[t], s, skill_idx, &p_slip, &p_guess) + EPSILON).ln();
            delta[[t, s]] = best_val + emission;
        }
    }

    let mut path = Array1::<i32>::zeros(n_trials);
    path[n_trials - 1] = if delta[[n_trials - 1, 1]] > delta[[n_trials - 1, 0]] {
        1
    } else {
        0
    };

    for t in (0..n_trials - 1).rev() {
        path[t] = psi[[t + 1, path[t + 1] as usize]] as i32;
    }

    path.to_pyarray(py)
}

/// Compute log-likelihood for longitudinal IRT data.
#[pyfunction]
pub fn longitudinal_log_likelihood(
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> f64 {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let disc = discrimination.as_array();
    let diff = difficulty.as_array();

    let n_obs = responses.nrows();
    let n_items = responses.ncols();

    let ll: f64 = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut ll_i = 0.0;
            for j in 0..n_items {
                let resp = responses[[i, j]];
                if resp >= 0 {
                    let z = disc[j] * (theta[i] - diff[j]);
                    let p = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);

                    if resp == 1 {
                        ll_i += p.ln();
                    } else {
                        ll_i += (1.0 - p).ln();
                    }
                }
            }
            ll_i
        })
        .sum();

    ll
}

/// Compute growth curve predictions.
#[pyfunction]
pub fn compute_growth_trajectory<'py>(
    py: Python<'py>,
    growth_factors: PyReadonlyArray2<f64>,
    time_values: PyReadonlyArray1<f64>,
    growth_model: &str,
) -> Bound<'py, PyArray2<f64>> {
    let factors = growth_factors.as_array();
    let times = time_values.as_array();

    let n_persons = factors.nrows();
    let n_times = times.len();

    let trajectories: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let intercept = factors[[i, 0]];
            let slope = factors[[i, 1]];

            let mut traj = Vec::with_capacity(n_times);

            for t in 0..n_times {
                let mut theta = intercept + slope * times[t];

                if growth_model == "quadratic" && factors.ncols() > 2 {
                    let quad = factors[[i, 2]];
                    theta += quad * times[t] * times[t];
                }

                traj.push(theta);
            }

            traj
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_times));
    for (i, traj) in trajectories.into_iter().enumerate() {
        for (t, val) in traj.into_iter().enumerate() {
            result[[i, t]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Forward-filtering backward-sampling (FFBS) for BKT.
/// Returns sampled state sequence.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_ffbs<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    p_init: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
    seed: u64,
) -> Bound<'py, PyArray1<i32>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Uniform};

    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_trials = responses.len();

    let (alpha, _scaling) = forward_single(
        responses.as_slice().unwrap(),
        skills.as_slice().unwrap(),
        p_init.as_slice().unwrap(),
        p_learn.as_slice().unwrap(),
        p_forget.as_slice().unwrap(),
        p_slip.as_slice().unwrap(),
        p_guess.as_slice().unwrap(),
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0f64, 1.0).unwrap();

    let mut states = Array1::<i32>::zeros(n_trials);

    let p_learned = alpha[[n_trials - 1, 1]];
    states[n_trials - 1] = if uniform.sample(&mut rng) < p_learned {
        1
    } else {
        0
    };

    for t in (0..n_trials - 1).rev() {
        let skill_idx = skills[t + 1] as usize;
        let p_l = p_learn[skill_idx];
        let p_f = p_forget[skill_idx];

        let next_state = states[t + 1] as usize;

        let mut p_state = [0.0; 2];
        for s in 0..2 {
            let trans = transition_prob(s, next_state, p_l, p_f);
            p_state[s] = alpha[[t, s]] * trans;
        }

        let sum = p_state[0] + p_state[1];
        if sum > EPSILON {
            p_state[0] /= sum;
            p_state[1] /= sum;
        }

        states[t] = if uniform.sample(&mut rng) < p_state[1] {
            1
        } else {
            0
        };
    }

    states.to_pyarray(py)
}

/// Batch FFBS for multiple persons.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bkt_ffbs_batch<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    skill_assignments: PyReadonlyArray1<i32>,
    p_init: PyReadonlyArray1<f64>,
    p_learn: PyReadonlyArray1<f64>,
    p_forget: PyReadonlyArray1<f64>,
    p_slip: PyReadonlyArray1<f64>,
    p_guess: PyReadonlyArray1<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Uniform};

    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let n_persons = responses.nrows();
    let n_trials = responses.ncols();

    let results: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed + i as u64);
            let uniform = Uniform::new(0.0f64, 1.0).unwrap();

            let person_responses: Vec<i32> = (0..n_trials).map(|t| responses[[i, t]]).collect();

            let (alpha, _scaling) = forward_single(
                &person_responses,
                skills.as_slice().unwrap(),
                p_init.as_slice().unwrap(),
                p_learn.as_slice().unwrap(),
                p_forget.as_slice().unwrap(),
                p_slip.as_slice().unwrap(),
                p_guess.as_slice().unwrap(),
            );

            let mut states = vec![0i32; n_trials];

            let p_learned = alpha[[n_trials - 1, 1]];
            states[n_trials - 1] = if uniform.sample(&mut rng) < p_learned {
                1
            } else {
                0
            };

            for t in (0..n_trials - 1).rev() {
                let skill_idx = skills[t + 1] as usize;
                let p_l = p_learn[skill_idx];
                let p_f = p_forget[skill_idx];

                let next_state = states[t + 1] as usize;

                let mut p_state = [0.0; 2];
                for s in 0..2 {
                    let trans = transition_prob(s, next_state, p_l, p_f);
                    p_state[s] = alpha[[t, s]] * trans;
                }

                let sum = p_state[0] + p_state[1];
                if sum > EPSILON {
                    p_state[0] /= sum;
                    p_state[1] /= sum;
                }

                states[t] = if uniform.sample(&mut rng) < p_state[1] {
                    1
                } else {
                    0
                };
            }

            states
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_trials));
    for (i, states) in results.into_iter().enumerate() {
        for (t, s) in states.into_iter().enumerate() {
            result[[i, t]] = s;
        }
    }

    result.to_pyarray(py)
}

/// Register dynamic model functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bkt_forward, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_backward, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_forward_backward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_viterbi, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_ffbs, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_ffbs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(longitudinal_log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(compute_growth_trajectory, m)?)?;
    Ok(())
}
