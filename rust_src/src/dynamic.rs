//! Dynamic and longitudinal IRT models.
//!
//! This module provides Rust implementations for:
//! - Bayesian Knowledge Tracing (BKT) forward-backward algorithm
//! - Forward-filtering backward-sampling (FFBS)
//! - Longitudinal IRT likelihood computations

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::{prelude::*, rngs::StdRng};
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

    let (alpha, scaling) = forward_single(
        responses.as_slice().unwrap(),
        skills.as_slice().unwrap(),
        p_init.as_slice().unwrap(),
        p_learn.as_slice().unwrap(),
        p_forget.as_slice().unwrap(),
        p_slip.as_slice().unwrap(),
        p_guess.as_slice().unwrap(),
    );

    (
        alpha.to_pyarray(py),
        Array1::from_vec(scaling).to_pyarray(py),
    )
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

    backward_single(
        responses.as_slice().unwrap(),
        skills.as_slice().unwrap(),
        scaling.as_slice().unwrap(),
        p_learn.as_slice().unwrap(),
        p_forget.as_slice().unwrap(),
        p_slip.as_slice().unwrap(),
        p_guess.as_slice().unwrap(),
    )
    .to_pyarray(py)
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
    let mut previous_trial = vec![None; p_init.len()];

    for t in 0..n_trials {
        let skill_idx = skills[t] as usize;
        if let Some(previous) = previous_trial[skill_idx] {
            let p_l = p_learn[skill_idx];
            let p_f = p_forget[skill_idx];
            for state in 0..2 {
                let mut predicted = 0.0;
                for previous_state in 0..2 {
                    predicted += alpha[[previous, previous_state]]
                        * transition_prob(previous_state, state, p_l, p_f);
                }
                alpha[[t, state]] = predicted
                    * compute_emission_slice(responses[t], state, skill_idx, p_slip, p_guess);
            }
        } else {
            let p_0 = p_init[skill_idx];
            for state in 0..2 {
                let prior = if state == 1 { p_0 } else { 1.0 - p_0 };
                alpha[[t, state]] =
                    prior * compute_emission_slice(responses[t], state, skill_idx, p_slip, p_guess);
            }
        }

        scaling[t] = alpha[[t, 0]] + alpha[[t, 1]];
        if scaling[t] > EPSILON {
            alpha[[t, 0]] /= scaling[t];
            alpha[[t, 1]] /= scaling[t];
        }
        previous_trial[skill_idx] = Some(t);
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
    let mut next_trial = vec![None; p_learn.len()];

    for t in (0..n_trials).rev() {
        let skill_idx = skills[t] as usize;
        if let Some(next) = next_trial[skill_idx] {
            let p_l = p_learn[skill_idx];
            let p_f = p_forget[skill_idx];
            for state in 0..2 {
                let mut smoothed = 0.0;
                for next_state in 0..2 {
                    smoothed += transition_prob(state, next_state, p_l, p_f)
                        * compute_emission_slice(
                            responses[next],
                            next_state,
                            skill_idx,
                            p_slip,
                            p_guess,
                        )
                        * beta[[next, next_state]];
                }
                beta[[t, state]] = smoothed;
            }
            if scaling[next] > EPSILON {
                beta[[t, 0]] /= scaling[next];
                beta[[t, 1]] /= scaling[next];
            }
        } else {
            beta[[t, 0]] = 1.0;
            beta[[t, 1]] = 1.0;
        }
        next_trial[skill_idx] = Some(t);
    }

    beta
}

fn viterbi_single(
    responses: &[i32],
    skills: &[i32],
    p_init: &[f64],
    p_learn: &[f64],
    p_forget: &[f64],
    p_slip: &[f64],
    p_guess: &[f64],
) -> Vec<i32> {
    let n_trials = responses.len();
    let mut delta = Array2::zeros((n_trials, 2));
    let mut psi = Array2::<usize>::zeros((n_trials, 2));
    let mut previous_trial = vec![None; p_init.len()];
    let mut previous_for_trial = vec![None; n_trials];
    let mut last_trial = vec![None; p_init.len()];

    for t in 0..n_trials {
        let skill_idx = skills[t] as usize;
        if let Some(previous) = previous_trial[skill_idx] {
            for state in 0..2 {
                let mut best_value = f64::NEG_INFINITY;
                let mut best_previous = 0;
                for previous_state in 0..2 {
                    let transition = (transition_prob(
                        previous_state,
                        state,
                        p_learn[skill_idx],
                        p_forget[skill_idx],
                    ) + EPSILON)
                        .ln();
                    let value = delta[[previous, previous_state]] + transition;
                    if value > best_value {
                        best_value = value;
                        best_previous = previous_state;
                    }
                }
                psi[[t, state]] = best_previous;
                delta[[t, state]] = best_value;
            }
        } else {
            let p_0 = p_init[skill_idx];
            delta[[t, 0]] = (1.0 - p_0 + EPSILON).ln();
            delta[[t, 1]] = (p_0 + EPSILON).ln();
        }

        for state in 0..2 {
            delta[[t, state]] +=
                (compute_emission_slice(responses[t], state, skill_idx, p_slip, p_guess) + EPSILON)
                    .ln();
        }
        previous_for_trial[t] = previous_trial[skill_idx];
        previous_trial[skill_idx] = Some(t);
        last_trial[skill_idx] = Some(t);
    }

    let mut path = vec![0i32; n_trials];
    for last in last_trial.into_iter().flatten() {
        path[last] = if delta[[last, 1]] > delta[[last, 0]] {
            1
        } else {
            0
        };
        let mut current = last;
        while let Some(previous) = previous_for_trial[current] {
            path[previous] = psi[[current, path[current] as usize]] as i32;
            current = previous;
        }
    }
    path
}

#[allow(clippy::too_many_arguments)]
fn ffbs_single<R: Rng + ?Sized>(
    responses: &[i32],
    skills: &[i32],
    p_init: &[f64],
    p_learn: &[f64],
    p_forget: &[f64],
    p_slip: &[f64],
    p_guess: &[f64],
    rng: &mut R,
) -> Vec<i32> {
    let (alpha, _) = forward_single(
        responses, skills, p_init, p_learn, p_forget, p_slip, p_guess,
    );
    let n_trials = responses.len();
    let mut states = vec![0i32; n_trials];
    let mut next_trial = vec![None; p_init.len()];

    for t in (0..n_trials).rev() {
        let skill_idx = skills[t] as usize;
        let learned_probability = if let Some(next) = next_trial[skill_idx] {
            let next_state = states[next] as usize;
            let mut state_probability = [0.0; 2];
            for state in 0..2 {
                state_probability[state] = alpha[[t, state]]
                    * transition_prob(state, next_state, p_learn[skill_idx], p_forget[skill_idx]);
            }
            let total = state_probability[0] + state_probability[1];
            if total > EPSILON {
                state_probability[1] / total
            } else {
                alpha[[t, 1]]
            }
        } else {
            alpha[[t, 1]]
        };
        states[t] = if rng.random::<f64>() < learned_probability {
            1
        } else {
            0
        };
        next_trial[skill_idx] = Some(t);
    }
    states
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

    Array1::from_vec(viterbi_single(
        responses.as_slice().unwrap(),
        skills.as_slice().unwrap(),
        p_init.as_slice().unwrap(),
        p_learn.as_slice().unwrap(),
        p_forget.as_slice().unwrap(),
        p_slip.as_slice().unwrap(),
        p_guess.as_slice().unwrap(),
    ))
    .to_pyarray(py)
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
    let responses = responses.as_array();
    let skills = skill_assignments.as_array();
    let p_init = p_init.as_array();
    let p_learn = p_learn.as_array();
    let p_forget = p_forget.as_array();
    let p_slip = p_slip.as_array();
    let p_guess = p_guess.as_array();

    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_vec(ffbs_single(
        responses.as_slice().unwrap(),
        skills.as_slice().unwrap(),
        p_init.as_slice().unwrap(),
        p_learn.as_slice().unwrap(),
        p_forget.as_slice().unwrap(),
        p_slip.as_slice().unwrap(),
        p_guess.as_slice().unwrap(),
        &mut rng,
    ))
    .to_pyarray(py)
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
            let mut rng = StdRng::seed_from_u64(seed + i as u64);

            let person_responses: Vec<i32> = (0..n_trials).map(|t| responses[[i, t]]).collect();

            ffbs_single(
                &person_responses,
                skills.as_slice().unwrap(),
                p_init.as_slice().unwrap(),
                p_learn.as_slice().unwrap(),
                p_forget.as_slice().unwrap(),
                p_slip.as_slice().unwrap(),
                p_guess.as_slice().unwrap(),
                &mut rng,
            )
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
