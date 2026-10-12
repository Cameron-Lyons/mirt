//! Bayesian Knowledge Tracing (BKT) filtering, smoothing and Viterbi decoding.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::EPSILON;

const LOG_FLOOR: f64 = 1e-300;

/// Posterior state probabilities with per-trial scaling or per-person
/// log-likelihoods.
type PyStateResult<'py> = PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>;

/// Skill parameters of one call, one entry per skill.
struct SkillParameters {
    p_init: Vec<f64>,
    p_learn: Vec<f64>,
    p_forget: Vec<f64>,
    p_slip: Vec<f64>,
    p_guess: Vec<f64>,
}

impl SkillParameters {
    /// Copy the parameters, rejecting vectors of different lengths.
    fn new(
        p_init: Option<PyReadonlyArray1<f64>>,
        p_learn: PyReadonlyArray1<f64>,
        p_forget: PyReadonlyArray1<f64>,
        p_slip: PyReadonlyArray1<f64>,
        p_guess: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let p_learn = p_learn.as_array().to_vec();
        let n_skills = p_learn.len();
        let parameters = Self {
            p_init: p_init.map_or_else(|| vec![0.0; n_skills], |v| v.as_array().to_vec()),
            p_forget: p_forget.as_array().to_vec(),
            p_slip: p_slip.as_array().to_vec(),
            p_guess: p_guess.as_array().to_vec(),
            p_learn,
        };
        let lengths = [
            parameters.p_init.len(),
            parameters.p_forget.len(),
            parameters.p_slip.len(),
            parameters.p_guess.len(),
        ];
        if lengths.iter().any(|&length| length != n_skills) {
            return Err(PyValueError::new_err(
                "skill parameters must have one entry per skill",
            ));
        }
        Ok(parameters)
    }

    /// Copy the skill index of every trial, rejecting a count other than
    /// `n_trials` and indices outside `[0, n_skills)`.
    fn skills(&self, skills: PyReadonlyArray1<i32>, n_trials: usize) -> PyResult<Vec<i32>> {
        let skills = skills.as_array().to_vec();
        if skills.len() != n_trials {
            return Err(PyValueError::new_err(format!(
                "skill_assignments must have shape ({n_trials},), got ({},)",
                skills.len()
            )));
        }
        let n_skills = self.p_learn.len();
        if skills
            .iter()
            .any(|&skill| skill < 0 || skill as usize >= n_skills)
        {
            return Err(PyValueError::new_err(format!(
                "skill indices must lie in [0, {n_skills})"
            )));
        }
        Ok(skills)
    }
}

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
) -> PyStateResult<'py> {
    let responses = responses.as_array().to_vec();
    let parameters = SkillParameters::new(Some(p_init), p_learn, p_forget, p_slip, p_guess)?;
    let skills = parameters.skills(skill_assignments, responses.len())?;

    let (alpha, scaling) = forward_single(&responses, &skills, &parameters);

    Ok((
        alpha.to_pyarray(py),
        Array1::from_vec(scaling).to_pyarray(py),
    ))
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array().to_vec();
    let scaling = scaling.as_array().to_vec();
    let parameters = SkillParameters::new(None, p_learn, p_forget, p_slip, p_guess)?;
    let skills = parameters.skills(skill_assignments, responses.len())?;
    if scaling.len() != responses.len() {
        return Err(PyValueError::new_err(format!(
            "scaling must have shape ({},), got ({},)",
            responses.len(),
            scaling.len()
        )));
    }

    Ok(backward_single(&responses, &skills, &scaling, &parameters).to_pyarray(py))
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
) -> PyStateResult<'py> {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_trials = responses.ncols();
    let parameters = SkillParameters::new(Some(p_init), p_learn, p_forget, p_slip, p_guess)?;
    let skills = parameters.skills(skill_assignments, n_trials)?;

    let results: Vec<(Array2<f64>, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let person_responses: Vec<i32> = responses.row(i).to_vec();

            let (alpha, scaling) = forward_single(&person_responses, &skills, &parameters);
            let beta = backward_single(&person_responses, &skills, &scaling, &parameters);

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

            let log_likelihood: f64 = scaling.iter().map(|s| s.max(LOG_FLOOR).ln()).sum();

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

    Ok((gamma_out.to_pyarray(py), ll_out.to_pyarray(py)))
}

fn forward_single(
    responses: &[i32],
    skills: &[i32],
    parameters: &SkillParameters,
) -> (Array2<f64>, Vec<f64>) {
    let SkillParameters {
        p_init,
        p_learn,
        p_forget,
        p_slip,
        p_guess,
    } = parameters;
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
    parameters: &SkillParameters,
) -> Array2<f64> {
    let SkillParameters {
        p_learn,
        p_forget,
        p_slip,
        p_guess,
        ..
    } = parameters;
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

fn viterbi_single(responses: &[i32], skills: &[i32], parameters: &SkillParameters) -> Vec<i32> {
    let SkillParameters {
        p_init,
        p_learn,
        p_forget,
        p_slip,
        p_guess,
    } = parameters;
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
                    let transition = transition_prob(
                        previous_state,
                        state,
                        p_learn[skill_idx],
                        p_forget[skill_idx],
                    )
                    .max(LOG_FLOOR)
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
            delta[[t, 0]] = (1.0 - p_0).max(LOG_FLOOR).ln();
            delta[[t, 1]] = p_0.max(LOG_FLOOR).ln();
        }

        for state in 0..2 {
            delta[[t, state]] +=
                compute_emission_slice(responses[t], state, skill_idx, p_slip, p_guess)
                    .max(LOG_FLOOR)
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
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let responses = responses.as_array().to_vec();
    let parameters = SkillParameters::new(Some(p_init), p_learn, p_forget, p_slip, p_guess)?;
    let skills = parameters.skills(skill_assignments, responses.len())?;

    Ok(Array1::from_vec(viterbi_single(&responses, &skills, &parameters)).to_pyarray(py))
}

/// Register dynamic model functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bkt_forward, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_backward, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_forward_backward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(bkt_viterbi, m)?)?;
    Ok(())
}
