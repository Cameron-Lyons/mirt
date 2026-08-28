//! Parallel M-step optimization for various IRT models.
//!
//! This module provides parallelized M-step computation using Rayon,
//! enabling efficient parameter estimation for large-scale IRT analysis.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, log_sigmoid, sigmoid};

const MAX_LINE_SEARCH_STEPS: usize = 24;

#[derive(Clone, Copy)]
struct DichotomousMstepConfig {
    max_iter: usize,
    tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    damping: f64,
    regularization: f64,
}

fn expected_log_likelihood_2pl(
    correct_counts: &[f64],
    total_counts: &[f64],
    quad_points: &[f64],
    discrimination: f64,
    difficulty: f64,
) -> f64 {
    correct_counts
        .iter()
        .zip(total_counts)
        .zip(quad_points)
        .map(|((&correct, &total), &theta)| {
            let logit = discrimination * (theta - difficulty);
            correct * log_sigmoid(logit) + (total - correct) * log_sigmoid(-logit)
        })
        .sum()
}

fn optimize_dichotomous_item(
    correct_counts: &[f64],
    total_counts: &[f64],
    quad_points: &[f64],
    initial_discrimination: f64,
    initial_difficulty: f64,
    config: DichotomousMstepConfig,
) -> (f64, f64) {
    if total_counts.iter().sum::<f64>() <= EPSILON {
        return (initial_discrimination, initial_difficulty);
    }

    let mut discrimination = initial_discrimination;
    let mut difficulty = initial_difficulty;

    for _ in 0..config.max_iter {
        let intercept = -discrimination * difficulty;
        let mut gradient_slope = 0.0;
        let mut gradient_intercept = 0.0;
        let mut hessian_slope = -config.regularization;
        let mut hessian_intercept = -config.regularization;
        let mut hessian_cross = 0.0;

        for ((&correct, &total), &theta) in correct_counts.iter().zip(total_counts).zip(quad_points)
        {
            let probability = sigmoid(discrimination * theta + intercept);
            let residual = correct - total * probability;
            let information = total * probability * (1.0 - probability);

            gradient_slope += residual * theta;
            gradient_intercept += residual;
            hessian_slope -= information * theta * theta;
            hessian_intercept -= information;
            hessian_cross -= information * theta;
        }

        let determinant = hessian_slope * hessian_intercept - hessian_cross * hessian_cross;
        if determinant.abs() < EPSILON {
            break;
        }

        let delta_slope =
            (hessian_intercept * gradient_slope - hessian_cross * gradient_intercept) / determinant;
        let delta_intercept =
            (-hessian_cross * gradient_slope + hessian_slope * gradient_intercept) / determinant;
        let current_likelihood = expected_log_likelihood_2pl(
            correct_counts,
            total_counts,
            quad_points,
            discrimination,
            difficulty,
        );
        let likelihood_tolerance = 1e-12 * current_likelihood.abs().max(1.0);

        let mut step = config.damping;
        let mut accepted = None;
        for _ in 0..MAX_LINE_SEARCH_STEPS {
            let candidate_discrimination = (discrimination - step * delta_slope)
                .clamp(config.disc_bounds.0, config.disc_bounds.1);
            let candidate_intercept = intercept - step * delta_intercept;
            let candidate_difficulty = (-candidate_intercept / candidate_discrimination)
                .clamp(config.diff_bounds.0, config.diff_bounds.1);
            let candidate_likelihood = expected_log_likelihood_2pl(
                correct_counts,
                total_counts,
                quad_points,
                candidate_discrimination,
                candidate_difficulty,
            );
            if candidate_likelihood >= current_likelihood - likelihood_tolerance {
                accepted = Some((candidate_discrimination, candidate_difficulty));
                break;
            }
            step *= 0.5;
        }

        let Some((candidate_discrimination, candidate_difficulty)) = accepted else {
            break;
        };
        let parameter_change = (candidate_discrimination - discrimination)
            .abs()
            .max((candidate_difficulty - difficulty).abs());
        discrimination = candidate_discrimination;
        difficulty = candidate_difficulty;
        if parameter_change < config.tol {
            break;
        }
    }

    (discrimination, difficulty)
}

/// Parallel M-step optimization for independent 2PL items.
///
/// Uses exact expected-likelihood curvature in slope-intercept space and
/// backtracking to prevent accepted updates from reducing the objective.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights, quad_points, discrimination, difficulty, max_iter, tol, disc_bounds, diff_bounds, damping, regularization))]
#[allow(clippy::too_many_arguments)]
pub fn m_step_dichotomous_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
    disc_bounds: (f64, f64),
    diff_bounds: (f64, f64),
    damping: f64,
    regularization: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc_init = discrimination.as_array();
    let diff_init = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();
    let quad_points_vec = quad_points.to_vec();
    let config = DichotomousMstepConfig {
        max_iter,
        tol,
        disc_bounds,
        diff_bounds,
        damping,
        regularization,
    };

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
                    let w = posterior_weights[[i, q]];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            optimize_dichotomous_item(
                &r_k,
                &n_k,
                &quad_points_vec,
                disc_init[j],
                diff_init[j],
                config,
            )
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

    (disc_new.to_pyarray(py), diff_new.to_pyarray(py))
}

/// Parallel M-step for 3PL model including guessing parameter.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights, quad_points, discrimination, difficulty, guessing, max_iter, tol, disc_bounds, diff_bounds, guess_bounds, damping_ab, damping_c, regularization, regularization_c))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn m_step_3pl_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
    max_iter: usize,
    tol: f64,
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
) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let disc_init = discrimination.as_array();
    let diff_init = difficulty.as_array();
    let guess_init = guessing.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

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
                    let w = posterior_weights[[i, q]];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            let mut a = disc_init[j];
            let mut b = diff_init[j];
            let mut c = guess_init[j];

            for _ in 0..max_iter {
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

                if grad_a.abs() < tol && grad_b.abs() < tol && grad_c.abs() < tol {
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

    (
        disc_new.to_pyarray(py),
        diff_new.to_pyarray(py),
        guess_new.to_pyarray(py),
    )
}

/// Compute expected counts for dichotomous items in parallel.
///
/// Returns r_k (correct responses) and n_k (total responses) per quadrature point.
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights))]
pub fn compute_expected_counts_parallel<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = posterior_weights.ncols();

    let counts: Vec<(Vec<f64>, Vec<f64>)> = (0..n_items)
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
                    let w = posterior_weights[[i, q]];
                    n_k[q] += w;
                    if resp == 1 {
                        r_k[q] += w;
                    }
                }
            }

            (r_k, n_k)
        })
        .collect();

    let mut r_k_all = numpy::ndarray::Array2::zeros((n_items, n_quad));
    let mut n_k_all = numpy::ndarray::Array2::zeros((n_items, n_quad));

    for (j, (r_k, n_k)) in counts.into_iter().enumerate() {
        for q in 0..n_quad {
            r_k_all[[j, q]] = r_k[q];
            n_k_all[[j, q]] = n_k[q];
        }
    }

    (r_k_all.to_pyarray(py), n_k_all.to_pyarray(py))
}

/// Register M-step functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(m_step_dichotomous_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(m_step_3pl_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_parallel, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_iter: usize) -> DichotomousMstepConfig {
        DichotomousMstepConfig {
            max_iter,
            tol: 1e-12,
            disc_bounds: (0.1, 5.0),
            diff_bounds: (-6.0, 6.0),
            damping: 0.5,
            regularization: 0.01,
        }
    }

    #[test]
    fn canonical_newton_step_matches_reference() {
        let correct = [0.6, 0.9, 0.5];
        let total = [1.3, 1.1, 0.6];
        let points = [-1.5, 0.0, 1.5];

        let (discrimination, difficulty) =
            optimize_dichotomous_item(&correct, &total, &points, 0.8, -0.2, config(1));

        assert!((discrimination - 0.699_711_024_002_301_7).abs() < 1e-12);
        assert!((difficulty - -0.843_022_680_797_700_6).abs() < 1e-12);
    }

    #[test]
    fn optimization_is_monotone_and_empty_counts_are_unchanged() {
        let correct = [0.05, 0.4, 0.8, 0.3];
        let total = [0.8, 1.0, 1.2, 0.9];
        let points = [-3.0, -1.0, 1.0, 3.0];
        let initial = (4.8, 5.5);
        let before = expected_log_likelihood_2pl(&correct, &total, &points, initial.0, initial.1);
        let updated =
            optimize_dichotomous_item(&correct, &total, &points, initial.0, initial.1, config(20));
        let after = expected_log_likelihood_2pl(&correct, &total, &points, updated.0, updated.1);

        assert!(after >= before - 1e-12);
        assert!((0.1..=5.0).contains(&updated.0));
        assert!((-6.0..=6.0).contains(&updated.1));

        let empty = [0.0; 4];
        assert_eq!(
            optimize_dichotomous_item(&empty, &empty, &points, 1.7, -0.4, config(20)),
            (1.7, -0.4)
        );
    }
}
