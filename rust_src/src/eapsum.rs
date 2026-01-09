//! EAPsum scoring and Lord-Wingersky recursion functions.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

use crate::utils::{compute_eap_with_se, log_sigmoid, logsumexp, normalize_log_posterior, EPSILON};

/// Lord-Wingersky recursion for computing sum score distributions.
/// This is the core algorithm for EAPsum scoring.
///
/// For dichotomous items, computes P(sum_score = s | theta) for all s and theta.
/// Returns log probabilities for numerical stability.
#[pyfunction]
pub fn lord_wingersky_recursion<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    discrimination: PyReadonlyArray1<'py, f64>,
    difficulty: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_quad = theta.len();
    let n_items = discrimination.len();
    let max_score = n_items;

    let mut log_dist = Array2::from_elem((max_score + 1, n_quad), f64::NEG_INFINITY);

    for q in 0..n_quad {
        log_dist[[0, q]] = 0.0;
    }

    for j in 0..n_items {
        let a = discrimination[j];
        let b = difficulty[j];

        let mut log_p1 = Vec::with_capacity(n_quad);
        let mut log_p0 = Vec::with_capacity(n_quad);

        for q in 0..n_quad {
            let z = a * (theta[q] - b);
            log_p1.push(log_sigmoid(z));
            log_p0.push(log_sigmoid(-z));
        }

        let mut new_log_dist = Array2::from_elem((max_score + 1, n_quad), f64::NEG_INFINITY);

        for s in 0..=max_score {
            for q in 0..n_quad {
                let log_stay = log_dist[[s, q]] + log_p0[q];

                if s > 0 {
                    let log_up = log_dist[[s - 1, q]] + log_p1[q];
                    let max_val = log_stay.max(log_up);
                    if max_val.is_finite() {
                        new_log_dist[[s, q]] =
                            max_val + ((log_stay - max_val).exp() + (log_up - max_val).exp()).ln();
                    } else {
                        new_log_dist[[s, q]] = f64::NEG_INFINITY;
                    }
                } else {
                    new_log_dist[[s, q]] = log_stay;
                }
            }
        }

        log_dist = new_log_dist;
    }

    log_dist.to_pyarray(py)
}

/// Lord-Wingersky recursion for polytomous items (GRM/GPCM).
///
/// item_probs: shape (n_items, n_quad, max_categories) - P(X_j = k | theta)
/// Returns log P(sum_score = s | theta) for all s
#[pyfunction]
pub fn lord_wingersky_polytomous<'py>(
    py: Python<'py>,
    item_probs: PyReadonlyArray3<'py, f64>,
    max_score: usize,
) -> Bound<'py, PyArray2<f64>> {
    let item_probs = item_probs.as_array();

    let n_items = item_probs.shape()[0];
    let n_quad = item_probs.shape()[1];
    let max_cats = item_probs.shape()[2];

    let mut log_dist = Array2::from_elem((max_score + 1, n_quad), f64::NEG_INFINITY);

    for q in 0..n_quad {
        log_dist[[0, q]] = 0.0;
    }

    for j in 0..n_items {
        let mut new_log_dist = Array2::from_elem((max_score + 1, n_quad), f64::NEG_INFINITY);

        for s in 0..=max_score {
            for q in 0..n_quad {
                let mut log_terms = Vec::new();

                for c in 0..max_cats {
                    if s >= c {
                        let prev_s = s - c;
                        if log_dist[[prev_s, q]].is_finite() {
                            let p = item_probs[[j, q, c]].max(EPSILON);
                            let log_term = log_dist[[prev_s, q]] + p.ln();
                            if log_term.is_finite() {
                                log_terms.push(log_term);
                            }
                        }
                    }
                }

                if !log_terms.is_empty() {
                    new_log_dist[[s, q]] = logsumexp(&log_terms);
                }
            }
        }

        log_dist = new_log_dist;
    }

    log_dist.to_pyarray(py)
}

/// Compute EAPsum scores using pre-computed sum score distribution.
///
/// log_p_score_theta: shape (max_score + 1, n_quad) - log P(score | theta)
/// log_prior: shape (n_quad,) - log prior weights
/// sum_scores: shape (n_persons,) - observed sum scores
/// theta_points: shape (n_quad,) - quadrature points
///
/// Returns (theta_estimates, standard_errors)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn eapsum_from_distribution<'py>(
    py: Python<'py>,
    log_p_score_theta: PyReadonlyArray2<'py, f64>,
    log_prior: PyReadonlyArray1<'py, f64>,
    sum_scores: PyReadonlyArray1<'py, i32>,
    theta_points: PyReadonlyArray1<'py, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let log_p = log_p_score_theta.as_array();
    let log_prior = log_prior.as_array();
    let sum_scores = sum_scores.as_array();
    let theta_points = theta_points.as_array();

    let n_persons = sum_scores.len();
    let n_quad = theta_points.len();
    let max_score = log_p.shape()[0] - 1;

    let mut theta_est = Array1::zeros(n_persons);
    let mut theta_se = Array1::zeros(n_persons);

    let theta_vec: Vec<f64> = theta_points.to_vec();

    for i in 0..n_persons {
        let s = sum_scores[i].max(0).min(max_score as i32) as usize;

        let log_posterior: Vec<f64> = (0..n_quad).map(|q| log_p[[s, q]] + log_prior[q]).collect();

        let posterior = normalize_log_posterior(&log_posterior);
        let (theta_eap, se) = compute_eap_with_se(&posterior, &theta_vec);

        theta_est[i] = theta_eap;
        theta_se[i] = se;
    }

    (theta_est.to_pyarray(py), theta_se.to_pyarray(py))
}

/// Register EAPsum functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lord_wingersky_recursion, m)?)?;
    m.add_function(wrap_pyfunction!(lord_wingersky_polytomous, m)?)?;
    m.add_function(wrap_pyfunction!(eapsum_from_distribution, m)?)?;
    Ok(())
}
