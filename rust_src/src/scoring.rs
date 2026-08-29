//! Person scoring functions (EAP, WLE).

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{
    EPSILON, compute_eap_with_se, compute_log_weights, log_likelihood_2pl_single,
    normalize_log_posterior, sigmoid,
};

type PyScoreArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Compute EAP (Expected A Posteriori) scores
#[pyfunction]
pub fn compute_eap_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let quad_vec: Vec<f64> = quad_points.to_vec();
    let weight_vec: Vec<f64> = quad_weights.to_vec();
    let log_weights = compute_log_weights(&weight_vec);

    let results: Vec<(f64, f64)> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();

            let log_posterior: Vec<f64> = (0..n_quad)
                .map(|q| {
                    log_likelihood_2pl_single(&resp_row, quad_vec[q], &disc_vec, &diff_vec)
                        + log_weights[q]
                })
                .collect();

            let posterior = normalize_log_posterior(&log_posterior);
            compute_eap_with_se(&posterior, &quad_vec)
        })
        .collect();

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    (theta.to_pyarray(py), se.to_pyarray(py))
}

/// Compute WLE (Weighted Likelihood Estimation) scores
#[pyfunction]
#[pyo3(signature = (responses, discrimination, difficulty, theta_min, theta_max, tol, n_jobs=1))]
#[allow(clippy::too_many_arguments)]
pub fn compute_wle_scores<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    theta_min: f64,
    theta_max: f64,
    tol: f64,
    n_jobs: usize,
) -> PyResult<PyScoreArrays<'py>> {
    let responses = responses.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    if discrimination.len() != n_items || difficulty.len() != n_items {
        return Err(PyValueError::new_err(
            "discrimination and difficulty must contain one value per item",
        ));
    }
    if !discrimination.iter().all(|value| value.is_finite())
        || !difficulty.iter().all(|value| value.is_finite())
    {
        return Err(PyValueError::new_err("item parameters must be finite"));
    }
    if !theta_min.is_finite() || !theta_max.is_finite() || theta_min >= theta_max {
        return Err(PyValueError::new_err(
            "theta bounds must be finite and strictly increasing",
        ));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs must be a positive integer"));
    }
    if responses.iter().any(|&response| response > 1) {
        return Err(PyValueError::new_err(
            "observed responses must contain only 0 or 1",
        ));
    }

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let score_person = |i: usize| {
        let resp_row: Vec<i32> = responses.row(i).to_vec();
        wle_score_person(&resp_row, &disc_vec, &diff_vec, theta_min, theta_max, tol)
    };
    let results: Vec<(f64, f64)> = if n_jobs == 1 || n_persons < 2 {
        (0..n_persons).map(score_person).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs.min(n_persons))
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        pool.install(|| (0..n_persons).into_par_iter().map(score_person).collect())
    };

    let theta: Array1<f64> = results.iter().map(|(t, _)| *t).collect::<Vec<_>>().into();
    let se: Array1<f64> = results.iter().map(|(_, s)| *s).collect::<Vec<_>>().into();

    Ok((theta.to_pyarray(py), se.to_pyarray(py)))
}

/// WLE criterion function (log-likelihood + 0.5 * log(information))
#[inline]
fn wle_criterion(responses: &[i32], theta: f64, discrimination: &[f64], difficulty: &[f64]) -> f64 {
    let ll = log_likelihood_2pl_single(responses, theta, discrimination, difficulty);
    let info = observed_fisher_info_2pl(responses, theta, discrimination, difficulty);
    if info > EPSILON {
        ll + 0.5 * info.ln()
    } else {
        ll
    }
}

#[inline]
fn observed_fisher_info_2pl(
    responses: &[i32],
    theta: f64,
    discrimination: &[f64],
    difficulty: &[f64],
) -> f64 {
    responses
        .iter()
        .zip(discrimination.iter().zip(difficulty.iter()))
        .filter(|(response, _)| **response >= 0)
        .map(|(_, (a, b))| {
            let probability = sigmoid(a * (theta - b));
            a * a * probability * (1.0 - probability)
        })
        .sum()
}

fn wle_score_person(
    responses: &[i32],
    discrimination: &[f64],
    difficulty: &[f64],
    theta_min: f64,
    theta_max: f64,
    tol: f64,
) -> (f64, f64) {
    if responses.iter().all(|&response| response < 0) {
        return (0.0, f64::INFINITY);
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut lower = theta_min;
    let mut upper = theta_max;
    let mut left = upper - (upper - lower) / phi;
    let mut right = lower + (upper - lower) / phi;
    let mut left_value = wle_criterion(responses, left, discrimination, difficulty);
    let mut right_value = wle_criterion(responses, right, discrimination, difficulty);
    for _ in 0..256 {
        if (upper - lower) <= tol {
            break;
        }
        let previous_lower = lower;
        let previous_upper = upper;
        if left_value > right_value {
            upper = right;
            right = left;
            right_value = left_value;
            left = upper - (upper - lower) / phi;
            left_value = wle_criterion(responses, left, discrimination, difficulty);
        } else {
            lower = left;
            left = right;
            left_value = right_value;
            right = lower + (upper - lower) / phi;
            right_value = wle_criterion(responses, right, discrimination, difficulty);
        }
        if lower == previous_lower && upper == previous_upper {
            break;
        }
    }

    let theta = (lower + upper) / 2.0;
    let information = observed_fisher_info_2pl(responses, theta, discrimination, difficulty);
    let standard_error = if information > EPSILON {
        1.0 / information.sqrt()
    } else {
        f64::INFINITY
    };
    (theta, standard_error)
}

/// Register scoring functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_eap_scores, m)?)?;
    m.add_function(wrap_pyfunction!(compute_wle_scores, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{observed_fisher_info_2pl, wle_score_person};

    #[test]
    fn observed_information_ignores_missing_items() {
        let responses = [1, -1, 0];
        let discrimination = [1.0, 25.0, 2.0];
        let difficulty = [0.0, 0.0, 0.0];

        let information = observed_fisher_info_2pl(&responses, 0.0, &discrimination, &difficulty);

        assert!((information - 1.25).abs() < 1e-12);
    }

    #[test]
    fn all_missing_wle_has_unknown_precision() {
        let result = wle_score_person(&[-1, -4], &[1.0, 1.5], &[0.0, 0.5], -6.0, 6.0, 1e-6);

        assert_eq!(result.0, 0.0);
        assert!(result.1.is_infinite());
    }
}
