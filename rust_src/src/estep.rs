//! E-step and expected counts computation functions.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::likelihood::log_2pl_row;
use crate::likelihood_cache::cached_likelihoods;
use crate::utils::{compute_log_weights, logsumexp, normalized_log_gaussian_adjustment};

/// Complete E-step computation with posterior weights
#[pyfunction]
#[pyo3(signature = (responses, quad_points, quad_weights, discrimination, difficulty, prior_mean, prior_var))]
#[allow(clippy::too_many_arguments)]
pub fn e_step_complete<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    prior_mean: f64,
    prior_var: f64,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_quad = quad_points.len();

    let quad_vec = quad_points.to_vec();
    let weight_vec = quad_weights.to_vec();
    let log_weights = compute_log_weights(&weight_vec);
    let adjustment =
        normalized_log_gaussian_adjustment(&quad_vec, &weight_vec, prior_mean, prior_var);
    let (posterior, marginal) = py.detach(|| {
        let mut posterior = cached_likelihoods(
            responses,
            n_quad,
            &vec![2; responses.ncols()],
            true,
            |q, j, row| log_2pl_row(discrimination[j] * (quad_vec[q] - difficulty[j]), row),
        );
        let mut marginal = vec![0.0; n_persons];
        if n_quad > 0 {
            posterior
                .as_slice_mut()
                .expect("contiguous posterior")
                .par_chunks_mut(n_quad)
                .zip(marginal.par_iter_mut())
                .for_each(|(row, marginal)| {
                    for (q, value) in row.iter_mut().enumerate() {
                        *value += log_weights[q] + adjustment[q];
                    }
                    let norm = logsumexp(row);
                    *marginal = norm.exp();
                    for value in row {
                        *value = (*value - norm).exp();
                    }
                });
        }
        (posterior, marginal)
    });
    (posterior.into_pyarray(py), marginal.into_pyarray(py))
}

/// Compute r_k (expected counts) for dichotomous items
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights))]
pub fn compute_expected_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_k = Array1::zeros(n_quad);
    let mut n_k = Array1::zeros(n_quad);

    for i in 0..n_persons {
        let resp = responses[i];
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

    (r_k.to_pyarray(py), n_k.to_pyarray(py))
}

/// Compute r_kc (expected counts per category) for polytomous items
#[pyfunction]
#[pyo3(signature = (responses, posterior_weights, n_categories))]
pub fn compute_expected_counts_polytomous<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray1<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    n_categories: usize,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();

    let n_persons = responses.len();
    let n_quad = posterior_weights.ncols();

    let mut r_kc = Array2::zeros((n_quad, n_categories));

    for i in 0..n_persons {
        let resp = responses[i];
        if resp < 0 || resp as usize >= n_categories {
            continue;
        }
        for q in 0..n_quad {
            r_kc[[q, resp as usize]] += posterior_weights[[i, q]];
        }
    }

    r_kc.to_pyarray(py)
}

/// Register E-step functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(e_step_complete, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_counts_polytomous, m)?)?;
    Ok(())
}
