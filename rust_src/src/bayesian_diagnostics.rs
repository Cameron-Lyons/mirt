//! Bayesian diagnostics for IRT models.
//!
//! This module provides the pointwise log-likelihood matrix used by the
//! Python PSIS-LOO and WAIC implementations.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

#[inline]
fn binary_log_likelihood(response: i32, linear_predictor: f64) -> f64 {
    let probability = sigmoid(linear_predictor).clamp(EPSILON, 1.0 - EPSILON);
    if response == 1 {
        probability.ln()
    } else {
        (1.0 - probability).ln()
    }
}

#[pyfunction]
#[pyo3(signature = (
    responses,
    discrimination_chain,
    difficulty_chain,
    theta_chain,
    aggregation=0
))]
pub fn compute_pointwise_loglik_2pl<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    discrimination_chain: PyReadonlyArray2<f64>,
    difficulty_chain: PyReadonlyArray2<f64>,
    theta_chain: PyReadonlyArray2<f64>,
    aggregation: u8,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let responses = responses.as_array();
    let discrimination_chain = discrimination_chain.as_array();
    let difficulty_chain = difficulty_chain.as_array();
    let theta_chain = theta_chain.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_samples = discrimination_chain.nrows();

    if n_samples == 0 {
        return Err(PyValueError::new_err(
            "parameter chains must contain at least one sample",
        ));
    }
    if discrimination_chain.ncols() != n_items || difficulty_chain.dim() != (n_samples, n_items) {
        return Err(PyValueError::new_err(
            "item parameter chains must have shape (n_samples, n_items)",
        ));
    }
    if theta_chain.dim() != (n_samples, n_persons) {
        return Err(PyValueError::new_err(
            "theta_chain must have shape (n_samples, n_persons)",
        ));
    }
    if responses.iter().any(|&response| response > 1) {
        return Err(PyValueError::new_err(
            "observed responses must contain only 0 or 1",
        ));
    }
    if !discrimination_chain.iter().all(|value| value.is_finite())
        || !difficulty_chain.iter().all(|value| value.is_finite())
        || !theta_chain.iter().all(|value| value.is_finite())
    {
        return Err(PyValueError::new_err(
            "chains must contain only finite values",
        ));
    }
    if aggregation > 2 {
        return Err(PyValueError::new_err(
            "aggregation must be 0 (person), 1 (observation), or 2 (observed)",
        ));
    }

    let n_observed = responses.iter().filter(|&&response| response >= 0).count();
    let output_width = match aggregation {
        0 => Some(n_persons),
        1 => n_persons.checked_mul(n_items),
        _ => Some(n_observed),
    }
    .ok_or_else(|| PyValueError::new_err("requested output is too large"))?;
    let output_length = n_samples
        .checked_mul(output_width)
        .ok_or_else(|| PyValueError::new_err("requested output is too large"))?;
    let mut values = vec![0.0; output_length];

    if output_width > 0 {
        values
            .par_chunks_mut(output_width)
            .enumerate()
            .for_each(|(sample, output)| {
                let discrimination = discrimination_chain.row(sample);
                let difficulty = difficulty_chain.row(sample);
                let theta = theta_chain.row(sample);
                let mut observed_index = 0;

                for person in 0..n_persons {
                    let mut person_value = 0.0;
                    for item in 0..n_items {
                        let response = responses[[person, item]];
                        if response < 0 {
                            continue;
                        }
                        let linear_predictor =
                            discrimination[item] * (theta[person] - difficulty[item]);
                        let pointwise = binary_log_likelihood(response, linear_predictor);
                        match aggregation {
                            0 => person_value += pointwise,
                            1 => output[person * n_items + item] = pointwise,
                            _ => {
                                output[observed_index] = pointwise;
                                observed_index += 1;
                            }
                        }
                    }
                    if aggregation == 0 {
                        output[person] = person_value;
                    }
                }
            });
    }

    let result = Array2::from_shape_vec((n_samples, output_width), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(result.to_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_pointwise_loglik_2pl, m)?)?;
    Ok(())
}
