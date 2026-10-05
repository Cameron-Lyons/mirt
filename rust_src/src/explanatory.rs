//! Explanatory IRT models (latent regression).

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Integrate 2PL response-pattern likelihoods over person-specific normal priors.
///
/// The reduction streams over quadrature points for each person, avoiding the
/// person-by-quadrature and person-by-item temporary arrays needed by the NumPy
/// implementation. Negative responses are treated as missing.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn explanatory_marginal_log_likelihood<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    prior_means: PyReadonlyArray1<f64>,
    residual_std: f64,
    quad_nodes: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let responses = responses.as_array();
    let prior_means = prior_means.as_array();
    let quad_nodes = quad_nodes.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    if prior_means.len() != n_persons {
        return Err(PyValueError::new_err(format!(
            "prior_means must have length {n_persons}"
        )));
    }
    if discrimination.len() != n_items || difficulty.len() != n_items {
        return Err(PyValueError::new_err(format!(
            "discrimination and difficulty must have length {n_items}"
        )));
    }
    if quad_nodes.is_empty() || quad_weights.len() != quad_nodes.len() {
        return Err(PyValueError::new_err(
            "quadrature nodes and weights must be non-empty and have equal lengths",
        ));
    }
    if !residual_std.is_finite() || residual_std <= 0.0 {
        return Err(PyValueError::new_err(
            "residual_std must be finite and positive",
        ));
    }
    if prior_means.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("prior_means must be finite"));
    }
    if quad_nodes.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("quadrature nodes must be finite"));
    }
    if quad_weights
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(PyValueError::new_err(
            "quadrature weights must be finite and positive",
        ));
    }
    if discrimination.iter().any(|value| !value.is_finite())
        || difficulty.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "item parameters must contain only finite values",
        ));
    }
    if responses.iter().any(|response| *response > 1) {
        return Err(PyValueError::new_err(
            "observed responses must contain only 0 or 1",
        ));
    }

    let result: Vec<f64> = (0..n_persons)
        .into_par_iter()
        .map(|person| {
            let mean = prior_means[person];
            let mut maximum = f64::NEG_INFINITY;
            let mut scaled_sum = 0.0;

            for point in 0..quad_nodes.len() {
                let theta = mean + residual_std * quad_nodes[point];
                let mut term = quad_weights[point].ln();
                for item in 0..n_items {
                    let response = responses[[person, item]];
                    if response < 0 {
                        continue;
                    }
                    let z = discrimination[item] * (theta - difficulty[item]);
                    let probability = sigmoid(z).clamp(EPSILON, 1.0 - EPSILON);
                    term += if response == 1 {
                        probability.ln()
                    } else {
                        (-probability).ln_1p()
                    };
                }

                if term <= maximum {
                    scaled_sum += (term - maximum).exp();
                } else {
                    scaled_sum = scaled_sum * (maximum - term).exp() + 1.0;
                    maximum = term;
                }
            }
            maximum + scaled_sum.ln()
        })
        .collect();

    Ok(Array1::from_vec(result).to_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(explanatory_marginal_log_likelihood, m)?)?;
    Ok(())
}
