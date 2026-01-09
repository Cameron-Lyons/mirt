//! Advanced MIRT model probability computations.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::sigmoid;

/// Compute probabilities for partially compensatory MIRT model
#[pyfunction]
pub fn compute_partially_compensatory_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
    compensation: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let compensation = compensation.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        let p_k = sigmoid(z);
                        prob *= p_k.powf(compensation[[j, k]]);
                    }
                    prob
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute probabilities for noncompensatory (conjunctive) MIRT model
#[pyfunction]
pub fn compute_noncompensatory_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        prob *= sigmoid(z);
                    }
                    prob
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute probabilities for disjunctive MIRT model
#[pyfunction]
pub fn compute_disjunctive_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray2<f64>,
    difficulty: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.nrows();
    let n_factors = theta.ncols();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            (0..n_items)
                .map(|j| {
                    let mut prob_fail_all = 1.0;
                    for k in 0..n_factors {
                        let z = discrimination[[j, k]] * (theta[[i, k]] - difficulty[[j, k]]);
                        let p_k = sigmoid(z);
                        prob_fail_all *= 1.0 - p_k;
                    }
                    1.0 - prob_fail_all
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in probs.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute probabilities for sequential response model
#[pyfunction]
pub fn compute_sequential_probs<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    n_categories: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();
    let n_categories = n_categories.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();
    let max_cats = n_categories.iter().max().copied().unwrap_or(2) as usize;

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            let mut person_probs = vec![0.0; n_items * max_cats];

            for j in 0..n_items {
                let n_cat = n_categories[j] as usize;
                let a = discrimination[j];

                let mut step_probs = Vec::with_capacity(n_cat - 1);
                for k in 0..(n_cat - 1) {
                    let z = a * (theta_i - thresholds[[j, k]]);
                    step_probs.push(sigmoid(z));
                }

                for k in 0..n_cat {
                    let mut prob: f64 = step_probs.iter().take(k).product();

                    if k < n_cat - 1 && k < step_probs.len() {
                        prob *= 1.0 - step_probs[k];
                    }

                    person_probs[j * max_cats + k] = prob;
                }
            }

            person_probs
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items * max_cats));
    for (i, row) in probs.iter().enumerate() {
        for (jk, &val) in row.iter().enumerate() {
            result[[i, jk]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Register MIRT model functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_partially_compensatory_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_noncompensatory_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_disjunctive_probs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sequential_probs, m)?)?;
    Ok(())
}
