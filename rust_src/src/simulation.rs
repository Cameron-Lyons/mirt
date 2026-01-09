//! Response simulation functions for various IRT models.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Simulate responses from Graded Response Model
#[pyfunction]
pub fn simulate_grm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items).map(|i| thresholds.row(i).to_vec()).collect();

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    let mut cum_probs = vec![1.0; n_categories];
                    for k in 0..(n_categories - 1) {
                        let z = disc_vec[j] * (theta_i - thresh_vec[j][k]);
                        cum_probs[k + 1] = sigmoid(z);
                    }

                    let mut cat_probs = vec![0.0; n_categories];
                    for k in 0..n_categories {
                        let next = if k < n_categories - 1 {
                            cum_probs[k + 1]
                        } else {
                            0.0
                        };
                        cat_probs[k] = (cum_probs[k] - next).max(0.0);
                    }

                    let sum: f64 = cat_probs.iter().sum();
                    if sum > EPSILON {
                        for p in &mut cat_probs {
                            *p /= sum;
                        }
                    }

                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Simulate responses from Generalized Partial Credit Model
#[pyfunction]
pub fn simulate_gpcm<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    discrimination: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray2<f64>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let thresholds = thresholds.as_array();

    let n_persons = theta.nrows();
    let n_items = discrimination.len();
    let n_categories = thresholds.ncols() + 1;

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let thresh_vec: Vec<Vec<f64>> = (0..n_items).map(|i| thresholds.row(i).to_vec()).collect();

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[[i, 0]];

            (0..n_items)
                .map(|j| {
                    let mut numerators = vec![0.0; n_categories];
                    for (k, num) in numerators.iter_mut().enumerate() {
                        let cumsum: f64 = thresh_vec[j][..k]
                            .iter()
                            .map(|&t| disc_vec[j] * (theta_i - t))
                            .sum();
                        *num = cumsum.exp();
                    }

                    let sum: f64 = numerators.iter().sum();
                    let cat_probs: Vec<f64> = numerators.iter().map(|&n| n / sum).collect();

                    let u: f64 = rng.random();
                    let mut cumsum = 0.0;
                    for (k, &p) in cat_probs.iter().enumerate() {
                        cumsum += p;
                        if u < cumsum {
                            return k as i32;
                        }
                    }
                    (n_categories - 1) as i32
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Simulate dichotomous responses (2PL/3PL)
#[pyfunction]
pub fn simulate_dichotomous<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: Option<PyReadonlyArray1<f64>>,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();
    let guess_vec: Vec<f64> = guessing
        .map(|g| g.as_array().to_vec())
        .unwrap_or_else(|| vec![0.0; n_items]);

    let responses: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let z = disc_vec[j] * (theta_i - diff_vec[j]);
                    let p_star = sigmoid(z);
                    let p = guess_vec[j] + (1.0 - guess_vec[j]) * p_star;

                    let u: f64 = rng.random();
                    if u < p { 1 } else { 0 }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in responses.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Register simulation functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_grm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gpcm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_dichotomous, m)?)?;
    Ok(())
}
