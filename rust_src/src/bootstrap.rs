//! Bootstrap sampling, imputation, and QMC functions.

use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

use crate::utils::sigmoid;

/// Generate bootstrap sample indices
#[pyfunction]
pub fn generate_bootstrap_indices<'py>(
    py: Python<'py>,
    n_persons: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let indices: Vec<Vec<i64>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(seed + b as u64);
            (0..n_persons)
                .map(|_| rng.random_range(0..n_persons as i64))
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_bootstrap, n_persons));
    for (b, row) in indices.iter().enumerate() {
        for (i, &val) in row.iter().enumerate() {
            result[[b, i]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Resample responses matrix
#[pyfunction]
pub fn resample_responses<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    indices: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let indices = indices.as_array();

    let n_sample = indices.len();
    let n_items = responses.ncols();

    let mut result = Array2::zeros((n_sample, n_items));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_items {
            result[[i, j]] = responses[[idx as usize, j]];
        }
    }

    result.to_pyarray(py)
}

/// Impute missing data using model probabilities
#[pyfunction]
pub fn impute_from_probabilities<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    seed: u64,
) -> Bound<'py, PyArray2<i32>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputed: Vec<Vec<i32>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let mut rng = Pcg64::seed_from_u64(seed + i as u64);
            let theta_i = theta[i];

            (0..n_items)
                .map(|j| {
                    let orig = responses[[i, j]];
                    if orig != missing_code {
                        orig
                    } else {
                        let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                        let u: f64 = rng.random();
                        if u < p {
                            1
                        } else {
                            0
                        }
                    }
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in imputed.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Multiple imputation in parallel
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn multiple_imputation<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta_mean: PyReadonlyArray1<f64>,
    theta_se: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    missing_code: i32,
    n_imputations: usize,
    seed: u64,
) -> Bound<'py, PyArray3<i32>> {
    let responses = responses.as_array();
    let theta_mean = theta_mean.as_array();
    let theta_se = theta_se.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let imputations: Vec<Vec<Vec<i32>>> = (0..n_imputations)
        .into_par_iter()
        .map(|m| {
            let base_seed = seed + (m * n_persons) as u64;

            (0..n_persons)
                .map(|i| {
                    let mut rng = Pcg64::seed_from_u64(base_seed + i as u64);

                    let normal = Normal::new(0.0, 1.0).unwrap();
                    let theta_i = theta_mean[i] + rng.sample(normal) * theta_se[i];

                    (0..n_items)
                        .map(|j| {
                            let orig = responses[[i, j]];
                            if orig != missing_code {
                                orig
                            } else {
                                let p = sigmoid(disc_vec[j] * (theta_i - diff_vec[j]));
                                let u: f64 = rng.random();
                                if u < p {
                                    1
                                } else {
                                    0
                                }
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut result = Array3::zeros((n_imputations, n_persons, n_items));
    for (m, imp) in imputations.iter().enumerate() {
        for (i, row) in imp.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[m, i, j]] = val;
            }
        }
    }

    result.to_pyarray(py)
}

/// Generate quasi-Monte Carlo samples
#[pyfunction]
pub fn generate_qmc_samples<'py>(
    py: Python<'py>,
    n_persons: usize,
    n_samples: usize,
    n_factors: usize,
    _seed: u64,
) -> Bound<'py, PyArray3<f64>> {
    let samples: Vec<Vec<Vec<f64>>> = (0..n_persons)
        .into_par_iter()
        .map(|_i| {
            (0..n_samples)
                .map(|s| {
                    (0..n_factors)
                        .map(|f| {
                            let base = (2 + f) as f64;
                            let mut result = 0.0;
                            let mut fraction = 1.0 / base;
                            let mut n = s + 1;
                            while n > 0 {
                                result += fraction * (n as f64 % base);
                                n /= base as usize;
                                fraction /= base;
                            }
                            let u = result.clamp(0.001, 0.999);
                            let t = (-2.0 * (1.0 - u).ln()).sqrt();
                            let c0 = 2.515517;
                            let c1 = 0.802853;
                            let c2 = 0.010328;
                            let d1 = 1.432788;
                            let d2 = 0.189269;
                            let d3 = 0.001308;
                            let z = t
                                - (c0 + c1 * t + c2 * t * t)
                                    / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
                            if u < 0.5 {
                                -z
                            } else {
                                z
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut result = Array3::zeros((n_persons, n_samples, n_factors));
    for (i, person_samples) in samples.iter().enumerate() {
        for (s, sample) in person_samples.iter().enumerate() {
            for (f, &val) in sample.iter().enumerate() {
                result[[i, s, f]] = val;
            }
        }
    }

    result.to_pyarray(py)
}

/// Register bootstrap functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(resample_responses, m)?)?;
    m.add_function(wrap_pyfunction!(impute_from_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(multiple_imputation, m)?)?;
    m.add_function(wrap_pyfunction!(generate_qmc_samples, m)?)?;
    Ok(())
}
