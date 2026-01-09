//! Diagnostic functions (residuals, Q3, LD chi2, fit statistics, margins).

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{EPSILON, sigmoid};

/// Compute observed univariate and bivariate margins
#[pyfunction]
pub fn compute_observed_margins<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let responses = responses.as_array();
    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let obs_uni: Array1<f64> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let valid: Vec<f64> = responses
                .column(j)
                .iter()
                .filter(|&&r| r >= 0)
                .map(|&r| r as f64)
                .collect();
            if valid.is_empty() {
                0.0
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        })
        .collect::<Vec<f64>>()
        .into();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum = 0.0;
            let mut count = 0;
            for p in 0..n_persons {
                let ri = responses[[p, i]];
                let rj = responses[[p, j]];
                if ri >= 0 && rj >= 0 {
                    sum += (ri * rj) as f64;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            ((i, j), mean)
        })
        .collect();

    let mut obs_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        obs_bi[[i, j]] = val;
        obs_bi[[j, i]] = val;
    }

    (obs_uni.to_pyarray(py), obs_bi.to_pyarray(py))
}

/// Compute expected univariate and bivariate margins
#[pyfunction]
pub fn compute_expected_margins<'py>(
    py: Python<'py>,
    quad_points: PyReadonlyArray1<f64>,
    quad_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>) {
    let quad_points = quad_points.as_array();
    let quad_weights = quad_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_quad = quad_points.len();
    let n_items = discrimination.len();

    let probs: Vec<Vec<f64>> = (0..n_items)
        .map(|j| {
            (0..n_quad)
                .map(|q| sigmoid(discrimination[j] * (quad_points[q] - difficulty[j])))
                .collect()
        })
        .collect();

    let exp_uni: Array1<f64> = (0..n_items)
        .map(|j| {
            probs[j]
                .iter()
                .zip(quad_weights.iter())
                .map(|(&p, &w)| p * w)
                .sum()
        })
        .collect::<Vec<f64>>()
        .into();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let bivariate: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let exp: f64 = (0..n_quad)
                .map(|q| probs[i][q] * probs[j][q] * quad_weights[q])
                .sum();
            ((i, j), exp)
        })
        .collect();

    let mut exp_bi = Array2::zeros((n_items, n_items));
    for ((i, j), val) in bivariate {
        exp_bi[[i, j]] = val;
        exp_bi[[j, i]] = val;
    }

    (exp_uni.to_pyarray(py), exp_bi.to_pyarray(py))
}

/// Compute standardized residuals
#[pyfunction]
pub fn compute_standardized_residuals<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let residuals: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return f64::NAN;
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let expected = p;
                    let variance = p * (1.0 - p);
                    (resp as f64 - expected) / (variance + EPSILON).sqrt()
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_items));
    for (i, row) in residuals.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute Q3 (residual correlation) matrix - Yen's local dependence statistic
#[pyfunction]
pub fn compute_q3_matrix<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let residuals: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return f64::NAN;
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let expected = p;
                    let variance = p * (1.0 - p);
                    (resp as f64 - expected) / (variance + EPSILON).sqrt()
                })
                .collect()
        })
        .collect();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let correlations: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x2 = 0.0;
            let mut sum_y2 = 0.0;
            let mut n = 0.0;

            for person_residuals in &residuals {
                let r_i = person_residuals[i];
                let r_j = person_residuals[j];
                if r_i.is_nan() || r_j.is_nan() {
                    continue;
                }
                sum_xy += r_i * r_j;
                sum_x += r_i;
                sum_y += r_j;
                sum_x2 += r_i * r_i;
                sum_y2 += r_j * r_j;
                n += 1.0;
            }

            if n < 3.0 {
                return ((i, j), f64::NAN);
            }

            let mean_x = sum_x / n;
            let mean_y = sum_y / n;
            let var_x = sum_x2 / n - mean_x * mean_x;
            let var_y = sum_y2 / n - mean_y * mean_y;
            let cov_xy = sum_xy / n - mean_x * mean_y;

            let denom = (var_x * var_y).sqrt();
            let corr = if denom > EPSILON { cov_xy / denom } else { 0.0 };

            ((i, j), corr)
        })
        .collect();

    let mut q3_matrix = Array2::zeros((n_items, n_items));
    for ((i, j), corr) in correlations {
        q3_matrix[[i, j]] = corr;
        q3_matrix[[j, i]] = corr;
    }

    q3_matrix.to_pyarray(py)
}

/// Compute LD chi-square statistics for all item pairs
#[pyfunction]
pub fn compute_ld_chi2_matrix<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let pairs: Vec<(usize, usize)> = (0..n_items)
        .flat_map(|i| ((i + 1)..n_items).map(move |j| (i, j)))
        .collect();

    let chi2_values: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut obs = [[0.0; 2]; 2];
            let mut exp = [[0.0; 2]; 2];

            for p in 0..n_persons {
                let r_i = responses[[p, i]];
                let r_j = responses[[p, j]];
                if r_i < 0 || r_j < 0 {
                    continue;
                }

                let theta_p = theta[p];
                let p_i = sigmoid(discrimination[i] * (theta_p - difficulty[i]));
                let p_j = sigmoid(discrimination[j] * (theta_p - difficulty[j]));

                obs[r_i as usize][r_j as usize] += 1.0;

                exp[0][0] += (1.0 - p_i) * (1.0 - p_j);
                exp[0][1] += (1.0 - p_i) * p_j;
                exp[1][0] += p_i * (1.0 - p_j);
                exp[1][1] += p_i * p_j;
            }

            let mut chi2 = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    let e = exp[a][b].max(0.5);
                    chi2 += (obs[a][b] - e).powi(2) / e;
                }
            }

            ((i, j), chi2)
        })
        .collect();

    let mut chi2_matrix = Array2::zeros((n_items, n_items));
    for ((i, j), chi2) in chi2_values {
        chi2_matrix[[i, j]] = chi2;
        chi2_matrix[[j, i]] = chi2;
    }

    chi2_matrix.to_pyarray(py)
}

/// Compute item and person fit statistics (infit/outfit)
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn compute_fit_statistics<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let responses = responses.as_array();
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();

    let z_sq_var: Vec<Vec<(f64, f64)>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let resp = responses[[i, j]];
                    if resp < 0 {
                        return (f64::NAN, f64::NAN);
                    }
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let var = p * (1.0 - p);
                    let raw_resid = resp as f64 - p;
                    let z_sq = (raw_resid * raw_resid) / (var + EPSILON);
                    (z_sq, var)
                })
                .collect()
        })
        .collect();

    let item_stats: Vec<(f64, f64)> = (0..n_items)
        .into_par_iter()
        .map(|j| {
            let mut sum_z_sq = 0.0;
            let mut sum_z_sq_var = 0.0;
            let mut sum_var = 0.0;
            let mut count = 0.0;

            for person_data in &z_sq_var {
                let (z_sq, var) = person_data[j];
                if z_sq.is_nan() {
                    continue;
                }
                sum_z_sq += z_sq;
                sum_z_sq_var += z_sq * var;
                sum_var += var;
                count += 1.0;
            }

            let outfit = if count > 0.0 {
                sum_z_sq / count
            } else {
                f64::NAN
            };
            let infit = if sum_var > EPSILON {
                sum_z_sq_var / sum_var
            } else {
                f64::NAN
            };

            (outfit, infit)
        })
        .collect();

    let person_stats: Vec<(f64, f64)> = z_sq_var
        .par_iter()
        .map(|person_data| {
            let mut sum_z_sq = 0.0;
            let mut sum_z_sq_var = 0.0;
            let mut sum_var = 0.0;
            let mut count = 0.0;

            for &(z_sq, var) in person_data {
                if z_sq.is_nan() {
                    continue;
                }
                sum_z_sq += z_sq;
                sum_z_sq_var += z_sq * var;
                sum_var += var;
                count += 1.0;
            }

            let outfit = if count > 0.0 {
                sum_z_sq / count
            } else {
                f64::NAN
            };
            let infit = if sum_var > EPSILON {
                sum_z_sq_var / sum_var
            } else {
                f64::NAN
            };

            (outfit, infit)
        })
        .collect();

    let item_outfit: Array1<f64> = item_stats
        .iter()
        .map(|(o, _)| *o)
        .collect::<Vec<_>>()
        .into();
    let item_infit: Array1<f64> = item_stats
        .iter()
        .map(|(_, i)| *i)
        .collect::<Vec<_>>()
        .into();
    let person_outfit: Array1<f64> = person_stats
        .iter()
        .map(|(o, _)| *o)
        .collect::<Vec<_>>()
        .into();
    let person_infit: Array1<f64> = person_stats
        .iter()
        .map(|(_, i)| *i)
        .collect::<Vec<_>>()
        .into();

    (
        item_outfit.to_pyarray(py),
        item_infit.to_pyarray(py),
        person_outfit.to_pyarray(py),
        person_infit.to_pyarray(py),
    )
}

/// Compute probabilities for all items in batch (2PL model).
///
/// Parallelizes over persons for efficient computation.
#[pyfunction]
pub fn compute_probabilities_batch<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| sigmoid(discrimination[j] * (theta_i - difficulty[j])))
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

/// Compute probabilities with guessing parameter (3PL model).
#[pyfunction]
pub fn compute_probabilities_batch_3pl<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
    guessing: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();
    let guessing = guessing.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let probs: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let p_star = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    guessing[j] + (1.0 - guessing[j]) * p_star
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

/// Compute expected values and variances for all items in batch.
///
/// Returns (expected, variance) arrays of shape (n_persons, n_items).
#[pyfunction]
pub fn compute_expected_variance_batch<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let theta = theta.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = theta.len();
    let n_items = discrimination.len();

    let results: Vec<Vec<(f64, f64)>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let theta_i = theta[i];
            (0..n_items)
                .map(|j| {
                    let p = sigmoid(discrimination[j] * (theta_i - difficulty[j]));
                    let expected = p;
                    let variance = p * (1.0 - p);
                    (expected, variance)
                })
                .collect()
        })
        .collect();

    let mut expected = Array2::zeros((n_persons, n_items));
    let mut variance = Array2::zeros((n_persons, n_items));

    for (i, row) in results.iter().enumerate() {
        for (j, &(e, v)) in row.iter().enumerate() {
            expected[[i, j]] = e;
            variance[[i, j]] = v;
        }
    }

    (expected.to_pyarray(py), variance.to_pyarray(py))
}

/// Register diagnostics functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_observed_margins, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_margins, m)?)?;
    m.add_function(wrap_pyfunction!(compute_standardized_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(compute_q3_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ld_chi2_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fit_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_probabilities_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_probabilities_batch_3pl, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_variance_batch, m)?)?;
    Ok(())
}
