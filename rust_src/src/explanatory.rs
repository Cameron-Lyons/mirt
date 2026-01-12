//! Explanatory IRT models (LLTM, Latent Regression).
//!
//! This module provides optimized implementations for:
//! - Linear Logistic Test Model (LLTM) log-likelihood
//! - Latent regression with person covariates
//! - Combined explanatory IRT E-step

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::log_sigmoid;

/// Compute item difficulties from feature weights.
fn compute_difficulties(
    item_features: ArrayView2<f64>,
    feature_weights: ArrayView1<f64>,
) -> Array1<f64> {
    item_features.dot(&feature_weights)
}

/// Compute log-likelihoods for LLTM model.
#[pyfunction]
pub fn lltm_log_likelihoods<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    item_features: PyReadonlyArray2<f64>,
    feature_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let item_features = item_features.as_array();
    let feature_weights = feature_weights.as_array();
    let discrimination = discrimination.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let difficulties = compute_difficulties(item_features, feature_weights);
    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulties.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];
                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let r = resp_row[j];
                        if r >= 0 {
                            let z = disc_vec[j] * (theta - diff_vec[j]);
                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }
                    ll
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute expected counts r_k for LLTM M-step.
///
/// Returns the expected number of correct responses at each quadrature point
/// for each item feature.
#[pyfunction]
pub fn lltm_expected_feature_counts<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    item_features: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let item_features = item_features.as_array();

    let n_items = responses.ncols();
    let n_quad = posterior_weights.ncols();
    let n_features = item_features.ncols();

    let mut feature_counts = Array2::<f64>::zeros((n_features, n_quad));

    for j in 0..n_items {
        let item_resp: Vec<i32> = responses.column(j).to_vec();
        let feature_row = item_features.row(j);

        for q in 0..n_quad {
            let mut weighted_correct = 0.0;
            for (i, &r) in item_resp.iter().enumerate() {
                if r > 0 {
                    weighted_correct += posterior_weights[[i, q]];
                }
            }

            for (k, &f) in feature_row.iter().enumerate() {
                feature_counts[[k, q]] += f * weighted_correct;
            }
        }
    }

    feature_counts.to_pyarray(py)
}

/// Compute gradient of LLTM log-likelihood with respect to feature weights.
#[pyfunction]
pub fn lltm_gradient<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    item_features: PyReadonlyArray2<f64>,
    feature_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let responses = responses.as_array();
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let item_features = item_features.as_array();
    let feature_weights = feature_weights.as_array();
    let discrimination = discrimination.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();
    let n_features = item_features.ncols();

    let difficulties = compute_difficulties(item_features, feature_weights);

    let mut gradient = Array1::<f64>::zeros(n_features);

    for j in 0..n_items {
        let a_j = discrimination[j];
        let b_j = difficulties[j];
        let feature_row = item_features.row(j);

        for q in 0..n_quad {
            let theta = quad_points[q];
            let z = a_j * (theta - b_j);
            let p = 1.0 / (1.0 + (-z).exp());

            let mut r_jq = 0.0;
            let mut n_jq = 0.0;

            for i in 0..n_persons {
                let r = responses[[i, j]];
                if r >= 0 {
                    let w = posterior_weights[[i, q]];
                    r_jq += (r as f64) * w;
                    n_jq += w;
                }
            }

            let grad_contribution = a_j * (r_jq - n_jq * p);
            for (k, &f) in feature_row.iter().enumerate() {
                gradient[k] -= f * grad_contribution;
            }
        }
    }

    gradient.to_pyarray(py)
}

/// Compute Hessian of LLTM log-likelihood with respect to feature weights.
#[pyfunction]
pub fn lltm_hessian<'py>(
    py: Python<'py>,
    posterior_weights: PyReadonlyArray2<f64>,
    quad_points: PyReadonlyArray1<f64>,
    item_features: PyReadonlyArray2<f64>,
    feature_weights: PyReadonlyArray1<f64>,
    discrimination: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let posterior_weights = posterior_weights.as_array();
    let quad_points = quad_points.as_array();
    let item_features = item_features.as_array();
    let feature_weights = feature_weights.as_array();
    let discrimination = discrimination.as_array();

    let n_items = item_features.nrows();
    let n_quad = quad_points.len();
    let n_features = item_features.ncols();

    let difficulties = compute_difficulties(item_features, feature_weights);
    let n_k: Vec<f64> = posterior_weights.sum_axis(Axis(0)).to_vec();

    let mut hessian = Array2::<f64>::zeros((n_features, n_features));

    for j in 0..n_items {
        let a_j = discrimination[j];
        let b_j = difficulties[j];
        let feature_row = item_features.row(j);

        for q in 0..n_quad {
            let theta = quad_points[q];
            let z = a_j * (theta - b_j);
            let p = 1.0 / (1.0 + (-z).exp());
            let pq = p * (1.0 - p);

            let hess_weight = a_j * a_j * n_k[q] * pq;

            for (k1, &f1) in feature_row.iter().enumerate() {
                for (k2, &f2) in feature_row.iter().enumerate() {
                    hessian[[k1, k2]] -= f1 * f2 * hess_weight;
                }
            }
        }
    }

    hessian.to_pyarray(py)
}

/// Compute log-likelihoods with latent regression prior.
///
/// This computes the log-likelihood with a person-specific prior mean
/// based on covariates.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn latent_regression_log_likelihoods<'py>(
    py: Python<'py>,
    responses: PyReadonlyArray2<i32>,
    quad_points: PyReadonlyArray1<f64>,
    _quad_weights: PyReadonlyArray1<f64>,
    person_covariates: PyReadonlyArray2<f64>,
    regression_weights: PyReadonlyArray1<f64>,
    residual_variance: f64,
    discrimination: PyReadonlyArray1<f64>,
    difficulty: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let responses = responses.as_array();
    let quad_points = quad_points.as_array();
    let person_covariates = person_covariates.as_array();
    let regression_weights = regression_weights.as_array();
    let discrimination = discrimination.as_array();
    let difficulty = difficulty.as_array();

    let n_persons = responses.nrows();
    let n_items = responses.ncols();
    let n_quad = quad_points.len();

    let prior_means: Vec<f64> = (0..n_persons)
        .map(|i| person_covariates.row(i).dot(&regression_weights))
        .collect();

    let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let log_sigma = 0.5 * residual_variance.ln();

    let disc_vec: Vec<f64> = discrimination.to_vec();
    let diff_vec: Vec<f64> = difficulty.to_vec();

    let log_likes: Vec<Vec<f64>> = (0..n_persons)
        .into_par_iter()
        .map(|i| {
            let resp_row: Vec<i32> = responses.row(i).to_vec();
            let prior_mu = prior_means[i];

            (0..n_quad)
                .map(|q| {
                    let theta = quad_points[q];

                    let mut ll = 0.0;
                    for j in 0..n_items {
                        let r = resp_row[j];
                        if r >= 0 {
                            let z = disc_vec[j] * (theta - diff_vec[j]);
                            let log_p = log_sigmoid(z);
                            let log_1_minus_p = log_sigmoid(-z);
                            ll += (r as f64) * log_p + (1.0 - r as f64) * log_1_minus_p;
                        }
                    }

                    let theta_centered = theta - prior_mu;
                    let log_prior = -log_sqrt_2pi
                        - log_sigma
                        - 0.5 * theta_centered * theta_centered / residual_variance;

                    ll + log_prior
                })
                .collect()
        })
        .collect();

    let mut result = Array2::zeros((n_persons, n_quad));
    for (i, row) in log_likes.iter().enumerate() {
        for (q, &val) in row.iter().enumerate() {
            result[[i, q]] = val;
        }
    }

    result.to_pyarray(py)
}

/// Compute regression weight gradient for latent regression.
#[pyfunction]
pub fn latent_regression_gradient<'py>(
    py: Python<'py>,
    theta_eap: PyReadonlyArray1<f64>,
    person_covariates: PyReadonlyArray2<f64>,
    regression_weights: PyReadonlyArray1<f64>,
    residual_variance: f64,
) -> Bound<'py, PyArray1<f64>> {
    let theta_eap = theta_eap.as_array();
    let person_covariates = person_covariates.as_array();
    let regression_weights = regression_weights.as_array();

    let n_persons = person_covariates.nrows();
    let n_weights = regression_weights.len();

    let mut gradient = Array1::<f64>::zeros(n_weights);

    for i in 0..n_persons {
        let x_i = person_covariates.row(i);
        let mu_i = x_i.dot(&regression_weights);
        let residual = theta_eap[i] - mu_i;

        for k in 0..n_weights {
            gradient[k] += x_i[k] * residual / residual_variance;
        }
    }

    gradient.to_pyarray(py)
}

/// Compute regression weight update (closed-form solution).
#[pyfunction]
pub fn latent_regression_update<'py>(
    py: Python<'py>,
    theta_eap: PyReadonlyArray1<f64>,
    person_covariates: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray1<f64>>, f64) {
    let theta_eap = theta_eap.as_array();
    let person_covariates = person_covariates.as_array();

    let n_persons = person_covariates.nrows();
    let n_weights = person_covariates.ncols();

    let xtx = person_covariates.t().dot(&person_covariates);
    let xty = person_covariates.t().dot(&theta_eap);

    let beta = match ndarray_linalg_solve(&xtx, &xty) {
        Some(b) => b,
        None => {
            let lambda = 1e-6;
            let mut xtx_reg = xtx.clone();
            for i in 0..n_weights {
                xtx_reg[[i, i]] += lambda;
            }
            ndarray_linalg_solve(&xtx_reg, &xty).unwrap_or_else(|| Array1::zeros(n_weights))
        }
    };

    let predictions = person_covariates.dot(&beta);
    let residuals = &theta_eap - &predictions;
    let residual_variance = residuals.mapv(|x| x * x).sum() / (n_persons as f64);

    (beta.to_pyarray(py), residual_variance.max(1e-6))
}

fn ndarray_linalg_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return None;
    }

    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        for j in 0..=n {
            let tmp = aug[[i, j]];
            aug[[i, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = tmp;
        }

        if aug[[i, i]].abs() < 1e-12 {
            return None;
        }

        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Some(x)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lltm_log_likelihoods, m)?)?;
    m.add_function(wrap_pyfunction!(lltm_expected_feature_counts, m)?)?;
    m.add_function(wrap_pyfunction!(lltm_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(lltm_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(latent_regression_log_likelihoods, m)?)?;
    m.add_function(wrap_pyfunction!(latent_regression_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(latent_regression_update, m)?)?;
    Ok(())
}
